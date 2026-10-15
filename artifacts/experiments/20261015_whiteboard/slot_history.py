"""Does PI's belief keep an object's temporal context when the object changes screen slot? (2026-10-15 night, user question;
GPU inference, no training; layouts are diagnostic only)

Zombie hits follow a hidden cooldown (10-03), so predicting the next health drop needs the adjacent zombie's own history. PI's
backbone runs a Mamba per SCREEN slot: when a zombie moves or the camera scrolls, it lands in a slot whose history belongs to
whatever was there before. Teacher-forced on seeded training windows with layouts (real context, no imagination), at every frame
t with a zombie in a player-adjacent cell (22, 30, 32, 40):
  history   the number of steps every adjacent zombie has held its slot (same layout cell, no camera scroll); groups
            changed (0-1 steps: it arrived in its slot within the last step or two), stayed 2-3, stayed 4+
Label: real health drop t -> t+1 (bank HUD reader, <= -0.5 units, as decision_proxies). PI's consequence predictions at t from
its belief: gate = max sigmoid(gate) over the 18 HUD tokens, and at the health token
alone (the HUD token whose real change norm best separates drops from non-drops on these frames, chosen before scoring); gen_drop = health(f_t) - health(LN(gen HUD at t)). AUC per group,
base rates; also cells adjacent zombies at t whose slot history is longer (stayed >= 4 steps) for a dose reading.
Usage: slot_history.py -> artifacts/eda/whiteboard/slot_history.json
"""
import json
import os
import sys
from pathlib import Path

os.environ.setdefault('TRITON_F32_DEFAULT', 'ieee')
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / '20261012_oracle'))
sys.path.insert(0, str(HERE))

import numpy as np
import torch
from torch.nn import functional as F

import audit_s5_direct as A5
import direct_model as DM
import direct_train as BASE
import pi_model as PI
import rep_layers as RL
from decision_proxies import hud_health
from d4mj.train import autocast_context

OUT = HERE.parents[1] / 'eda/whiteboard'
T = A5.BASE / 'transport'
N_WINDOWS = int(os.environ.get('SLOT_WINDOWS', 1600))
ADJ = (22, 30, 32, 40)


@torch.no_grad()
def main():
    config = BASE.settings()
    _, windows, actions, tokens = BASE.load_data()
    valid = torch.from_numpy(np.load(A5.BASE / 'layouts_valid.npy'))
    windows = windows[valid[windows[:, 0]]]
    sel = torch.randperm(len(windows), generator=torch.Generator().manual_seed(20261024))[:N_WINDOWS]
    lay = np.load(A5.BASE / 'layouts.npy', mmap_mode='r')
    bank = torch.load(A5.BASE / 'snap_bank_readers.pt', map_location='cpu', weights_only=False)
    hhead = bank['heads']['hud']
    scroll = A5.load_scroll()
    res = {}
    for seed in (1, 2):
        w = PI.PIWorld().cuda().eval()
        w.load_state_dict(torch.load(T / f'pi_seed{seed}/pi_u3000.pt', map_location='cpu', weights_only=False)['world'])
        rows = []                                                                   # (group, history_len, drop, gate, gen_drop)
        for b0 in range(0, N_WINDOWS, 8):
            frames, ac, choices = BASE.batch(tokens, actions, windows, sel[b0:b0 + 8])
            est = scroll.estimate(frames[:, :-1].float(), frames[:, 1:].float())
            with autocast_context(config):
                belief, ha = w.backbone_full(frames, ac)
                _, gen, glogit, _, _ = w.afterstate_parts(frames, ac, belief, ha, scroll=F.pad(est, (0, 1)))
            gen, glogit = gen.float().cpu(), glogit.float().cpu()
            fr = frames.float().cpu()
            health = hud_health(fr, hhead)                                          # [B,16]
            gen_health = hud_health(torch.cat([fr[..., :63, :], F.layer_norm(gen[..., 63:, :], (192,))], -2), hhead)
            est = est.cpu()
            for i, (row, start) in enumerate(choices.tolist()):
                L = torch.from_numpy(np.array(lay[row, start:start + 16]) > 0)[..., 0]     # [16,63] zombie
                for t in range(4, 15):
                    adj = [x for x in ADJ if bool(L[t, x])]
                    if not adj:
                        continue
                    hist = []
                    for x in adj:
                        k = 0
                        while k < t and bool(L[t - k - 1, x]) and int(est[i, t - k - 1]) == 0:
                            k += 1
                        hist.append(k)                                              # steps the zombie has held this slot
                    h = min(hist)
                    drop = float(health[i, t + 1] - health[i, t]) <= -.5
                    rows.append((h, drop, torch.sigmoid(glogit[i, t, 63:]), float(health[i, t] - gen_health[i, t]),
                                 (fr[i, t + 1, 63:] - fr[i, t, 63:]).norm(dim=-1)))
            if b0 % 400 == 0:
                print(json.dumps({'seed': seed, 'windows_done': b0 + 8, 'events': len(rows)}), flush=True)
        h = torch.tensor([r[0] for r in rows]); y = torch.tensor([r[1] for r in rows])
        gates = torch.stack([r[2] for r in rows]); gd = torch.tensor([r[3] for r in rows]); dn = torch.stack([r[4] for r in rows])
        tok = int((dn[y].mean(0) - dn[~y].mean(0)).argmax())                      # the HUD token whose real change tracks drops
        gate = gates.max(-1).values; gate_h = gates[:, tok]
        out = {'health_token': 63 + tok}
        for name, m in (('changed_0-1', h <= 1), ('stayed_2-3', (h >= 2) & (h <= 3)), ('stayed_4+', h >= 4), ('all', h >= 0)):
            if m.sum() == 0:
                continue
            auc = lambda sc: None if RL.auc(sc[m], y[m]) is None else round(float(RL.auc(sc[m], y[m])), 4)
            out[name] = {'frames': int(m.sum()), 'drop_rate': round(float(y[m].float().mean()), 4), 'auc_gate_maxhud': auc(gate),
                         'auc_gate_health_token': auc(gate_h), 'auc_gen_drop': auc(gd),
                         'gate_mean_on_drops': round(float(gate[m & y].mean()), 4) if (m & y).any() else None,
                         'gen_drop_mean_on_drops': round(float(gd[m & y].mean()), 3) if (m & y).any() else None}
        res[f'seed{seed}'] = out
        print(json.dumps({f'seed{seed}': out}), flush=True)
        del w; torch.cuda.empty_cache()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'slot_history.json').write_text(json.dumps(res, indent=1))


if __name__ == '__main__':
    main()
