"""Does PI track the zombie's hidden attack cooldown? (2026-10-16; R48 localization; GPU inference; readers as labels)

Source (game_logic.py:815-845): an adjacent zombie attacks when its cooldown <= 0; an attack resets it to 5 and deals 7 to a
sleeper (2 awake) and wakes the sleeper. The cooldown is hidden but readable from history: frames since the last health drop.
Training pool (sleep_incidence_masks.pt): transitions t -> t+1 with a zombie adjacent at t, split by
  state      asleep for 1-5 frames ("early sleep") / awake
  history    frames since the last health drop: recent (0-5, cooldown likely active) / expired (>= 6)
Real P(drop at t+1) per cell, and PI's (s1, s2; teacher-forced, 4 sampled one-step outputs, the event as the last transition of a
16-frame window) P(predicted drop >= 0.5). All early-sleep-expired events (rare), up to 300 per other cell.
Also, on the DV16 roots: frames since the last health drop in the 8-frame context (root health history, HUD reader).
Usage: sleep_cooldown.py -> artifacts/eda/whiteboard/sleep_cooldown.json
"""
import json
import os
import sys
from pathlib import Path

os.environ.setdefault('TRITON_F32_DEFAULT', 'ieee')
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / '20261012_oracle'))
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / '20260927_levers'))

import numpy as np
import torch
from torch.nn import functional as F

import dv16 as DV
from decision_proxies import hud_health

CAP, DRAWS = 300, 4


@torch.no_grad()
def main():
    import direct_train as BASE
    import direct_model as DM
    import transport_model as TM
    import audit_s5_direct as A5
    from d4mj.train import autocast_context
    config = BASE.settings()
    bank = torch.load(A5.BASE / 'snap_bank_readers.pt', map_location='cpu', weights_only=False)
    m = torch.load(DV.OUT / 'sleep_incidence_masks.pt', weights_only=False)
    _, _, actions, tokens = BASE.load_data()
    alive = torch.load(BASE.D.POOL / 'labels.pt', map_location='cpu', weights_only=False, mmap=True)['alive']
    a, h, z = m['asleep'], m['health'].float(), m['zombie_adjacent']
    N, T = a.shape
    ok = alive[:, :-1] & alive[:, 1:]
    drop = h[:, 1:] <= h[:, :-1] - .5
    age = torch.zeros(N, T); since = torch.full((N, T), 99.)
    run = torch.zeros(N); last = torch.full((N,), -99.)
    for t in range(T):
        run = torch.where(a[:, t], run + 1, torch.zeros_like(run)); age[:, t] = run
        if t > 0:
            last = torch.where(drop[:, t - 1], torch.full_like(last, float(t)), last)
        since[:, t] = t - last
    age, since = age[:, :-1], since[:, :-1]
    base = z[:, :-1] & ok
    early = (age >= 1) & (age <= 5)
    cells = {'early_sleep_recent': base & early & (since <= 5), 'early_sleep_expired': base & early & (since >= 6),
             'awake_recent': base & (age == 0) & (since <= 5), 'awake_expired': base & (age == 0) & (since >= 6)}
    g = torch.Generator().manual_seed(20261044)
    events, res = {}, {'real': {}}
    for k, mask in cells.items():
        idx = torch.nonzero(mask)
        res['real'][k] = {'transitions': len(idx), 'P_drop': round(float(drop[mask].float().mean()), 4)}
        events[k] = idx[torch.randperm(len(idx), generator=g)[:CAP]]
    for wname in ('pi_s1', 'pi_s2'):
        world = DV.load_world(*DV.WORLDS[wname][:2])
        gn = torch.Generator().manual_seed(20261045)
        out = {}
        for k, ev in events.items():
            pd, rd = [], []
            for b0 in range(0, len(ev), 4):
                chunk = ev[b0:b0 + 4]
                starts = (chunk[:, 1] - 14).clamp_min(0)
                fr = torch.stack([torch.from_numpy(np.array(tokens[int(r), int(st):int(st) + 16])) for (r, _), st in zip(chunk, starts)]).float()
                ac = torch.stack([actions[int(r), int(st):int(st) + 15] for (r, _), st in zip(chunk, starts)])
                pos = (chunk[:, 1] - starts).repeat_interleave(DRAWS)
                fr = fr.repeat_interleave(DRAWS, 0).cuda(); ac = F.pad(ac, (0, 1)).repeat_interleave(DRAWS, 0).cuda()
                with autocast_context(config):
                    belief, ha = world.backbone_full(fr, ac)
                    noise = DM.draw_noise(fr.shape[:2], gn)[0].cuda(); gum = TM.draw_gumbel(fr.shape[:2], gn)[0].cuda()
                    o, _ = world.complete(fr, ac, belief, ha, noise, gum)
                hp = hud_health(o[torch.arange(len(fr)), pos.cuda()].float().cpu(), bank['heads']['hud'])
                ht = torch.stack([h[int(r), int(t)] for r, t in chunk]).repeat_interleave(DRAWS)
                pd.append(hp <= ht - .5); rd.append(torch.stack([drop[int(r), int(t)] for r, t in chunk]).repeat_interleave(DRAWS))
            pd, rd = torch.cat(pd).float(), torch.cat(rd).float()
            out[k] = {'events': len(ev), 'P_real_drop_sampled': round(float(rd.mean()), 4), 'P_predicted_drop': round(float(pd.mean()), 4),
                      'P_predicted_drop_given_real_drop': round(float(pd[rd > 0].mean()), 4) if (rd > 0).any() else None,
                      'P_predicted_drop_given_no_drop': round(float(pd[rd == 0].mean()), 4) if (rd == 0).any() else None}
        res[wname] = out
        print(json.dumps({wname: out}, indent=1), flush=True)
        del world; torch.cuda.empty_cache()
    enc = DV.RS.Encoder()
    roots, _ = DV.load_roots()
    hist = []
    for r in roots:
        hc = hud_health(enc.recontext(enc.content(r['frames'])).float().cpu(), bank['heads']['hud'])            # [8]
        d = (hc[1:] <= hc[:-1] - .5).nonzero()
        hist.append(99 if len(d) == 0 else 7 - int(d[-1]))                                                       # frames since
    hist = torch.tensor(hist)
    res['dv16_roots_frames_since_last_drop'] = {'recent_0_5': int((hist <= 5).sum()), 'expired_6_plus_or_none': int((hist >= 6).sum()),
                                                'none_in_context': int((hist == 99).sum())}
    print(json.dumps(res['dv16_roots_frames_since_last_drop']))
    (DV.OUT / 'sleep_cooldown.json').write_text(json.dumps(res, indent=1))


if __name__ == '__main__':
    main()
