"""Does PI predict the damage of an attack ONE step ahead, asleep vs awake? (2026-10-16; R48 localization; GPU inference)

Events from the training pool (sleep_incidence_masks.pt: probe-asleep, HUD-reader health, mob-reader zombie adjacency; both frames
alive), 300 seeded transitions t -> t+1 per class:
  asleep_attacked   asleep at t, zombie adjacent at t, health drops >= 0.5 at t+1
  awake_attacked    awake at t, zombie adjacent at t, health drops >= 0.5
  asleep_adjacent_safe / awake_adjacent_safe   the same with no drop (false-alarm controls)
Each event is the last transition of a 16-frame window (frames t-14 .. t+1; start clamped to 0). PI u3000 (s1, s2), teacher-forced,
4 sampled one-step outputs per window. Readouts at t+1: P(predicted drop >= 0.5 below the real health at t), mean predicted health
change vs real, and the share of HUD cells PI edits; quantiles of the predicted and real health change per event and the spread
across draws (is the hedge a probability blend, a regression blur, or sampling?).
Env ONESTEP_ROW: place every event at that window row (default: the last transition, row 14 when possible; D-lineage worlds draw
false drops at row 14, the end-aligned-death cue, so health is judged at row 9). Env ONESTEP_WORLDS (dv16 world names; deterministic ones use reference(), HUD edit share then 1), ONESTEP_TAG (output suffix).
Usage: sleep_attack_onestep.py -> artifacts/eda/whiteboard/sleep_attack_onestep<tag>.json
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

N_EVENTS, DRAWS = 300, 4
WORLDS = tuple(os.environ.get('ONESTEP_WORLDS', 'pi_s1,pi_s2').split(','))
TAG = os.environ.get('ONESTEP_TAG', '')
ROW = int(os.environ['ONESTEP_ROW']) if os.environ.get('ONESTEP_ROW') else None


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
    s, z, h = m['asleep'][:, :-1], m['zombie_adjacent'][:, :-1], m['health'].float()
    ok = alive[:, :-1] & alive[:, 1:]
    drop = h[:, 1:] <= h[:, :-1] - .5
    classes = {'asleep_attacked': s & z & drop, 'awake_attacked': ~s & z & drop,
               'asleep_adjacent_safe': s & z & ~drop, 'awake_adjacent_safe': ~s & z & ~drop}
    g = torch.Generator().manual_seed(20261042)
    events = {}
    for k, mask in classes.items():
        idx = torch.nonzero(mask & ok)
        if ROW is not None:                                                                 # the event at window row ROW exactly
            idx = idx[(idx[:, 1] >= ROW) & (idx[:, 1] - ROW <= 48)]
        events[k] = idx[torch.randperm(len(idx), generator=g)[:N_EVENTS]]
    res = {'pool_counts': {k: int((v & ok).sum()) for k, v in classes.items()}}
    for wname in WORLDS:
        kind, ckpt, _ = DV.WORLDS[wname]
        world = DV.load_world(kind, ckpt)
        gn = torch.Generator().manual_seed(20261043)
        out = {}
        for k, ev in events.items():
            pred_drop, d_pred, d_real, hud_edit = [], [], [], []
            for b0 in range(0, len(ev), 4):
                chunk = ev[b0:b0 + 4]
                starts = (chunk[:, 1] - (14 if ROW is None else ROW)).clamp_min(0)
                fr = torch.stack([torch.from_numpy(np.array(tokens[int(r), int(st):int(st) + 16])) for (r, _), st in zip(chunk, starts)]).float()
                ac = torch.stack([actions[int(r), int(st):int(st) + 15] for (r, _), st in zip(chunk, starts)])
                pos = chunk[:, 1] - starts                                                       # index of frame t in the window
                fr = fr.repeat_interleave(DRAWS, 0).cuda(); ac = F.pad(ac, (0, 1)).repeat_interleave(DRAWS, 0).cuda()
                pos_r = pos.repeat_interleave(DRAWS)
                with autocast_context(config):
                    belief, ha = world.backbone_full(fr, ac)
                    noise = DM.draw_noise(fr.shape[:2], gn)[0].cuda(); gum = TM.draw_gumbel(fr.shape[:2], gn)[0].cuda()
                    if kind == 'pi':
                        o, aux = world.complete(fr, ac, belief, ha, noise, gum)
                    else:                                                                   # deterministic references (R51)
                        o = F.layer_norm(world.reference(fr, ac, belief, ha).float(), (192,))
                        aux = {'edits': torch.ones(o.shape[:3], dtype=torch.bool, device=o.device)}
                ar = torch.arange(len(fr))
                nxt = o[ar, pos_r.cuda()].float().cpu()                                          # predicted frame t+1
                hp = hud_health(nxt, bank['heads']['hud'])
                ht = torch.stack([h[int(r), int(t)] for r, t in chunk]).repeat_interleave(DRAWS)
                hr = torch.stack([h[int(r), int(t) + 1] for r, t in chunk]).repeat_interleave(DRAWS)
                pred_drop.append(hp <= ht - .5); d_pred.append(hp - ht); d_real.append(hr - ht)
                hud_edit.append(aux['edits'][ar, pos_r.cuda(), 63:].float().mean(-1).cpu())
            pd, dp, dr, he = (torch.cat(x).float() for x in (pred_drop, d_pred, d_real, hud_edit))
            q = torch.tensor([.1, .25, .5, .75, .9])
            out[k] = {'events': len(ev), 'P_predicted_drop': round(float(pd.mean()), 4), 'mean_pred_health_change': round(float(dp.mean()), 3),
                      'mean_real_health_change': round(float(dr.mean()), 3), 'hud_edit_share': round(float(he.mean()), 4),
                      'pred_change_quantiles_10_25_50_75_90': [round(float(v), 2) for v in dp.quantile(q)],
                      'real_change_quantiles_10_25_50_75_90': [round(float(v), 2) for v in dr.quantile(q)],
                      'pred_change_draw_spread': round(float(dp.view(-1, DRAWS).std(1).mean()), 3),
                      'pred_minus_real_abs_median': round(float((dp - dr).abs().median()), 2)}
        res[wname] = out
        print(json.dumps({wname: out}, indent=1), flush=True)
        del world; torch.cuda.empty_cache()
    (DV.OUT / f'sleep_attack_onestep{TAG}.json').write_text(json.dumps(res, indent=1))


if __name__ == '__main__':
    main()
