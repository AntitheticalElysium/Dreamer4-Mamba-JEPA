"""R27 follow-up (2026-10-15 night, GPU inference; layouts are diagnostic only): is PI's chance head less directional than IC's
on REAL contexts?

Seeded training windows with layouts (as ic_posterior.py milestones, seed 20261016, 400 windows), teacher-forced. For every
zombie in the afterstate layout (the root layout under the label-free scroll estimate) at an interior cell x, the chance head's
canonical pair probabilities p(x, y) = sigmoid(pair_prior_logit) with its interior neighbours y are split by whether y is
closer to the player cell (31, Manhattan) than x (TOWARD), farther (AWAY) or equal. Real outcome from the next layout
(transport_stage2.swaps, type zombie): moved toward / away / stayed. Reported by distance to the player (1-2, 3-4, 5+):
  model_toward, model_away   mean summed pair probability
  real_toward, real_away     observed move rates
  share_toward               toward / (toward + away), model vs real (direction bias independent of overall move rate)
Models: IC v3 (ic_seed<s>/ic_u3000.pt) and PI (pi_seed<s>/pi_u3000.pt), both seeds.
Usage: pursuit_onestep.py -> artifacts/eda/whiteboard/pursuit_onestep.json
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

import audit_s5_direct as A5
import direct_model as DM
import direct_train as BASE
import transport_ic_train as IC
import transport_model as TM
import transport_stage2 as S2
import pi_model as PI
from d4mj.train import autocast_context

OUT = HERE.parents[1] / 'eda/whiteboard'
T = A5.BASE / 'transport'
INTERIOR = set(r * 9 + c for r in range(1, 6) for c in range(1, 8))


def dist(x):
    r, c = divmod(x, 9)
    return abs(r - 3) + abs(c - 4)


def band(d):
    return '1-2' if d <= 2 else ('3-4' if d <= 4 else '5+')


@torch.no_grad()
def main():
    config = BASE.settings()
    _, windows, actions, tokens = BASE.load_data()
    valid = torch.from_numpy(np.load(A5.BASE / 'layouts_valid.npy'))
    windows = windows[valid[windows[:, 0]]]
    sel = torch.randperm(len(windows), generator=torch.Generator().manual_seed(20261016))[:400]
    lay = np.load(A5.BASE / 'layouts.npy', mmap_mode='r')
    scroll = A5.load_scroll()
    pair_index = {p: i for i, p in enumerate(IC.PAIRS)}
    res = {}
    for name, path, kind in (('ic_s1', T / 'ic_seed1/ic_u3000.pt', 'ic'), ('pi_s1', T / 'pi_seed1/pi_u3000.pt', 'pi'),
                             ('ic_s2', T / 'ic_seed2/ic_u3000.pt', 'ic'), ('pi_s2', T / 'pi_seed2/pi_u3000.pt', 'pi')):
        world = (PI.PIWorld() if kind == 'pi' else TM.TransportWorld('operator')).cuda().eval()
        world.load_state_dict(torch.load(path, map_location='cpu', weights_only=False)['world'])
        g = torch.Generator().manual_seed(1)
        acc = {b: {'n': 0, 'model_toward': 0., 'model_away': 0., 'real_toward': 0, 'real_away': 0} for b in ('1-2', '3-4', '5+')}
        for b0 in range(0, 400, 8):
            frames, ac, choices = BASE.batch(tokens, actions, windows, sel[b0:b0 + 8])
            noise = DM.draw_noise(frames.shape[:2], g)[0].cuda()
            with autocast_context(config):
                belief, _ = world.backbone_full(frames, ac)
                logits = world.completion.propose(world.completion.trunk(belief, noise)).float()[:, :15]
            logits = logits.masked_fill(~world.completion.valid, -torch.inf)
            prob = torch.sigmoid(IC.pair_prior_logit(logits)).cpu()                       # [B,15,P]
            est = scroll.estimate(frames[:, :-1].float().cpu(), frames[:, 1:].float().cpu())
            for i, (row, start) in enumerate(choices.tolist()):
                L = torch.from_numpy(np.array(lay[row, start:start + 16]) > 0)
                for t in range(15):
                    prev = S2.afterstate_layout(L[t], *scroll.SHIFTS[int(est[i, t])])
                    moves = {s_: d_ for s_, d_, ty in S2.swaps(prev, L[t + 1]) if ty == 0}
                    for x in range(63):
                        if x not in INTERIOR or x == 31 or int(prev[x, 0]) != 1:
                            continue
                        a = acc[band(dist(x))]
                        a['n'] += 1
                        for y in (x - 9, x + 9, x - 1, x + 1):
                            pr = (min(x, y), max(x, y))
                            if pr not in pair_index:
                                continue
                            p = float(prob[i, t, pair_index[pr]])
                            if dist(y) < dist(x):
                                a['model_toward'] += p
                            elif dist(y) > dist(x):
                                a['model_away'] += p
                        if x in moves:
                            y = moves[x]
                            a['real_toward' if dist(y) < dist(x) else 'real_away'] += int(dist(y) != dist(x))
        out = {}
        for b, a in acc.items():
            n = max(a['n'], 1)
            mt, ma, rt, ra = a['model_toward'] / n, a['model_away'] / n, a['real_toward'] / n, a['real_away'] / n
            out[b] = {'zombies': a['n'], 'model_toward': round(mt, 4), 'model_away': round(ma, 4), 'real_toward': round(rt, 4),
                      'real_away': round(ra, 4), 'model_share_toward': round(mt / max(mt + ma, 1e-9), 3),
                      'real_share_toward': round(rt / max(rt + ra, 1e-9), 3)}
        res[name] = out
        print(json.dumps({name: out}), flush=True)
        del world; torch.cuda.empty_cache()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'pursuit_onestep.json').write_text(json.dumps(res, indent=1))


if __name__ == '__main__':
    main()
