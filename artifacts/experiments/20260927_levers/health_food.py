"""Why does D's imagined food degrade? (2026-10-09 03:00; E21 D: food R2 at depth 16 0.880 -> 0.582.)
Hypothesis under test: food decrements are rare, timer-driven discrete changes; a mean-seeking (L2) world predicts a fraction of a
decrement wherever the timer phase is uncertain; teacher-forced that is small, self-fed it compounds over 16 steps.
On teval's held roots and paths (Raw diagnosis cache, window 15): HUD readings (teval Probes.hud x 9: column 0 health, 1 food as
teval's food_r2; columns 2, 3 reported unnamed) of
  tf     each step's teacher-forced prediction (true history): change predicted - true current, on steps where the TRUE value is
         unchanged (bias = a fractional drift per step) and on steps where it changes
  gen    the self-fed rollout: imagined - true value per depth (alive roots)
for each world. -> artifacts/eda/health_chain_v1/food_<tag>.json
Usage: health_food.py <tag> <name=world.pt> ...
"""
import json
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
ROOT = HERE.parents[2]
OUT = ROOT / 'artifacts/eda/health_chain_v1'
COLUMNS = {'health': 0, 'food': 1, 'hud2': 2, 'hud3': 3}          # teval's facts: hud[0] health, hud[1] food (food_r2); 2, 3 unnamed


@torch.no_grad()
def main():
    import teval as T
    from d4mj.config import config_from_dict
    import spatial as S
    tag, worlds = sys.argv[1], dict(a.split('=', 1) for a in sys.argv[2:])
    dev = torch.device('cuda')
    config = config_from_dict(torch.load(S.CHECKPOINT, map_location='cpu', weights_only=False)['config'])
    meta, train_roots, train_seeds = T.split()
    cache = T.build_cache('raw', torch.device('cpu'))
    P = T.Probes(cache, meta, train_roots, train_seeds)
    read = lambda x: P.hud(x[:, 63:81].flatten(1).float()) * 9                       # [n, columns]
    R, H = len(meta['seed']), cache['fut'].shape[1]
    alive = ~meta['future_dead'][:, 0].cumsum(1).bool()
    res = {}
    for name, path in worlds.items():
        w, _ = T.load_world(Path(path), dev)
        tf_change, true_change, gen_err = [], [], torch.zeros(R, H, 4)
        for i in range(0, R, 16):
            ctx, ca = cache['ctx'][i:i + 16].float(), cache['ctx_a'][i:i + 16]
            fa, fut = cache['fut_a'][i:i + 16], cache['fut'].float()[i:i + 16]
            t_frames, g_frames, a_hist = [ctx[:, j] for j in range(4)], [ctx[:, j] for j in range(4)], [ca[:, j] for j in range(3)]
            for k in range(H):
                wk = 4 if k == 0 else 15
                a = torch.stack(a_hist[-(wk - 1):] + [fa[:, k]], 1)
                t = T.step(w, torch.stack(t_frames[-wk:], 1), a, dev, config)
                g = T.step(w, torch.stack(g_frames[-wk:], 1), a, dev, config)
                cur, nxt = read(t_frames[-1]), read(fut[:, k])
                m = alive[i:i + 16, k]
                tf_change.append((read(t) - cur)[m]); true_change.append((nxt - cur)[m])
                gen_err[i:i + 16, k] = (read(g) - nxt)[:, :4]
                t_frames.append(fut[:, k]); g_frames.append(g); a_hist.append(fa[:, k])
        tfc, trc = torch.cat(tf_change)[:, :4], torch.cat(true_change)[:, :4]
        r = {}
        for col, j in COLUMNS.items():
            unch = trc[:, j].abs() < 0.5
            r[col] = {'tf_change_on_unchanged_mean': round(float(tfc[unch, j].mean()), 4),
                      'tf_change_on_unchanged_abs': round(float(tfc[unch, j].abs().mean()), 4),
                      'tf_change_on_changes_mean': round(float(tfc[~unch, j].mean()), 4),
                      'true_change_on_changes_mean': round(float(trc[~unch, j].mean()), 4),
                      'share_changing': round(float((~unch).float().mean()), 4),
                      'gen_bias_by_depth': [round(float(gen_err[alive[:, k], k, j].mean()), 3) for k in (0, 3, 7, 15)],
                      'gen_abs_by_depth': [round(float(gen_err[alive[:, k], k, j].abs().mean()), 3) for k in (0, 3, 7, 15)]}
        res[name] = r
        print(json.dumps({name: r}), flush=True)
    (OUT / f'food_{tag}.json').write_text(json.dumps(res, indent=1))


if __name__ == '__main__':
    sys.path.insert(0, str(ROOT / 'artifacts/experiments/20260926_diagnosis'))
    sys.path.insert(0, str(ROOT / 'artifacts/experiments/20260921_readout_ladder'))
    main()
