"""10-10 PROGRAM C3 (axis 3, imagination stability): how sensitive is a deterministic world's self-fed rollout to a tiny change of its
input? (Observed today: M16's rollouts differ at depth 16 for 986 / 1,002 roots between two numerically different runs of the SAME
input; mean relative error 0.5%, max 2,107.) GPU.
Per world: teval's self-fed rollout of the factual 16 actions (rollout_export.py's code path and batch; the unperturbed rollout must
equal the saved export bitwise -- checked), and the same rollout with the ROOT frame's tokens perturbed, x + sigma * N(0, I) then
layer-normed (tokens are layer-normed), sigma in 0.001 / 0.01 / 0.1 (generator seed 20261010). Per depth (alive sample 0):
  divergence      mean |g'_k - g_k|^2 / V; the initial perturbation's size |x' - x|^2 / V for reference
  scroll_differs  share of roots whose imagined scroll class (scroll.estimate) differs from the unperturbed rollout's at step k,
                  and share whose cumulative view offset differs
  tile_differs    share of map cells whose probed tile class differs (scoreboard.AllProbes, TEST roots)
  growth          median over roots of log(|g'_k - g_k|^2 / |x' - x|^2) / k (a Lyapunov-like rate per step)
Usage: chaos.py <variant=world.pt> ...  -> artifacts/eda/scoreboard_v1/<variant>/chaos.json
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE))
OUT = ROOT / 'artifacts/eda/scoreboard_v1'
H = 16
SIGMAS = (0.001, 0.01, 0.1)


def main():
    import teval as T
    import driftanat as DA
    import scoreboard as SB
    from d4mj.config import config_from_dict
    from scroll import estimate
    import spatial as Sp
    dev = torch.device('cuda')
    config = config_from_dict(torch.load(Sp.CHECKPOINT, map_location='cpu', weights_only=False)['config'])
    meta, train_roots, train_seeds = T.split()
    test = ~train_roots
    cache = T.build_cache('raw', torch.device('cpu'))
    R = len(meta['seed'])
    ctx, ca, fa, fut = cache['ctx'], cache['ctx_a'], cache['fut_a'], cache['fut']
    V = float(sum(((fut[i:i + 64].float() - fut.float().flatten(0, 1).mean(0)) ** 2).sum((-1, -2)).sum() for i in range(0, R, 64)) / (R * H))
    alive = ~meta['future_dead'][:, 0].cumsum(1).bool()
    probes = SB.AllProbes(T, cache, meta, train_roots, train_seeds)
    for arg in sys.argv[1:]:
        name, path = arg.split('=', 1)
        world, st = T.load_world(Path(path), dev)
        cap = world.time.shape[0] - 1
        bs = 4 if st['args'].get('backbone') in ('fmamba', 'fcanvas') else 16

        def rollout(root_tok):
            gen = torch.empty(R, H, 81, 192, dtype=torch.float16)
            with torch.no_grad():
                for i in range(0, R, bs):
                    b = min(bs, R - i)
                    c4 = ctx[i:i + b].float().clone()
                    c4[:, -1] = root_tok[i:i + b]
                    frames, hist = [c4[:, j] for j in range(4)], [ca[i:i + b, j] for j in range(3)]
                    for k in range(H):
                        w = 4 if k == 0 else cap
                        g = T.step(world, torch.stack(frames[-w:], 1), torch.stack(hist[-(w - 1):] + [fa[i:i + b, k]], 1), dev, config)
                        gen[i:i + b, k] = g.half(); frames.append(g); hist.append(fa[i:i + b, k])
            return gen
        base = rollout(ctx[:, -1].float())
        saved = OUT / name / 'gen.npy'
        res = {'variant': name, 'world': st['name'], 'V': V}
        if saved.exists():
            res['unperturbed_equals_export'] = bool(torch.equal(base, torch.from_numpy(np.load(saved, mmap_mode='r'))))
        prev = lambda g: torch.cat([ctx[:, -1:].float(), g[:, :-1].float()], 1)
        sh0 = torch.cat([estimate(prev(base)[i:i + 32].flatten(0, 1), base[i:i + 32].float().flatten(0, 1)).view(-1, H) for i in range(0, R, 32)])
        tile0 = torch.stack([probes.tile(base[:, k, SB.MAPC].float().flatten(0, 1)).view(R, 62, 17).argmax(-1) for k in range(H)], 1)
        gen_ = torch.Generator().manual_seed(20261010)
        for sigma in SIGMAS:
            x = ctx[:, -1].float()
            xp = F.layer_norm(x + sigma * torch.randn(x.shape, generator=gen_), (192,))
            d0 = ((xp - x) ** 2).sum((-1, -2))                                                         # [R]
            pert = rollout(xp)
            dk = torch.stack([((pert[:, k].float() - base[:, k].float()) ** 2).sum((-1, -2)) for k in range(H)], 1)   # [R,16]
            sh1 = torch.cat([estimate(prev(pert)[i:i + 32].flatten(0, 1), pert[i:i + 32].float().flatten(0, 1)).view(-1, H) for i in range(0, R, 32)])
            off_diff = (DA.offsets(sh1) != DA.offsets(sh0)).any(-1)
            tile1 = torch.stack([probes.tile(pert[:, k, SB.MAPC].float().flatten(0, 1)).view(R, 62, 17).argmax(-1) for k in range(H)], 1)
            rate = (torch.log(dk.clamp(min=1e-12) / d0[:, None].clamp(min=1e-12)) / torch.arange(1, H + 1)[None])
            row = {'initial_over_V': round(float(d0.mean() / V), 7)}
            for k in (1, 2, 4, 8, 16):
                m = alive[:, k - 1]
                row[str(k)] = {'divergence_over_V': round(float(dk[m, k - 1].mean() / V), 6),
                               'scroll_differs': round(float((sh1 != sh0)[m, k - 1].float().mean()), 4),
                               'offset_differs': round(float(off_diff[m, k - 1].float().mean()), 4),
                               'tile_differs_test': round(float((tile1 != tile0)[m & test, k - 1].float().mean()), 4),
                               'growth_rate_median': round(float(rate[m, k - 1].median()), 4)}
            res[f'sigma_{sigma}'] = row
            print(json.dumps({name: {f'sigma_{sigma}': row}}), flush=True)
        (OUT / name / 'chaos.json').write_text(json.dumps(res, indent=1))
        del world; torch.cuda.empty_cache()


if __name__ == '__main__':
    sys.path.insert(0, str(ROOT / 'artifacts/experiments/20260926_diagnosis'))
    sys.path.insert(0, str(ROOT / 'artifacts/experiments/20260921_readout_ladder'))
    main()
