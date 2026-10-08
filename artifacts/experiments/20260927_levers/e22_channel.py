"""E22 channel diagnostics (2026-10-09 10:05; NOTEBOOK "Substitution test of the warm start", predeclared 09:58). Is the Delta
channel of a dworld stage-A file used, and if not, which side is silent? On held 16-frame windows (teval's Raw diagnosis cache:
the 4 context frames + the first 12 true future frames with their actions; roots alive through them), batches of 8 in root order,
posterior and world in eval mode (no EMA, no revival), training's autocast and teacher loss (the file's --recon):
  gain       teacher loss with Delta zeroed - with the posterior's Delta (same weights, same windows)
  cond       the decoder's gradient on the Delta projection (cond.weight): mean norm and consistency
             ||sum_b g_b|| / sum_b ||g_b|| (1 = one direction in every batch; about 1 / sqrt(batches) = noise)
  encoder    gradient on the posterior encoder (conv net, action plane, quantizer pre-projection) from the teacher term (through
             the straight-through code) and from the commitment term: mean norms, their ratio, consistency of each
  explained  per region (Posterior.encode's 4 codes = 5 x 5 blocks of the 10 x 10 padded token grid), the share of the
             zeroed-Delta residual variance (true next frame - prediction with Delta zeroed, over the region's tiles) explained by
             the region's code: between-code sum of squares / total sum of squares; chance under a random assignment of codes
             with the same counts = (codes used - 1) / (n - 1)
Usage: e22_channel.py <name=dworld_a.pt> ...  -> artifacts/eda/health_chain_v1/e22_channel.json
"""
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
ROOT = HERE.parents[2]
OUT = ROOT / 'artifacts/eda/health_chain_v1'
BATCH = 8


def main():
    import teval as T
    import dworld as DW
    import tworld as TW
    from d4mj.config import config_from_dict
    from d4mj.train import autocast_context
    config = config_from_dict(torch.load(TW.S.CHECKPOINT, map_location='cpu', weights_only=False)['config'])
    dev = torch.device('cuda')
    meta, _, _ = T.split()
    cache = T.build_cache('raw', torch.device('cpu'))
    keep = torch.where(~meta['future_dead'][:, 0].cumsum(1).bool()[:, 11])[0]
    s_all = torch.cat([cache['ctx'], cache['fut'][:, :12]], 1)[keep]                       # [n, 16, 81, 192]
    a_all = torch.cat([cache['ctx_a'], cache['fut_a'][:, :12]], 1)[keep]                   # [n, 15]
    cells = torch.arange(81).view(9, 9)
    regions = [cells[5 * i:5 * i + 5, 5 * j:5 * j + 5].flatten().to(dev) for i in range(2) for j in range(2)]
    res = {}
    for name, path in (x.split('=', 1) for x in sys.argv[1:]):
        world, post, st = DW.load_a(Path(path), dev)
        recon = st['args'].get('recon', 'l1')
        enc = [p for m in (post.net, post.action, post.quantizer.pre) for p in m.parameters()]
        loss = {'delta': 0.0, 'zeroed': 0.0}
        acc = {k: [0, 0.0] for k in ('cond', 'enc_teacher', 'enc_commit')}                # [sum of gradients, sum of norms]
        sums = [torch.zeros(DW.CODES, len(r) * TW.S.WIDTH, dtype=torch.float64, device=dev) for r in regions]
        counts = [torch.zeros(DW.CODES, dtype=torch.float64, device=dev) for _ in regions]
        ss = [0.0 for _ in regions]
        torch.cuda.reset_peak_memory_stats()
        nb = 0
        for i in range(0, len(keep), BATCH):
            s, a = s_all[i:i + BATCH].to(dev).float(), a_all[i:i + BATCH].to(dev)
            with autocast_context(config):
                cond, tokens, commit = post(s, a)
                world.delta = cond
                teacher = TW.rollout_losses(world, s, a, 'teacher', recon=recon)
                world.delta = None
            gt = torch.autograd.grad(teacher, [post.cond.weight] + enc, retain_graph=True, allow_unused=True)
            gc = torch.autograd.grad(commit, enc, allow_unused=True)
            flat = lambda gs, ps: torch.cat([(g if g is not None else torch.zeros_like(p)).flatten().float() for g, p in zip(gs, ps)])
            for k, g in (('cond', gt[0].flatten().float()), ('enc_teacher', flat(gt[1:], enc)), ('enc_commit', flat(gc, enc))):
                acc[k][0] = acc[k][0] + g
                acc[k][1] += float(g.norm())
            with torch.no_grad(), autocast_context(config):
                zeroed = TW.rollout_losses(world, s, a, 'teacher', recon=recon)
                pred = world(s, F.pad(a, (0, 1)))[0]
            loss['delta'] += float(teacher); loss['zeroed'] += float(zeroed)
            r = (s[:, 1:] - pred[:, :15].float()).double()                                    # [b, 15, 81, 192] zeroed residual
            for j, cells_j in enumerate(regions):
                rj = r[:, :, cells_j].flatten(2).flatten(0, 1)                                 # [b * 15, tiles * 192]
                cj = tokens[..., j].flatten()
                sums[j].index_add_(0, cj, rj)
                counts[j].index_add_(0, cj, torch.ones_like(cj, dtype=torch.float64))
                ss[j] += float(rj.pow(2).sum())
            nb += 1
        out = {'world': st['name'], 'recon': recon, 'windows': len(keep), 'batches': nb,
               'teacher_with_delta': round(loss['delta'] / nb, 6), 'teacher_zeroed': round(loss['zeroed'] / nb, 6)}
        out['gain'] = round((loss['zeroed'] - loss['delta']) / nb, 7)
        out['gain_relative'] = round((loss['zeroed'] - loss['delta']) / loss['zeroed'], 5)
        for k, (g_sum, n_sum) in acc.items():
            out[k] = {'mean_norm': float(f'{n_sum / nb:.4g}'), 'consistency': round(float(g_sum.norm()) / n_sum, 4)}
        out['enc_teacher_over_commit_norm'] = float(f"{acc['enc_teacher'][1] / acc['enc_commit'][1]:.4g}")
        out['noise_consistency'] = round(nb ** -0.5, 4)
        out['cond_weight_rms'] = float(f'{float(post.cond.weight.pow(2).mean().sqrt()):.4g}')
        expl = []
        for j in range(len(regions)):
            n = counts[j].sum()
            total = ss[j] - float(sums[j].sum(0).pow(2).sum() / n)
            used = counts[j] > 0
            between = float((sums[j][used].pow(2).sum(1) / counts[j][used]).sum()) - float(sums[j].sum(0).pow(2).sum() / n)
            k = int(used.sum())
            expl.append({'codes_used': k, 'explained': round(between / total, 4), 'chance': round((k - 1) / (float(n) - 1), 4)})
        out['explained_by_region'] = expl
        out['peak_gb'] = round(torch.cuda.max_memory_allocated() / 2 ** 30, 2)
        res[name] = out
        print(json.dumps({name: out}), flush=True)
        del world, post; torch.cuda.empty_cache()
    (OUT / 'e22_channel.json').write_text(json.dumps(res, indent=1))


if __name__ == '__main__':
    sys.path.insert(0, str(ROOT / 'artifacts/experiments/20260926_diagnosis'))
    sys.path.insert(0, str(ROOT / 'artifacts/experiments/20260921_readout_ladder'))
    main()
