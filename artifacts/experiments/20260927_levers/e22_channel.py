"""E22 channel diagnostics (2026-10-09, written 09:59; NOTEBOOK "Substitution test of the warm start", predeclared 09:58). Is the Delta
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
Update-0 states (written 10:03): e22_init/dworld_a_s7_{warm,scratch}_u0.pt = train_a's exact initialization (seed 7: TWorld, then the
Posterior) with D's weights loaded (warm) or not (scratch): the channel's gradient at the start of each run (zero-init cond: the
encoder's teacher gradient is exactly 0 there, its consistency None).
Size (10:31; the 09:58 rule's 'cond RMS' was a proxy for it): delta_over_tiles = mean over batches of ||Delta|| / ||tile embedding||
(Frobenius, the 15 conditioned frames; Delta is added to world.embed(s), spatial.py's Linear 192 -> 256).
Carries (11:08; E22F P1 passed, P2 failed -> 'measure what the channel carries'): per-token squared error of the teacher-forced
prediction with the posterior's Delta vs Delta zeroed, over every target of the window, by class: HUD (63-80), player (31), map
cells that change (|s_t+1 - s_t| > tworld.EVENT_TAU), map cells that do not, and map cells holding a true zombie in the target frame
(simulator labels exist for target frames 4-15 = futures 0-11); reduction = 1 - with/zeroed, share = the class's part of the gain.
Changed cells split (11:49; E22FJ carries 36% of changed-cell error, 8% of zombie cells): scroll.estimate on the true pair; 'entering' =
the new edge row / column of a scroll step (check_e16's mask), 'changed_scroll_other' = other changed map cells on scroll steps,
'changed_no_scroll' = changed map cells on steps without scroll; 'zombie_cells_no_scroll' likewise.
Usage: e22_channel.py <name=dworld_a.pt> ...  -> artifacts/eda/health_chain_v1/e22_channel.json (u0 runs: --out <file>)
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
    from scroll import SHIFTS, estimate
    from d4mj.config import config_from_dict
    from d4mj.train import autocast_context
    config = config_from_dict(torch.load(TW.S.CHECKPOINT, map_location='cpu', weights_only=False)['config'])
    dev = torch.device('cuda')
    meta, _, _ = T.split()
    cache = T.build_cache('raw', torch.device('cpu'))
    keep = torch.where(~meta['future_dead'][:, 0].cumsum(1).bool()[:, 11])[0]
    s_all = torch.cat([cache['ctx'], cache['fut'][:, :12]], 1)[keep]                       # [n, 16, 81, 192]
    a_all = torch.cat([cache['ctx_a'], cache['fut_a'][:, :12]], 1)[keep]                   # [n, 15]
    allv = torch.cat([meta['root_visible'][:, None], meta['future_visible'][:, 0]], 1).float()
    zt = (allv[..., 1071:1512].reshape(len(allv), 17, 7, 9, 7)[..., 0] > 0)[keep].flatten(2)          # [n, 17, 63]: frames 3..19
    cells = torch.arange(81).view(9, 9)
    regions = [cells[5 * i:5 * i + 5, 5 * j:5 * j + 5].flatten().to(dev) for i in range(2) for j in range(2)]
    res = {}
    for name, path in (x.split('=', 1) for x in sys.argv[1:] if '=' in x):
        world, post, st = DW.load_a(Path(path), dev)
        recon = st['args'].get('recon', 'l1')
        enc = [p for m in (post.net, post.action, post.quantizer.pre) for p in m.parameters()]
        loss = {'delta': 0.0, 'zeroed': 0.0}
        acc = {k: [0, 0.0] for k in ('cond', 'enc_teacher', 'enc_commit')}                # [sum of gradients, sum of norms]
        sums = [torch.zeros(DW.CODES, len(r) * TW.S.WIDTH, dtype=torch.float64, device=dev) for r in regions]
        counts = [torch.zeros(DW.CODES, dtype=torch.float64, device=dev) for _ in regions]
        ss = [0.0 for _ in regions]
        torch.cuda.reset_peak_memory_stats()
        nb, ratio = 0, 0.0
        carry = {}                                                                   # class -> [with Delta, zeroed, count]
        for i in range(0, len(keep), BATCH):
            s, a = s_all[i:i + BATCH].to(dev).float(), a_all[i:i + BATCH].to(dev)
            with autocast_context(config):
                cond, tokens, commit = post(s, a)
                world.delta = cond
                teacher = TW.rollout_losses(world, s, a, 'teacher', recon=recon)
                ratio += float(cond[:, :-1].detach().float().norm() / world.embed(s[:, :-1]).detach().float().norm())
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
                world.delta = cond.detach()
                pred_d = world(s, F.pad(a, (0, 1)))[0]
                world.delta = None
            loss['delta'] += float(teacher); loss['zeroed'] += float(zeroed)
            r = (s[:, 1:] - pred[:, :15].float()).double()                                    # [b, 15, 81, 192] zeroed residual
            e0, e1 = r.pow(2).sum(-1), (s[:, 1:] - pred_d[:, :15].float()).double().pow(2).sum(-1)       # [b, 15, 81]
            change = (s[:, 1:] - s[:, :-1]).norm(dim=-1) > TW.EVENT_TAU
            mapm = torch.zeros(81, dtype=torch.bool, device=dev); mapm[:63] = True; mapm[31] = False
            zomb = torch.zeros_like(change)
            zomb[:, 3:, :63] = zt[i:i + BATCH, 1:13].to(dev)                                 # targets 4..15 = frames 4..15
            classes = {'hud': torch.zeros_like(change), 'player': torch.zeros_like(change),
                       'map_changed': change & mapm, 'map_unchanged': ~change & mapm, 'zombie_cells': zomb}
            classes['hud'][..., 63:] = True; classes['player'][..., 31] = True
            sh = estimate(s[:, :-1], s[:, 1:])                                                  # [b, 15] index into SHIFTS
            ent = torch.zeros_like(change)
            for k_s, (dr, dc) in enumerate(SHIFTS):
                m_s = (sh == k_s)[..., None]
                if dr: ent[..., (6 if dr == 1 else 0) * 9:(6 if dr == 1 else 0) * 9 + 9] |= m_s
                if dc: ent[..., [r * 9 + (8 if dc == 1 else 0) for r in range(7)]] |= m_s
            scr = (sh != 0)[..., None]
            classes.update({'entering': ent & mapm, 'changed_scroll_other': change & mapm & scr & ~ent,
                            'changed_no_scroll': change & mapm & ~scr, 'zombie_cells_no_scroll': zomb & ~scr})
            for cname, m in classes.items():
                c = carry.setdefault(cname, [0.0, 0.0, 0])
                c[0] += float(e1[m].sum()); c[1] += float(e0[m].sum()); c[2] += int(m.sum())
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
            out[k] = {'mean_norm': float(f'{n_sum / nb:.4g}'), 'consistency': round(float(g_sum.norm()) / n_sum, 4) if n_sum > 0 else None}
        out['enc_teacher_over_commit_norm'] = float(f"{acc['enc_teacher'][1] / acc['enc_commit'][1]:.4g}")   # 0 at zero-init cond
        out['noise_consistency'] = round(nb ** -0.5, 4)
        out['delta_over_tiles'] = float(f'{ratio / nb:.4g}')
        total_gain = sum(c[1] - c[0] for k, c in carry.items() if k in ('hud', 'player', 'map_changed', 'map_unchanged'))
        out['carries'] = {k: {'tokens': c[2], 'err_with_delta': float(f'{c[0] / c[2]:.5g}'), 'err_zeroed': float(f'{c[1] / c[2]:.5g}'),
                              'reduction': round(1 - c[0] / c[1], 4), 'share_of_gain': round((c[1] - c[0]) / total_gain, 4)}
                          for k, c in carry.items()}
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
    (OUT / (sys.argv[sys.argv.index('--out') + 1] if '--out' in sys.argv else 'e22_channel.json')).write_text(json.dumps(res, indent=1))


if __name__ == '__main__':
    sys.path.insert(0, str(ROOT / 'artifacts/experiments/20260926_diagnosis'))
    sys.path.insert(0, str(ROOT / 'artifacts/experiments/20260921_readout_ladder'))
    main()
