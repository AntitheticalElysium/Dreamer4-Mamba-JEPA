"""Can moved content be found WITHOUT labels? (2026-10-09, written 12:50-12:53; E22Z: weighting zombie cells (label-derived) makes
the channel carry the random move; a generic version needs a label-free mask.) Measured on held diagnosis transitions (root + the 16
true futures, sample 0, alive), token space only, labels used ONLY to validate:
  scroll     scroll.estimate on the true pair -> shift (dr, dc); the compensated previous frame p(r, c) = s_t(r + dr, c + dc) (the
             convention of scroll.estimate); cells without a source (the entering edge) are excluded
  change     d(c) = |s_t+1(c) - p(c)|; a cell changes when d > tworld.EVENT_TAU (2.5)
  ratio      for a changed cell, min over its 4 neighbours c' of |s_t+1(c) - p(c')| / d(c) (content moved IN from a neighbour) and
             min over neighbours of |p(c) - s_t+1(c')| / d(c) (content moved OUT to a neighbour); the smaller of the two
  moved      changed cells with ratio < rho
Reported: the ratio's histogram on changed cells (is it bimodal, and where is the valley?), and for rho in a grid: share of map
cells marked, recall of zombie cells (zombie at t or t+1, simulator mob channel 0), share of marked cells holding any mob at t or t+1
(mob channels 0-6, per channel), and the share of D's teacher-forced squared error inside the mask (D's predictions with window
min(4 + k, 15), as health_imagination's teacher).
Runs on GPU for D's predictions. Usage: e22_motionmask.py <D world.pt>  -> artifacts/eda/health_chain_v1/e22_motionmask.json
"""
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, 'artifacts/experiments/20260927_levers')
import check_decision_step as CD  # noqa: E402
SD, T, H = CD.SD, CD.T, CD.H
OUT = Path('artifacts/eda/health_chain_v1')


def moved_ratio(a, b):
    """a = s_t, b = s_t+1 [n, 81, D] (any device) -> ratio [n, 63] (inf where unchanged or no source), change norm [n, 63],
    entering [n, 63]. Device-aware and vectorized 12:58 (dworld --motion-weight); bit-identical to the 12:53 version (checked)."""
    from scroll import SHIFTS, estimate
    import tworld as TW
    n, dev = len(a), a.device
    sh = estimate(a, b)
    ga, gb = a[:, :63].float().view(n, 7, 9, -1), b[:, :63].float().view(n, 7, 9, -1)
    shifts = torch.tensor(SHIFTS, device=dev)
    dr, dc = shifts[sh, 0], shifts[sh, 1]
    r7, c9 = torch.arange(7, device=dev), torch.arange(9, device=dev)
    rr, cc = r7[None, :, None] + dr[:, None, None], c9[None, None, :] + dc[:, None, None]
    valid = (rr >= 0) & (rr < 7) & (cc >= 0) & (cc < 9)                                          # [n, 7, 9]
    p = ga[torch.arange(n, device=dev)[:, None, None], rr.clamp(0, 6), cc.clamp(0, 8)]          # compensated s_t
    d = (gb - p).norm(dim=-1)
    big = torch.full_like(d, float('inf'))
    best_in, best_out = big.clone(), big.clone()
    for er, ec in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        r2, c2 = r7[:, None] + er, c9[None, :] + ec
        ok = ((r2 >= 0) & (r2 < 7) & (c2 >= 0) & (c2 < 9))[None] & valid & valid[:, r2.clamp(0, 6), c2.clamp(0, 8)]
        pn, bn = p[:, r2.clamp(0, 6), c2.clamp(0, 8)], gb[:, r2.clamp(0, 6), c2.clamp(0, 8)]
        best_in = torch.minimum(best_in, torch.where(ok, (gb - pn).norm(dim=-1), big))
        best_out = torch.minimum(best_out, torch.where(ok, (p - bn).norm(dim=-1), big))
    changed = valid & (d > TW.EVENT_TAU)
    ratio = torch.where(changed, torch.minimum(best_in, best_out) / d.clamp(min=1e-6), big)
    return ratio.flatten(1), d.flatten(1), (~valid).flatten(1)


@torch.no_grad()
def main():
    from d4mj.config import config_from_dict
    import spatial as Sp
    device = torch.device('cuda')
    config = config_from_dict(torch.load(Sp.CHECKPOINT, map_location='cpu', weights_only=False)['config'])
    meta, _, _ = T.split()
    fut5, _ = SD.token_cache(device)
    cache = T.build_cache('raw', device)
    R = len(meta['seed'])
    ctx, ca, fa = cache['ctx'], cache['ctx_a'], cache['fut_a']
    alive = ~meta['future_dead'][:, 0].cumsum(1).bool()
    allv = torch.cat([meta['root_visible'][:, None], meta['future_visible'][:, 0]], 1).float()
    mobs = allv[..., 1071:1512].reshape(R, 17, 63, 7) > 0                                           # [R, 17, 63, 7]
    frames = torch.cat([ctx[:, -1:].cpu(), fut5[:, 0].cpu()], 1)                                    # [R, 17, 81, 192]
    world, _ = T.load_world(Path(sys.argv[1]), device)
    err = torch.empty(R, H, 63)
    for i in range(0, R, 16):                                                                      # D's teacher-forced error
        b = min(16, R - i)
        tfr, hist = [ctx[i:i + b, j].float() for j in range(4)], [ca[i:i + b, j] for j in range(3)]
        for k in range(H):
            w = min(4 + k, 15)
            acts = torch.stack(hist[-(w - 1):] + [fa[i:i + b, k]], 1)
            pred = T.step(world, torch.stack(tfr[-w:], 1), acts, device, config)
            err[i:i + b, k] = (pred[:, :63].float().cpu() - fut5[i:i + b, 0, k, :63].float().cpu()).pow(2).sum(-1)
            tfr.append(fut5[i:i + b, 0, k].float()); hist.append(fa[i:i + b, k])
    ratio = torch.stack([torch.cat([moved_ratio(frames[i:i + 64, k], frames[i:i + 64, k + 1])[0] for i in range(0, R, 64)]) for k in range(H)], 1)
    live = alive[:, :, None].expand(-1, -1, 63)
    zomb = (mobs[:, :-1, :, 0] | mobs[:, 1:, :, 0]) & live
    anymob = (mobs[:, :-1].any(-1) | mobs[:, 1:].any(-1)) & live
    fin = torch.isfinite(ratio) & live
    res = {'changed_cells': int(fin.sum()), 'map_cells': int(live.sum()), 'zombie_cells': int(zomb.sum())}
    edges = torch.linspace(0, 1.5, 31)
    res['ratio_hist_changed'] = {f'{float(edges[j]):.2f}': int(((ratio >= edges[j]) & (ratio < edges[j + 1]) & fin).sum()) for j in range(30)}
    res['ratio_hist_changed_zombie'] = {f'{float(edges[j]):.2f}': int(((ratio >= edges[j]) & (ratio < edges[j + 1]) & fin & zomb).sum()) for j in range(30)}
    total_err = float(err[live].sum())
    res['by_rho'] = {}
    for rho in (0.3, 0.4, 0.5, 0.6, 0.7, 0.8):
        m = (ratio < rho) & live
        per_ch = {str(ch): round(float((m & ((mobs[:, :-1, :, ch] | mobs[:, 1:, :, ch]))).sum() / m.sum().clamp(min=1)), 4) for ch in range(7)}
        res['by_rho'][str(rho)] = {'marked_share_of_map_cells': round(float(m.sum() / live.sum()), 5),
                                   'zombie_recall': round(float((m & zomb).sum() / zomb.sum()), 4),
                                   'marked_with_any_mob': round(float((m & anymob).sum() / m.sum().clamp(min=1)), 4),
                                   'marked_with_mob_channel': per_ch,
                                   'D_error_share_inside': round(float(err[m].sum()) / total_err, 4),
                                   'zombie_D_error_share': round(float(err[zomb].sum()) / total_err, 4)}
        print(json.dumps({rho: res['by_rho'][str(rho)]}), flush=True)
    (OUT / 'e22_motionmask.json').write_text(json.dumps(res, indent=1))


if __name__ == '__main__':
    sys.path.insert(0, 'artifacts/experiments/20260926_diagnosis')
    sys.path.insert(0, 'artifacts/experiments/20260921_readout_ladder')
    main()
