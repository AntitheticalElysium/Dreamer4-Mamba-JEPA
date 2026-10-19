"""R53 analysis of dv16_trace_*.pt: where is action-dependent risk lost - in the imagined SCENE or in the imagined CONSEQUENCE?

Per frame: dmin = Manhattan distance from the player cell (row 3, col 4) to the nearest decoded zombie (8 if none), adj = a
zombie at distance 1. Depths compared: 1, 2, 4, 8, 16 (the stored real ones). Per world:
  scene ranking      within-root pairwise AUC over the 5 escape choices of the imagined mean adjacency (any of the depths <= k)
                     against the REAL P16 order; the same for the REAL key-0 scene (reference: how much of the risk order a
                     single real future's scene carries)
  scene fidelity     within-root Spearman correlation over the 5 choices between imagined mean dmin and real key-0 dmin, per depth
  consequence        imagined P(health drop >= 0.5 within the next step | adjacent now) at depths 1-15; real at depth 1 -> 2
  damage ranking     within-root AUC of imagined health loss by depth k against the real P16 order (R51's quantity, by depth)
  first / later      damage ranking from first-step damage only vs from further damage after depth 1 only; the share of the
                     16-step health loss that is first-step; across-action variance of the M-mean health loss vs its sampling variance
  pursuit            P(dmin decreases at the next step | 2 <= dmin <= 4 now), imagined over depths 1-15; real over depth 1 -> 2
Readings, declared before the result:
  - if PI's scene ranking matches D's but its damage ranking does not, the loss is in the consequence mapping;
  - if PI's scene ranking is already below D's, the loss is in the scene dynamics (and the pursuit / fidelity numbers say which).
Usage: dv16_trace_analysis.py -> artifacts/eda/whiteboard/dv16_trace_analysis.json
"""
import json
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / '20261012_oracle'))
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / '20260927_levers'))
import dv16 as DV

DEPTHS = (1, 2, 4, 8, 16)
IDX = [d - 1 for d in DEPTHS]
ROW, COL = torch.arange(63) // 9, torch.arange(63) % 9
DIST = (ROW - 3).abs() + (COL - 4).abs()                                                    # [63], player at cell 31


def dmin(z):
    """z [..., 63] bool -> [...] distance to the nearest zombie (8 if none)"""
    d = torch.where(z, DIST.to(z.device).float(), torch.full(z.shape, 8.))
    return d.min(-1).values.clamp(max=8)


def auc_rank(score, P):
    """score, P [R,5] -> within-root pairwise AUC of score against the P order (ties 0.5), roots with spread only"""
    num = den = 0.
    for i in range(len(P)):
        for a in range(5):
            for b in range(5):
                if P[i, a] > P[i, b]:
                    den += 1; num += float(score[i, a] > score[i, b]) + .5 * float(score[i, a] == score[i, b])
    return round(num / den, 4)


def spearman_rows(x, y):
    rx, ry = x.argsort(-1).argsort(-1).float(), y.argsort(-1).argsort(-1).float()
    rx, ry = rx - rx.mean(-1, keepdim=True), ry - ry.mean(-1, keepdim=True)
    c = (rx * ry).sum(-1) / (rx.norm(dim=-1) * ry.norm(dim=-1)).clamp_min(1e-9)
    ok = (x.std(-1) > 0) & (y.std(-1) > 0)
    return round(float(c[ok].mean()), 4), int(ok.sum())


def main():
    roots, _ = DV.load_roots()
    P = torch.stack([r['P'] for r in roots])[..., 2][:, :5]
    real = torch.load(DV.OUT / 'dv16_trace_real.pt', weights_only=False)
    rz, rh, h0 = real['zombies'], real['health'], real['h0']                                   # [R,5,5,63], [R,5,5], [R]
    radj, rd = (dmin(rz) == 1).float(), dmin(rz)
    res = {'real': {}}
    for k, d in enumerate(DEPTHS):
        res['real'][f'scene_rank_adj_by_d{d}'] = auc_rank(radj[:, :, :k + 1].amax(-1), P)
        res['real'][f'damage_rank_by_d{d}'] = auc_rank((h0[:, None] - rh[:, :, k]).clamp_min(0), P)
    res['real']['damage_rank_later_only'] = auc_rank((rh[:, :, :1] - rh[:, :, 1:]).clamp_min(0).amax(-1), P)
    res['real']['share_of_HL_from_first_step'] = round(float((h0[:, None] - rh[:, :, 0]).clamp_min(0).mean() / (h0[:, None, None] - rh).clamp_min(0).amax(-1).mean()), 4)
    a1 = radj[:, :, 0] > 0
    res['real']['P_drop_next_given_adjacent_d1'] = round(float(((rh[:, :, 1] <= rh[:, :, 0] - .5) & a1).sum() / a1.sum()), 4)
    far = (rd[:, :, 0] >= 2) & (rd[:, :, 0] <= 4)
    res['real']['pursuit_d1_to_d2'] = round(float(((rd[:, :, 1] < rd[:, :, 0]) & far).sum() / far.sum()), 4)
    for w in ('D_parent_ref', 'ic_s1', 'ic_s2', 'pi_s1', 'pi_s2'):
        f = DV.OUT / f'dv16_trace_{w}.pt'
        if not f.exists():
            continue
        x = torch.load(f, weights_only=False)
        z, h = x['zombies'], x['health'].float()                                              # [R,5,M,16,63], [R,5,M,16]
        dm = dmin(z); adj = (dm == 1).float()
        out = {}
        for k, d in enumerate(DEPTHS):
            sel = IDX[:k + 1]
            out[f'scene_rank_adj_by_d{d}'] = auc_rank(adj[..., sel].amax(-1).mean(2), P)
            out[f'damage_rank_by_d{d}'] = auc_rank((h0[:, None, None, None] - h[..., :d]).clamp_min(0).amax(-1).mean(2), P)
            out[f'scene_fidelity_dmin_d{d}'] = spearman_rows(dm[..., IDX[k]].mean(2), rd[:, :, k])
        d1 = (h0[:, None, None] - h[..., 0]).clamp_min(0)                                    # [R,5,M] first-step damage
        later = (h[..., :1] - h[..., 1:]).clamp_min(0).amax(-1)                              # [R,5,M] further damage after depth 1
        out['damage_rank_first_step'] = auc_rank(d1.mean(2), P)
        out['damage_rank_later_only'] = auc_rank(later.mean(2), P)
        tot = (h0[:, None, None, None] - h).clamp_min(0).amax(-1)                            # [R,5,M] the R51 HL quantity
        M = tot.shape[2]
        between = tot.mean(2).var(1).mean()                                                  # across actions, of the M-mean
        within = tot.var(2).mean() / M if M > 1 else torch.tensor(0.)                         # sampling noise of the M-mean
        out['HL_action_variance'] = round(float(between), 4); out['HL_sampling_variance_of_mean'] = round(float(within), 4)
        out['share_of_HL_from_first_step'] = round(float(d1.mean() / tot.mean().clamp_min(1e-9)), 4)
        nxt_drop = h[..., 1:] <= h[..., :-1] - .5
        a = adj[..., :-1] > 0
        out['P_drop_next_given_adjacent'] = round(float((nxt_drop & a).sum() / a.sum()), 4)
        out['P_drop_next_given_not_adjacent'] = round(float((nxt_drop & ~a).sum() / (~a).sum()), 4)
        fr = (dm[..., :-1] >= 2) & (dm[..., :-1] <= 4)
        out['pursuit'] = round(float(((dm[..., 1:] < dm[..., :-1]) & fr).sum() / fr.sum()), 4)
        out['adjacent_rate_by_depth'] = [round(float(adj[..., i].mean()), 3) for i in IDX]
        res[w] = out
    res['real']['adjacent_rate_by_depth'] = [round(float(radj[:, :, k].mean()), 3) for k in range(5)]
    print(json.dumps(res, indent=1))
    (DV.OUT / 'dv16_trace_analysis.json').write_text(json.dumps(res, indent=1))


if __name__ == '__main__':
    main()
