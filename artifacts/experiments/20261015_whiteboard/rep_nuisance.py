"""Row 1 (nuisance): can rendering nuisance be removed from Raw tokens by a fixed linear projection without losing content?
(2026-10-15, CPU, frozen encoder, no training)

Same data as rep_layers (597 futures x 16 steps from 120 roots; roots split 80 fit / 40 held out). Raw tokens (final layer,
LN). A nuisance subspace U_m (m dims) is estimated on the fit roots; tokens are projected to P = I - U_m U_m^T; on held-out
roots, as a function of m in (0, 2, 4, 8, 16, 32, 64):
  drift      persistent cells (own aligned tile + mob channels unchanged t -> t+1): RMS change relative to the RMS distance
             between different cells of the same frame, day still / day camera moved / night still
  content    ridge probes from the projected own token: tile class accuracy, zombie / cow / skeleton AUC
  transfer   single-cow moves, camera still: relative error of z'_dst ~ z_dst + (z_src - z'_src), same terrain
Two estimates of U_m:
  labelled     top eigenvectors of the within-class covariance (cells grouped by tile class x mob channels; the
               variation that never changes content). An upper bound that uses labels.
  label_free   top principal directions of the frame-to-frame change of ALL cells (camera aligned by the parameter-free
               token scroll estimator, no labels). Rare real content changes are a small part of all changes.
Usage: rep_nuisance.py -> artifacts/eda/whiteboard/rep_nuisance.json
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

os.environ.setdefault('TRITON_F32_DEFAULT', 'ieee')
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import rep_layers as RL  # noqa: E402

ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / 'artifacts/experiments/20261012_oracle'))


def main():
    torch.set_num_threads(8)
    import rollouts as Rr
    import audit_s5_direct as A5
    scroll = A5.load_scroll()
    files = sorted(Rr.STORE.glob('seed-*.pt'))
    roots = [r for f in files for r in torch.load(f, weights_only=False)]
    pick = torch.randperm(len(roots), generator=torch.Generator().manual_seed(20261015))[:120].tolist()
    fit_roots = set(pick[:80])
    enc = RL.encoder()
    seqs = []
    for ri in pick:
        r = roots[ri]
        for k in range(5):
            if bool(r['future_dead'][k].any()):
                continue
            seqs.append((torch.cat([r['context'][-1:], r['future_frames'][k]]), torch.cat([r['root_visible'][None], r['future_visible'][k]]), ri))
    frames = torch.cat([s[0] for s in seqs]); vis = torch.cat([s[1] for s in seqs])
    n = len(seqs)
    RL.LAYERS = (12,)
    z = RL.layer_tokens(enc, frames)[12].view(n, 17, 81, 192).float()
    tiles, mobs, light = RL.facts(vis)
    tiles, mobs, light = tiles.view(n, 17, 63), mobs.view(n, 17, 63, 7), light.view(n, 17)
    fit = torch.tensor([s[2] in fit_roots for s in seqs])
    # persistent-cell pairs and cow moves (labels only for the evaluation masks)
    prs, moves = [], []
    for s in range(n):
        for t in range(16):
            dr, dc = RL.camera(tiles[s, t], tiles[s, t + 1])
            for r in range(7):
                for c in range(9):
                    if not (0 <= r + dr < 7 and 0 <= c + dc < 9):
                        continue
                    j, jp = r * 9 + c, (r + dr) * 9 + c + dc
                    if 31 in (j, jp):
                        continue
                    if bool(tiles[s, t + 1, j] == tiles[s, t, jp]) and bool((mobs[s, t + 1, j] == mobs[s, t, jp]).all()):
                        key = 'night_still' if (light[s, t] < .5 and (dr, dc) == (0, 0)) else ('day_still' if (dr, dc) == (0, 0) else ('day_moved' if light[s, t] >= .5 else None))
                        if key:
                            prs.append((s, t, j, jp, key))
            if (dr, dc) == (0, 0):
                ct = torch.where(mobs[s, t, :63, 1])[0].tolist(); cn = torch.where(mobs[s, t + 1, :63, 1])[0].tolist()
                if len(ct) == 1 and len(cn) == 1 and ct != cn and abs(ct[0] // 9 - cn[0] // 9) + abs(ct[0] % 9 - cn[0] % 9) == 1 and 31 not in (ct[0], cn[0]):
                    if bool(tiles[s, t, ct[0]] == tiles[s, t, cn[0]]):
                        moves.append((s, t, ct[0], cn[0]))
    # nuisance subspaces (fit roots only)
    cls = tiles[..., :63] * 128 + (mobs[..., :63, :3].long() * torch.tensor([1, 2, 4])).sum(-1)       # [n,17,63]
    zz = z[:, :, :63]
    X = zz[fit].reshape(-1, 192); C = cls[fit].reshape(-1)
    mean_by = torch.zeros(int(C.max()) + 1, 192).index_add_(0, C, X) / torch.bincount(C, minlength=int(C.max()) + 1).clamp_min(1)[:, None]
    W = X - mean_by[C]
    sub = W[torch.randperm(len(W), generator=torch.Generator().manual_seed(3))[:200000]]
    _, _, Vl = torch.linalg.svd(sub - sub.mean(0), full_matrices=False)               # labelled within-class directions
    est = scroll.estimate(z[fit, :-1], z[fit, 1:])                                       # label-free camera
    dz = []
    for s_i, s in enumerate(torch.where(fit)[0].tolist()):
        for t in range(16):
            dr, dc = scroll.SHIFTS[int(est[s_i, t])]
            for r in range(7):
                for c in range(9):
                    if 0 <= r + dr < 7 and 0 <= c + dc < 9 and 31 not in (r * 9 + c, (r + dr) * 9 + c + dc):
                        dz.append(z[s, t + 1, r * 9 + c] - z[s, t, (r + dr) * 9 + c + dc])
    dz = torch.stack(dz)
    dz = dz[torch.randperm(len(dz), generator=torch.Generator().manual_seed(4))[:200000]]
    _, _, Vf = torch.linalg.svd(dz, full_matrices=False)                                # label-free change directions (uncentred)
    held = ~fit
    hidx = torch.where(held)[0]
    hset = set(hidx.tolist())
    res = {'sequences': n, 'fit_sequences': int(fit.sum()), 'held_sequences': int(held.sum()), 'estimates': {}}
    for name, V in (('labelled', Vl), ('label_free', Vf)):
        rows = {}
        for m in (0, 2, 4, 8, 16, 32, 64):
            U = V[:m].T                                                                  # [192,m]
            proj = lambda x: x - (x @ U) @ U.T if m else x
            zp = proj(z)
            row = {}
            scale_pairs = [p for p in prs if p[0] in hset]
            for key in ('day_still', 'day_moved', 'night_still'):
                pp = [p for p in scale_pairs if p[4] == key]
                if not pp:
                    continue
                s_, t_, j_, jp_ = (torch.tensor([p[i] for p in pp]) for i in range(4))
                d = (zp[s_, t_ + 1, j_] - zp[s_, t_, jp_]).square().mean(-1).sqrt()
                perm = torch.randperm(len(pp), generator=torch.Generator().manual_seed(5))
                between = (zp[s_, t_, jp_] - zp[s_[perm], t_[perm], jp_[perm]]).square().mean(-1).sqrt()
                row[f'drift_{key}'] = round(float(d.mean() / between.mean()), 4)
            cells = torch.tensor([j for j in range(63) if j != 31])
            Xp = zp[:, :, cells].reshape(-1, 192); yt = tiles[:, :, cells].reshape(-1); ym = mobs[:, :, cells].reshape(-1, 7)
            seq_of = torch.arange(n)[:, None, None].expand(n, 17, len(cells)).reshape(-1)
            tri = torch.where(fit[seq_of])[0]; tei = torch.where(held[seq_of])[0]
            g = torch.Generator().manual_seed(6)
            tri = tri[torch.randperm(len(tri), generator=g)[:60000]]; tei = tei[torch.randperm(len(tei), generator=g)[:40000]]
            pt = RL.ridge(Xp, F.one_hot(yt, 17).float(), tri, tei)
            row['tile_acc'] = round(float((pt.argmax(-1) == yt[tei]).float().mean()), 4)
            for ch, nm in ((0, 'zombie'), (1, 'cow'), (2, 'skeleton')):
                row[f'{nm}_auc'] = round(RL.auc(RL.ridge(Xp, ym[:, ch].float()[:, None], tri, tei)[:, 0], ym[tei, ch]), 4)
            errs = []
            for s, t, a_, b_ in moves:
                if s in hset:
                    pred = zp[s, t, b_] + (zp[s, t, a_] - zp[s, t + 1, a_])
                    errs.append(float((zp[s, t + 1, b_] - pred).norm() / (zp[s, t + 1, b_] - zp[s, t, b_]).norm().clamp_min(1e-6)))
            row['cow_transfer_same_terrain_median'] = round(float(np.median(errs)), 4) if errs else None
            rows[m] = row
            print(json.dumps({'estimate': name, 'm': m, **row}), flush=True)
        res['estimates'][name] = rows
    (RL.OUT / 'rep_nuisance.json').write_text(json.dumps(res, indent=1))


if __name__ == '__main__':
    main()
