"""Representation probe (2026-10-15, CPU, frozen Raw encoder, no training): is there a layer of our own JEPA encoder whose
patch tokens are persistent where content persists, invariant to night nuisance, local, and transferable when a mob moves?

The Raw encoder is a 12-layer ViT, width 192, patch 7 on 63x63 frames: one patch = one Craftax tile. Layer 0 (patch
embedding + position) is a linear function of the tile's own 7x7 pixels; later layers mix context. The world models use the
final layer (LN, no affine). Every layer below gets the same LN (no affine), so distances are comparable.

Data: diagnosis_futures_v1 (pixels + exact visible state): 120 roots (seeded), 5 independent futures x 16 steps, plus the
root frame: consecutive pairs (t, t+1). Camera shift between t and t+1 from the exact tile maps (the shift with fewest
aligned tile mismatches).
Per layer:
  drift_persistent   RMS change of a token whose own aligned tile class and mob channels are unchanged, relative to the RMS
                     distance between two different persistent cells of the same frame (scale): split day / night,
                     camera still / moved, and whether any 4-neighbour's content changed (context change)
  probe_tile_acc     ridge probe (own token) -> own tile class, held-out roots
  probe_mob_auc      ridge -> own cell has zombie / cow / skeleton
  leak_auc           ridge -> a 4-neighbour has a cow, on cells with no mob (context leakage)
  move_transfer      single-cow moves (camera still): relative error of z'_dst ~ z_dst + (z_src - z'_src), same vs different
                     terrain under source and destination
  cow_delta_cos      mean cosine between cow deltas (token with cow minus mean token of the same tile class without mob)
                     on different terrains: is mob content separable from background?
Usage: rep_layers.py -> artifacts/eda/whiteboard/rep_layers.json
"""
import json
import os
import sys
from pathlib import Path

os.environ.setdefault('TRITON_F32_DEFAULT', 'ieee')                    # the encoder bundle's recorded execution identity

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[3]
for d in ('20260927_levers', '20260926_diagnosis', '20260921_readout_ladder'):
    sys.path.insert(0, str(ROOT / 'artifacts/experiments' / d))
sys.path.insert(0, str(ROOT))
OUT = ROOT / 'artifacts/eda/whiteboard'
LAYERS = (0, 1, 2, 3, 4, 6, 8, 10, 12)
SHIFTS = ((0, 0), (-1, 0), (1, 0), (0, -1), (0, 1))


def encoder():
    from d4mj.checkpoint import read_lewm_bundle
    from d4mj.config import config_from_dict
    from d4mj.world_api import ModelBundle
    import teval as T
    payload = read_lewm_bundle(T.ENCODERS['raw'])
    bundle = ModelBundle.create(config_from_dict(payload['config']))
    bundle.encoder.load_state_dict(payload['modules']['encoder'])
    return bundle.encoder.cpu().eval()


@torch.no_grad()
def layer_tokens(enc, frames):
    """uint8 [n,63,63,3] -> {layer: LN tokens [n,81,192] fp16}; the final layer reproduces Raw (backbone LN then LN)"""
    out = {l: [] for l in LAYERS}
    for i in range(0, len(frames), 256):
        px = frames[i:i + 256].permute(0, 3, 1, 2).float() / 255.0
        px = (px - enc.pixel_mean) / enc.pixel_std
        hs = enc.backbone(px, interpolate_pos_encoding=True, output_hidden_states=True)
        for l in LAYERS:
            h = hs.last_hidden_state if l == 12 else hs.hidden_states[l]
            out[l].append(F.layer_norm(h[:, 1:].float(), (192,)).half())
    return {l: torch.cat(v) for l, v in out.items()}


def facts(vis):
    v = vis.float()
    tiles = v[..., :1071].reshape(*v.shape[:-1], 63, 17).argmax(-1)
    mobs = v[..., 1071:1512].reshape(*v.shape[:-1], 63, 7) > 0
    return tiles, mobs, v[..., 1521]


def camera(tile_a, tile_b):
    """exact-map shift: successor[r,c] = prev[r+dr, c+dc]"""
    a, b = tile_a.view(7, 9), tile_b.view(7, 9)
    best = None
    for s, (dr, dc) in enumerate(SHIFTS):
        rs, cs = slice(max(0, -dr), 7 - max(0, dr)), slice(max(0, -dc), 9 - max(0, dc))
        rt, ct = slice(max(0, dr), 7 + min(0, dr)), slice(max(0, dc), 9 + min(0, dc))
        err = float((b[rs, cs] != a[rt, ct]).float().mean())
        if best is None or err < best[0]:
            best = (err, s)
    return SHIFTS[best[1]]


def ridge(x, y, tr, te, lam=1.0):
    mu, sd = x[tr].mean(0), x[tr].std(0).clamp_min(1e-6)
    xs = lambda q: torch.cat([((q - mu) / sd).double(), torch.ones(len(q), 1, dtype=torch.float64)], 1)
    X = xs(x[tr])
    w = torch.linalg.solve(X.T @ X + lam * len(X) * torch.eye(X.shape[1], dtype=X.dtype), X.T @ y[tr].double())
    return (xs(x[te]) @ w).float()


def auc(score, label):
    """Mann-Whitney AUC with average ranks for ties"""
    label = label.bool()
    npos, nneg = int(label.sum()), int((~label).sum())
    if npos == 0 or nneg == 0:
        return None
    order = score.double().argsort()
    ranks = torch.empty(len(score), dtype=torch.float64); ranks[order] = torch.arange(1, len(score) + 1, dtype=torch.float64)
    s_sorted = score.double()[order]
    uniq, inv, counts = torch.unique_consecutive(s_sorted, return_inverse=True, return_counts=True)
    if len(uniq) < len(score):                                            # average ranks within ties
        csum = torch.cumsum(counts, 0).double(); avg = csum - (counts.double() - 1) / 2
        ranks[order] = avg[inv]
    return float((ranks[label].sum() - npos * (npos + 1) / 2) / (npos * nneg))


def main():
    torch.set_num_threads(8)
    import rollouts as Rr
    files = sorted(Rr.STORE.glob('seed-*.pt'))
    roots = [r for f in files for r in torch.load(f, weights_only=False)]
    g = torch.Generator().manual_seed(20261015)
    pick = torch.randperm(len(roots), generator=g)[:120].tolist()
    enc = encoder()
    seqs = []                                                             # (frames [17,...], vis [17,1534], root id)
    for ri in pick:
        r = roots[ri]
        for k in range(5):
            if bool(r['future_dead'][k].any()):
                continue
            frames = torch.cat([r['context'][-1:], r['future_frames'][k]])
            vis = torch.cat([r['root_visible'][None], r['future_visible'][k]])
            seqs.append((frames, vis, ri))
    frames = torch.cat([s[0] for s in seqs]); vis = torch.cat([s[1] for s in seqs])
    rid = torch.tensor([s[2] for s in seqs for _ in range(17)])
    print(json.dumps({'sequences': len(seqs), 'frames': len(frames)}), flush=True)
    toks = layer_tokens(enc, frames)
    with torch.no_grad():                                                 # the final layer must reproduce the Raw tokens exactly
        ref = F.layer_norm(enc._hidden(frames[:16, None])[2].float(), (192,))
    res_check = float((toks[12][:16].float() - ref).abs().max())
    assert res_check < 1e-2, res_check
    print(json.dumps({'final_layer_vs_raw_maxabs_fp16': res_check}), flush=True)
    tiles, mobs, light = facts(vis)
    n = len(seqs)
    T17 = lambda x: x.view(n, 17, *x.shape[1:])
    tiles, mobs, light, rid17 = T17(tiles), T17(mobs), T17(light), T17(rid)
    toks = {l: T17(v) for l, v in toks.items()}
    # pairs and per-cell masks
    pairs = []                                                            # (seq, t, dr, dc)
    for s in range(n):
        for t in range(16):
            dr, dc = camera(tiles[s, t], tiles[s, t + 1])
            pairs.append((s, t, dr, dc))
    res = {'sequences': n, 'pairs': len(pairs), 'layers': {}}
    # persistent-cell index lists (shared across layers)
    idx_prev, idx_next, strat = [], [], []                               # flat cell indices into [n,17,81]
    move_rows = []                                                       # (s, t, src, dst, same_terrain)
    for s, t, dr, dc in pairs:
        for r in range(7):
            for c in range(9):
                if not (0 <= r + dr < 7 and 0 <= c + dc < 9):
                    continue
                j, jp = r * 9 + c, (r + dr) * 9 + c + dc
                if 31 in (j, jp):
                    continue
                same = bool(tiles[s, t + 1, j] == tiles[s, t, jp]) and bool((mobs[s, t + 1, j] == mobs[s, t, jp]).all())
                if not same:
                    continue
                nb_changed = False
                for a, e in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    rr, cc = r + a, c + e
                    if 0 <= rr < 7 and 0 <= cc < 9 and 0 <= rr + dr < 7 and 0 <= cc + dc < 9:
                        q, qp = rr * 9 + cc, (rr + dr) * 9 + cc + dc
                        if bool(tiles[s, t + 1, q] != tiles[s, t, qp]) or bool((mobs[s, t + 1, q] != mobs[s, t, qp]).any()):
                            nb_changed = True
                idx_prev.append((s * 17 + t) * 81 + jp); idx_next.append((s * 17 + t + 1) * 81 + j)
                strat.append((bool(light[s, t] < .5), (dr, dc) != (0, 0), nb_changed))
        if (dr, dc) == (0, 0):
            cows_t = torch.where(mobs[s, t, :63, 1])[0].tolist()
            cows_n = torch.where(mobs[s, t + 1, :63, 1])[0].tolist()
            if len(cows_t) == 1 and len(cows_n) == 1 and cows_t != cows_n:
                a_, b_ = cows_t[0], cows_n[0]
                if abs(a_ // 9 - b_ // 9) + abs(a_ % 9 - b_ % 9) == 1 and 31 not in (a_, b_):
                    move_rows.append((s, t, a_, b_, bool(tiles[s, t, a_] == tiles[s, t, b_])))
    idx_prev, idx_next = torch.tensor(idx_prev), torch.tensor(idx_next)
    strat = torch.tensor(strat)                                          # [m,3] night, camera, neighbour changed
    print(json.dumps({'persistent_cells': len(idx_prev), 'night_share': float(strat[:, 0].float().mean()),
                      'cow_moves': len(move_rows)}), flush=True)
    # probe data: all map cells except the player
    cells = torch.tensor([j for j in range(63) if j != 31])
    train_roots = set(pick[:80])
    tr_seq = torch.tensor([s[2] in train_roots for s in seqs])
    for l in LAYERS:
        z = toks[l].float()                                              # [n,17,81,192]
        flat = z.view(-1, 192)
        d = (flat[idx_next] - flat[idx_prev]).square().mean(-1).sqrt()  # per-cell RMS change
        perm = torch.randperm(len(idx_prev), generator=torch.Generator().manual_seed(1))
        scale = float((flat[idx_prev] - flat[idx_prev[perm]]).square().mean(-1).sqrt().mean())
        row = {'scale_rms_between_cells': round(scale, 4)}
        for name, m in (('day_still_nbsame', ~strat[:, 0] & ~strat[:, 1] & ~strat[:, 2]), ('day_still_nbchanged', ~strat[:, 0] & ~strat[:, 1] & strat[:, 2]),
                        ('day_moved_nbsame', ~strat[:, 0] & strat[:, 1] & ~strat[:, 2]), ('night_still_nbsame', strat[:, 0] & ~strat[:, 1] & ~strat[:, 2]),
                        ('night_moved', strat[:, 0] & strat[:, 1])):
            row[f'drift_{name}'] = round(float(d[m].mean() / scale), 4) if m.any() else None
            row[f'n_{name}'] = int(m.sum())
        # probes (subsample cells for speed)
        X = z[:, :, cells].reshape(-1, 192)
        ytile = tiles[:, :, cells].reshape(-1); ymob = mobs[:, :, cells].reshape(-1, 7)
        nb_cow = torch.zeros(n, 17, 63, dtype=torch.bool)
        cow = mobs[..., :63, 1].view(n, 17, 7, 9)
        nbm = torch.zeros_like(cow)
        nbm[..., 1:, :] |= cow[..., :-1, :]; nbm[..., :-1, :] |= cow[..., 1:, :]; nbm[..., :, 1:] |= cow[..., :, :-1]; nbm[..., :, :-1] |= cow[..., :, 1:]
        nb_cow = nbm.view(n, 17, 63)[:, :, cells].reshape(-1)
        seq_of = torch.arange(n)[:, None, None].expand(n, 17, len(cells)).reshape(-1)
        tr = tr_seq[seq_of]; te = ~tr
        sub = torch.Generator().manual_seed(2)
        tri = torch.where(tr)[0]; tri = tri[torch.randperm(len(tri), generator=sub)[:60000]]
        tei = torch.where(te)[0]; tei = tei[torch.randperm(len(tei), generator=sub)[:40000]]
        pt = ridge(X, F.one_hot(ytile, 17).float(), tri, tei)
        row['probe_tile_acc'] = round(float((pt.argmax(-1) == ytile[tei]).float().mean()), 4)
        for ch, nm in ((0, 'zombie'), (1, 'cow'), (2, 'skeleton')):
            p = ridge(X, ymob[:, ch].float()[:, None], tri, tei)[:, 0]
            row[f'probe_{nm}_auc'] = round(auc(p, ymob[tei, ch]), 4)
        empty = ~ymob[:, :3].any(-1)
        tri_e, tei_e = tri[empty[tri]], tei[empty[tei]]
        p = ridge(X, nb_cow.float()[:, None], tri_e, tei_e)[:, 0]
        row['leak_neighbour_cow_auc'] = round(auc(p, nb_cow[tei_e]), 4)
        # move transfer
        errs = {True: [], False: []}
        for s, t, a_, b_, same in move_rows:
            zt, zn = z[s, t], z[s, t + 1]
            pred = zt[b_] + (zt[a_] - zn[a_])
            errs[same].append(float((zn[b_] - pred).norm() / (zn[b_] - zt[b_]).norm().clamp_min(1e-6)))
        row['move_transfer_rel_err_same_terrain_median'] = round(float(np.median(errs[True])), 4) if errs[True] else None
        row['move_transfer_rel_err_cross_terrain_median'] = round(float(np.median(errs[False])), 4) if errs[False] else None
        # cow delta across terrains (day frames)
        day = light[:, :] >= .5
        deltas = {}
        for tcls in torch.unique(tiles[..., :63][mobs[..., :63, 1] & day[..., None]]).tolist():
            withcow = mobs[..., :63, 1] & (tiles[..., :63] == tcls) & day[..., None]
            bg = ~mobs[..., :63, :3].any(-1) & (tiles[..., :63] == tcls) & day[..., None]
            if withcow.sum() >= 20 and bg.sum() >= 200:
                zz = z[..., :63, :]
                deltas[tcls] = zz[withcow].mean(0) - zz[bg].mean(0)
        keys = sorted(deltas)
        cos = [float(F.cosine_similarity(deltas[a][None], deltas[b][None])) for i, a in enumerate(keys) for b in keys[i + 1:]]
        row['cow_delta_cross_terrain_cos_mean'] = round(float(np.mean(cos)), 4) if cos else None
        row['cow_delta_terrains'] = keys
        res['layers'][l] = row
        print(json.dumps({'layer': l, **row}), flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'rep_layers.json').write_text(json.dumps(res, indent=1))


if __name__ == '__main__':
    main()
