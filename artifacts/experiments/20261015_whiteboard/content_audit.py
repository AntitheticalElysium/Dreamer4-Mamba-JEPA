"""Content invariance / information audit of candidate persistent representations (2026-10-15 late, user programme "persist
information, not representation"; CPU, frozen encoder, no training; exact simulator state for labels only).

The frozen Raw encoder factors as token = LN(Upper(PatchEmbed(pixels) + PosEmbed)) (recontext_stale.py). Candidates:
  raw            the Raw token (what PI persists)
  content        LN(PatchEmbed(pixels)): local, position-free (spatially equivariant by construction), linear in the pixels
  content_norm   LN(PatchEmbed(normalized pixels)): the map area of each frame standardized per colour channel and rescaled to
                 the dataset's global per-channel statistics (a general, label-free per-frame lighting normalization; the HUD
                 strip untouched)
  raw_norm       the full encoder on the normalized pixels (normalization + contextual recomputation)
Same data and measures as rep_layers.py (diagnosis_futures_v1, 120 seeded roots x 5 futures x 17 frames): drift of persistent
content relative to between-cell scale (day / night, camera still / moved, neighbours unchanged), ridge probes (tile class,
zombie / cow / skeleton AUC, neighbour-cow leak), single-cow move transfer, cow-delta cosine across terrains; plus HUD health R^2
(ridge from the 18 HUD representations, held-out roots).
Usage: content_audit.py -> artifacts/eda/whiteboard/content_audit.json
"""
import json
import os
import sys
from pathlib import Path

os.environ.setdefault('TRITON_F32_DEFAULT', 'ieee')
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import numpy as np
import torch
import torch.nn.functional as F

import rep_layers as RL

OUT = HERE.parents[1] / 'eda/whiteboard'
MAP_PX = 49                                                                       # 7 tile rows x 7 pixels


def normalize(frames, g_mean, g_std):
    """uint8 [n,63,63,3] -> float pixels in [0,1] with the map area standardized per frame and channel, rescaled globally"""
    x = frames.float() / 255.0
    m = x[:, :MAP_PX]
    mu, sd = m.mean((1, 2), keepdim=True), m.std((1, 2), keepdim=True).clamp_min(1e-3)
    x[:, :MAP_PX] = ((m - mu) / sd * g_std + g_mean).clamp(0, 1)
    return x


@torch.no_grad()
def represent(enc, px):
    """float pixels [n,63,63,3] in [0,1] -> (content LN [n,81,192], raw LN [n,81,192])"""
    bb = enc.backbone
    cs, rs = [], []
    for i in range(0, len(px), 256):
        p = px[i:i + 256].permute(0, 3, 1, 2)
        p = (p - enc.pixel_mean) / enc.pixel_std
        c = bb.embeddings.patch_embeddings(p)
        x = torch.cat([bb.embeddings.cls_token.expand(len(c), -1, -1), c], 1)
        x = x + bb.embeddings.interpolate_pos_encoding(x, 63, 63)
        for layer in bb.layers:
            x = layer(x, None)
        cs.append(F.layer_norm(c.float(), (192,)).half()); rs.append(F.layer_norm(bb.layernorm(x)[:, 1:].float(), (192,)).half())
    return torch.cat(cs), torch.cat(rs)


def main():
    torch.set_num_threads(16)
    import rollouts as Rr
    files = sorted(Rr.STORE.glob('seed-*.pt'))
    roots = [r for f in files for r in torch.load(f, weights_only=False)]
    pick = torch.randperm(len(roots), generator=torch.Generator().manual_seed(20261015))[:120].tolist()   # rep_layers' roots
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
    m = frames[:, :MAP_PX].float() / 255.0
    g_mean, g_std = m.mean((0, 1, 2)), m.std((1, 2)).mean(0)
    reps = {}
    reps['content'], reps['raw'] = represent(enc, frames.float() / 255.0)
    reps['content_norm'], reps['raw_norm'] = represent(enc, normalize(frames, g_mean, g_std))
    print(json.dumps({'sequences': n, 'frames': len(frames)}), flush=True)
    tiles, mobs, light = RL.facts(vis)
    T17 = lambda x: x.view(n, 17, *x.shape[1:])
    tiles, mobs, light = T17(tiles), T17(mobs), T17(light)
    health = T17(vis.float()[:, 1512])
    pairs = [(s, t, *RL.camera(tiles[s, t], tiles[s, t + 1])) for s in range(n) for t in range(16)]
    idx_prev, idx_next, strat = [], [], []
    for s, t, dr, dc in pairs:
        for r in range(7):
            for c in range(9):
                if not (0 <= r + dr < 7 and 0 <= c + dc < 9):
                    continue
                j, jp = r * 9 + c, (r + dr) * 9 + c + dc
                if 31 in (j, jp) or bool(tiles[s, t + 1, j] != tiles[s, t, jp]) or bool((mobs[s, t + 1, j] != mobs[s, t, jp]).any()):
                    continue
                nb = False
                for a, e in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    rr, cc = r + a, c + e
                    if 0 <= rr < 7 and 0 <= cc < 9 and 0 <= rr + dr < 7 and 0 <= cc + dc < 9:
                        q, qp = rr * 9 + cc, (rr + dr) * 9 + cc + dc
                        nb |= bool(tiles[s, t + 1, q] != tiles[s, t, qp]) or bool((mobs[s, t + 1, q] != mobs[s, t, qp]).any())
                idx_prev.append((s * 17 + t) * 81 + jp); idx_next.append((s * 17 + t + 1) * 81 + j)
                strat.append((bool(light[s, t] < .5), (dr, dc) != (0, 0), nb))
    idx_prev, idx_next, strat = torch.tensor(idx_prev), torch.tensor(idx_next), torch.tensor(strat)
    cells = torch.tensor([j for j in range(63) if j != 31])
    train_roots = set(pick[:80])
    tr_seq = torch.tensor([s[2] in train_roots for s in seqs])
    res = {'sequences': n, 'persistent_cells': len(idx_prev), 'night_share': round(float(strat[:, 0].float().mean()), 4), 'reps': {}}
    for name, z in reps.items():
        z = T17(z.float())
        flat = z.view(-1, 192)
        d = (flat[idx_next] - flat[idx_prev]).square().mean(-1).sqrt()
        perm = torch.randperm(len(idx_prev), generator=torch.Generator().manual_seed(1))
        scale = float((flat[idx_prev] - flat[idx_prev[perm]]).square().mean(-1).sqrt().mean())
        row = {}
        for nm, mk in (('day_still', ~strat[:, 0] & ~strat[:, 1] & ~strat[:, 2]), ('day_moved', ~strat[:, 0] & strat[:, 1] & ~strat[:, 2]),
                       ('day_still_nbchanged', ~strat[:, 0] & ~strat[:, 1] & strat[:, 2]),
                       ('night_still', strat[:, 0] & ~strat[:, 1] & ~strat[:, 2]), ('night_moved', strat[:, 0] & strat[:, 1])):
            row[f'drift_{nm}'] = round(float(d[mk].mean() / scale), 4) if mk.any() else None
        X = z[:, :, cells].reshape(-1, 192)
        ytile = tiles[:, :, cells].reshape(-1); ymob = mobs[:, :, cells].reshape(-1, 7)
        seq_of = torch.arange(n)[:, None, None].expand(n, 17, len(cells)).reshape(-1)
        tr = tr_seq[seq_of]
        sub = torch.Generator().manual_seed(2)
        tri = torch.where(tr)[0]; tri = tri[torch.randperm(len(tri), generator=sub)[:60000]]
        tei = torch.where(~tr)[0]; tei = tei[torch.randperm(len(tei), generator=sub)[:40000]]
        row['probe_tile_acc'] = round(float((RL.ridge(X, F.one_hot(ytile, 17).float(), tri, tei).argmax(-1) == ytile[tei]).float().mean()), 4)
        for ch, nm in ((0, 'zombie'), (1, 'cow'), (2, 'skeleton')):
            row[f'probe_{nm}_auc'] = round(RL.auc(RL.ridge(X, ymob[:, ch].float()[:, None], tri, tei)[:, 0], ymob[tei, ch]), 4)
        H = z[:, :, 63:].reshape(n * 17, -1); y = health.reshape(-1, 1)
        trf = torch.where(tr_seq[:, None].expand(n, 17).reshape(-1))[0]; tef = torch.where(~tr_seq[:, None].expand(n, 17).reshape(-1))[0]
        ph = RL.ridge(H, y, trf, tef)[:, 0]
        yt = y[tef, 0]
        row['hud_health_r2'] = round(float(1 - ((ph - yt) ** 2).sum() / ((yt - yt.mean()) ** 2).sum()), 4)
        row['scale_rms_between_cells'] = round(scale, 4)
        res['reps'][name] = row
        print(json.dumps({'rep': name, **row}), flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'content_audit.json').write_text(json.dumps(res, indent=1))


if __name__ == '__main__':
    main()
