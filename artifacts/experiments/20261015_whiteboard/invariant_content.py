"""Does a lighting-invariant content code exist? (2026-10-15 late, user programme; CPU; exact simulator state used to select
persistent pairs and for probe labels -- a diagnostic upper bound; a label-free version would take the pairs from PI's own
persistence decisions)

Inputs: the content code (LN of the frozen encoder's position-free patch embedding) and, for comparison, the Raw token, on
content_audit.py's data (diagnosis_futures_v1, 120 seeded roots; roots 0-79 fit, 80-119 test). Persistent pairs = the same exact
content at consecutive frames (camera-aligned), spanning day / night rendering changes and camera moves.
  lin_k     generalized-eigen projection to k dims maximizing between-cell variance over persistent-pair variance
            (S_b w = lambda S_w w, top k), k in 32 / 64 / 128
  vicreg    an MLP 192 -> 256 -> 128 trained on persistent pairs: invariance (MSE between the pair's codes) + variance hinge
            (std >= 1 per dim) + covariance penalty (VICReg weights 25 / 25 / 1), 3,000 steps, batch 1,024, Adam 1e-3
Test (held-out roots): drift of persistent content relative to between-cell scale (day still / day moved / night), ridge probes
(tile class accuracy; zombie / cow / skeleton AUC).
Usage: invariant_content.py -> artifacts/eda/whiteboard/invariant_content.json
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

import content_audit as CA
import rep_layers as RL

OUT = HERE.parents[1] / 'eda/whiteboard'


def evaluate(z, idx_prev, idx_next, strat, pair_test, tiles, mobs, cells, tr_seq, n):
    flat = z.reshape(-1, z.shape[-1])
    d = (flat[idx_next] - flat[idx_prev]).square().mean(-1).sqrt()
    pt = torch.where(pair_test)[0]
    perm = pt[torch.randperm(len(pt), generator=torch.Generator().manual_seed(1))]
    scale = float((flat[idx_prev[pt]] - flat[idx_prev[perm]]).square().mean(-1).sqrt().mean())
    row = {}
    for nm, mk in (('day_still', ~strat[:, 0] & ~strat[:, 1]), ('day_moved', ~strat[:, 0] & strat[:, 1]), ('night', strat[:, 0])):
        mk = mk & pair_test
        row[f'drift_{nm}'] = round(float(d[mk].mean() / scale), 4) if mk.any() else None
    z17 = z.view(n, 17, 81, -1)
    X = z17[:, :, cells].reshape(-1, z.shape[-1])
    ytile = tiles[:, :, cells].reshape(-1); ymob = mobs[:, :, cells].reshape(-1, 7)
    seq_of = torch.arange(n)[:, None, None].expand(n, 17, len(cells)).reshape(-1)
    tr = tr_seq[seq_of]
    sub = torch.Generator().manual_seed(2)
    tri = torch.where(tr)[0]; tri = tri[torch.randperm(len(tri), generator=sub)[:60000]]
    tei = torch.where(~tr)[0]; tei = tei[torch.randperm(len(tei), generator=sub)[:40000]]
    row['probe_tile_acc'] = round(float((RL.ridge(X, F.one_hot(ytile, 17).float(), tri, tei).argmax(-1) == ytile[tei]).float().mean()), 4)
    for ch, nm in ((0, 'zombie'), (1, 'cow'), (2, 'skeleton')):
        row[f'probe_{nm}_auc'] = round(RL.auc(RL.ridge(X, ymob[:, ch].float()[:, None], tri, tei)[:, 0], ymob[tei, ch]), 4)
    return row


def main():
    torch.set_num_threads(16)
    import rollouts as Rr
    files = sorted(Rr.STORE.glob('seed-*.pt'))
    roots = [r for f in files for r in torch.load(f, weights_only=False)]
    pick = torch.randperm(len(roots), generator=torch.Generator().manual_seed(20261015))[:120].tolist()
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
    content, raw = CA.represent(enc, frames.float() / 255.0)
    tiles, mobs, light = RL.facts(vis)
    T17 = lambda x: x.view(n, 17, *x.shape[1:])
    tiles, mobs, light = T17(tiles), T17(mobs), T17(light)
    idx_prev, idx_next, strat, pseq = [], [], [], []
    for s in range(n):
        for t in range(16):
            dr, dc = RL.camera(tiles[s, t], tiles[s, t + 1])
            for r in range(7):
                for c in range(9):
                    if not (0 <= r + dr < 7 and 0 <= c + dc < 9):
                        continue
                    j, jp = r * 9 + c, (r + dr) * 9 + c + dc
                    if 31 in (j, jp) or bool(tiles[s, t + 1, j] != tiles[s, t, jp]) or bool((mobs[s, t + 1, j] != mobs[s, t, jp]).any()):
                        continue
                    idx_prev.append((s * 17 + t) * 81 + jp); idx_next.append((s * 17 + t + 1) * 81 + j)
                    strat.append((bool(light[s, t] < .5), (dr, dc) != (0, 0))); pseq.append(s)
    idx_prev, idx_next, strat = torch.tensor(idx_prev), torch.tensor(idx_next), torch.tensor(strat)
    train_roots = set(pick[:80])
    tr_seq = torch.tensor([s[2] in train_roots for s in seqs])
    pair_train = tr_seq[torch.tensor(pseq)]; pair_test = ~pair_train
    cells = torch.tensor([j for j in range(63) if j != 31])
    res = {'sequences': n, 'pairs': len(idx_prev), 'night_share': round(float(strat[:, 0].float().mean()), 4), 'variants': {}}
    for base_name, base in (('content', content), ('raw', raw)):
        flat = base.float().reshape(-1, 192)
        row = evaluate(flat, idx_prev, idx_next, strat, pair_test, tiles, mobs, cells, tr_seq, n)
        res['variants'][base_name] = row; print(json.dumps({'variant': base_name, **row}), flush=True)
        # generalized eigen projection, fitted on training pairs / training cells
        dp = (flat[idx_next[pair_train]] - flat[idx_prev[pair_train]]).double()
        Sw = dp.T @ dp / len(dp) + 1e-4 * torch.eye(192, dtype=torch.float64)
        trcells = flat.view(n, 17, 81, 192)[tr_seq][:, :, :63].reshape(-1, 192).double()
        Sb = torch.cov(trcells.T)
        L = torch.linalg.cholesky(Sw)
        Linv = torch.linalg.inv(L)
        evals, evecs = torch.linalg.eigh(Linv @ Sb @ Linv.T)
        Wall = (Linv.T @ evecs).flip(-1)                                             # columns by descending ratio
        for k in (32, 64, 128):
            z = (flat.double() @ Wall[:, :k]).float()
            row = evaluate(z, idx_prev, idx_next, strat, pair_test, tiles, mobs, cells, tr_seq, n)
            res['variants'][f'{base_name}_lin{k}'] = row; print(json.dumps({'variant': f'{base_name}_lin{k}', **row}), flush=True)
        # VICReg MLP on training pairs
        torch.manual_seed(0)
        net = torch.nn.Sequential(torch.nn.Linear(192, 256), torch.nn.GELU(), torch.nn.Linear(256, 128))
        opt = torch.optim.Adam(net.parameters(), lr=1e-3)
        ptr = torch.where(pair_train)[0]
        g = torch.Generator().manual_seed(3)
        for step in range(3000):
            b = ptr[torch.randint(len(ptr), (1024,), generator=g)]
            za, zb = net(flat[idx_prev[b]]), net(flat[idx_next[b]])
            inv = F.mse_loss(za, zb)
            var = sum(F.relu(1 - torch.sqrt(x.var(0) + 1e-4)).mean() for x in (za, zb))
            cov = 0.
            for x in (za, zb):
                xc = x - x.mean(0)
                c = (xc.T @ xc) / (len(x) - 1)
                cov = cov + (c - torch.diag(torch.diag(c))).pow(2).sum() / x.shape[1]
            loss = 25 * inv + 25 * var + cov
            opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            z = torch.cat([net(flat[i:i + 65536]) for i in range(0, len(flat), 65536)])
        row = evaluate(z, idx_prev, idx_next, strat, pair_test, tiles, mobs, cells, tr_seq, n)
        res['variants'][f'{base_name}_vicreg'] = row; print(json.dumps({'variant': f'{base_name}_vicreg', **row}), flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'invariant_content.json').write_text(json.dumps(res, indent=1))


if __name__ == '__main__':
    main()
