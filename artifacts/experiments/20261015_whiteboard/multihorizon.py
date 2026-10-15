"""R54 (MULTIHORIZON_PREREG.md governs): multi-horizon EXPECTED-consequence heads on frozen beliefs (2026-10-16; GPU)

Fit (training pool, label-free): 30,000 seeded roots (trajectory r, start s <= 40, root at s + 7). The frozen backbone reads the
8-frame context (frozen Raw tokens, actions at frames s..s+7, the last being the first action), giving the belief at the 18 HUD
cells of the root and the action token. Targets: the observed HUD tokens at root + k, k in K. Heads: ridge per horizon (primary;
lambda 1e-3 N) from [cell belief, action token, cell one-hot, 1] to the cell's token; one MLP shared across cells and horizons
(secondary). Evaluation on DV16's 200 roots (dv16.load_roots), every first action: expected health (the linear bank HUD reader on
the predicted HUD tokens, no LN) at each horizon; risk = max_k (h_root - h_k)+; within-root AUC over noop + 4 moves vs the real P16
order; DV16 expected safe 5-way and 17-way; paired bootstrap vs the rollouts' HL (dv16_<world>.pt).
Env MH_ANCHOR=1 (post-hoc R54b, not preregistered): the heads also read the root's current HUD tokens, the anchor D's own
readout has (its reference mixes copy / neighbour / generated candidates) -> multihorizon_anchor.json.
Usage: multihorizon.py <backbone> ... (D_parent_ref pi_s1 pi_s2) -> artifacts/eda/whiteboard/multihorizon.json
"""
import json
import os
import sys
from pathlib import Path

os.environ.setdefault('TRITON_F32_DEFAULT', 'ieee')
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / '20261012_oracle'))
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / '20260927_levers'))

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

import dv16 as DV
from decision_proxies import hud_health

K = (1, 2, 3, 4, 6, 8, 12, 16)
ANCHOR = os.environ.get('MH_ANCHOR') == '1'                       # post-hoc (R54b): the head also reads the root's HUD tokens
TAG = '_anchor' if ANCHOR else ''
N_ROOTS, LAM = 30000, 1e-3
HUD = slice(63, 81)


@torch.no_grad()
def features(world, frames, acts, config):
    """frames [B,8,81,192], acts [B,8] -> HUD-cell belief [B,18,D], action token [B,D] at the root (index 7)"""
    from d4mj.train import autocast_context
    fr = torch.cat([frames, torch.zeros_like(frames[:, :1])], 1).cuda().float()
    ac = F.pad(acts, (0, 1)).cuda()
    with autocast_context(config):
        belief, ha = world.backbone_full(fr, ac)
    return belief[:, 7, HUD].float(), ha[:, 7].float()


def design(b, a, cur=None):
    """b [B,18,D], a [B,D] (cur [B,18,192]: the root's own HUD tokens, post-hoc anchor variant) -> [B,18,2D+19(+192)]"""
    B = len(b)
    eye = torch.eye(18, device=b.device)[None].expand(B, -1, -1)
    parts = [b, a[:, None].expand(-1, 18, -1), eye, torch.ones(B, 18, 1, device=b.device)]
    if ANCHOR:
        parts.insert(0, cur.to(b.device).float())
    return torch.cat(parts, -1)


class Head(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d + len(K), 512), nn.GELU(), nn.Linear(512, 512), nn.GELU(), nn.Linear(512, 192))

    def forward(self, x, k):
        return self.net(torch.cat([x, F.one_hot(k, len(K)).float()], -1))


def auc_rank(score, P):
    num = den = 0.
    for i in range(len(P)):
        for a in range(5):
            for b in range(5):
                if P[i, a] > P[i, b]:
                    den += 1; num += float(score[i, a] > score[i, b]) + .5 * float(score[i, a] == score[i, b])
    return num / den


def auc_roots(score, P):
    out = []
    for i in range(len(P)):
        num = den = 0.
        for a in range(5):
            for b in range(5):
                if P[i, a] > P[i, b]:
                    den += 1; num += float(score[i, a] > score[i, b]) + .5 * float(score[i, a] == score[i, b])
        out.append(num / den if den else float('nan'))
    return torch.tensor(out)


def expected_safe(risk, P, cols):
    v = []
    for i in range(len(P)):
        r, p = risk[i, cols], P[i, cols]
        m = r == r.min()
        v.append(1 - float(p[m].mean()))
    return torch.tensor(v)


def boot(d, g):
    d = d[~d.isnan()]
    idx = torch.randint(len(d), (4000, len(d)), generator=g)
    b = d[idx].mean(1)
    return [round(float(d.mean()), 4), round(float(b.quantile(.025)), 4), round(float(b.quantile(.975)), 4)]


def main():
    import direct_train as BASE
    import audit_s5_direct as A5
    config = BASE.settings()
    bank = torch.load(A5.BASE / 'snap_bank_readers.pt', map_location='cpu', weights_only=False)
    hud_head = bank['heads']['hud']
    _, _, actions, tokens = BASE.load_data()
    g = torch.Generator().manual_seed(20261054)
    rr = torch.randint(tokens.shape[0], (N_ROOTS,), generator=g)
    ss = torch.randint(0, 41, (N_ROOTS,), generator=g)
    targets = torch.zeros(N_ROOTS, len(K), 18, 192, dtype=torch.float16)
    for i, (r, s) in enumerate(zip(rr.tolist(), ss.tolist())):
        targets[i] = torch.from_numpy(np.array(tokens[r, s + 7 + np.array(K), HUD]))

    def batch(b0, n=64):
        rs = list(zip(rr[b0:b0 + n].tolist(), ss[b0:b0 + n].tolist()))
        fr = torch.stack([torch.from_numpy(np.array(tokens[r, s:s + 8])) for r, s in rs]).float()
        return fr, torch.stack([actions[r, s:s + 8] for r, s in rs])
    roots, _ = DV.load_roots()
    enc = DV.RS.Encoder()
    rtoks = torch.stack([enc.recontext(enc.content(r['frames'])).half().cpu() for r in roots])         # [200,8,81,192]
    h0 = hud_health(rtoks[:, -1].float(), hud_head)                                                   # [200]
    del enc; torch.cuda.empty_cache()
    P = torch.stack([r['P'] for r in roots])[..., 2]                                                  # [200,17]
    gb = torch.Generator().manual_seed(20261055)
    res = {'K': list(K), 'roots_fit': N_ROOTS}
    out_path = DV.OUT / f'multihorizon{TAG}.json'
    if out_path.exists():
        res.update(json.loads(out_path.read_text()))
    for name in sys.argv[1:]:
        kind, ckpt, _ = DV.WORLDS[name]
        world = DV.load_world('paired' if kind == 'paired' else 'pi', ckpt)
        Xs = []
        for b0 in range(0, N_ROOTS, 64):
            fr, ac = batch(b0)
            b, a = features(world, fr, ac, config)
            Xs.append(design(b, a, fr[:, 7, HUD]).half().cpu())
        X = torch.cat(Xs)                                                                              # [N,18,d]
        d = X.shape[-1]
        W = []
        for ki in range(len(K)):                                                                       # ridge per horizon
            XtX = torch.zeros(d, d, dtype=torch.float64, device='cuda'); XtY = torch.zeros(d, 192, dtype=torch.float64, device='cuda')
            for b0 in range(0, N_ROOTS, 2048):
                x = X[b0:b0 + 2048].flatten(0, 1).cuda().double(); y = targets[b0:b0 + 2048, ki].flatten(0, 1).cuda().double()
                XtX += x.T @ x; XtY += x.T @ y
            n = N_ROOTS * 18
            W.append(torch.linalg.solve(XtX + LAM * n * torch.eye(d, dtype=torch.float64, device='cuda'), XtY).float())
        head = Head(d).cuda()
        opt = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=0.01)
        gm = torch.Generator().manual_seed(20261056)
        for u in range(3000):
            i = torch.randint(N_ROOTS, (1024,), generator=gm); c = torch.randint(18, (1024,), generator=gm); k = torch.randint(len(K), (1024,), generator=gm)
            pred = head(X[i, c].cuda().float(), k.cuda())
            loss = F.mse_loss(pred, targets[i, k, c].cuda().float())
            opt.zero_grad(); loss.backward(); opt.step()
        head.eval()
        # evaluation: every first action on DV16's roots
        risk = {'ridge': torch.zeros(200, 17), 'mlp': torch.zeros(200, 17)}
        per_k = {'ridge': torch.zeros(200, 17, len(K))}
        with torch.no_grad():
            for i, r in enumerate(roots):
                a_ctx = torch.stack([torch.cat([r['led'][1:8], torch.tensor([a])]) for a in range(17)])     # [17,8]
                b, a = features(world, rtoks[i][None].expand(17, -1, -1, -1).float(), a_ctx, config)
                x = design(b, a, rtoks[i][7, HUD][None].expand(17, -1, -1).float())                     # [17,18,d]
                for hname in ('ridge', 'mlp'):
                    hk = []
                    for ki in range(len(K)):
                        pt = x @ W[ki] if hname == 'ridge' else head(x, torch.full(x.shape[:2], ki, device='cuda'))
                        full = torch.zeros(17, 81, 192, device='cuda'); full[:, HUD] = pt
                        hk.append(hud_health(full.cpu(), hud_head))
                    hk = torch.stack(hk, -1)                                                           # [17,K]
                    loss_k = (h0[i] - hk).clamp_min(0)
                    risk[hname][i] = loss_k.amax(-1)
                    if hname == 'ridge':
                        per_k['ridge'][i] = loss_k
        out = {}
        for hname, rk in risk.items():
            o = {'auc5': round(auc_rank(rk[:, :5], P[:, :5]), 4),
                 'es5': round(float(expected_safe(rk, P, list(range(5))).mean()), 4),
                 'es17': round(float(expected_safe(rk, P, list(range(17))).mean()), 4),
                 'sleep_chosen': round(float(sum(float((rk[i] == rk[i].min())[6]) / float((rk[i] == rk[i].min()).sum()) for i in range(200)) / 200), 4),
                 'mean_risk_sleep_vs_moves': [round(float(rk[:, 6].mean()), 3), round(float(rk[:, :5].mean()), 3)]}
            if hname == 'ridge':
                o['auc5_by_k'] = {str(k): round(auc_rank(per_k['ridge'][:, :5, ki], P[:, :5]), 4) for ki, k in enumerate(K)}
            a_head = auc_roots(rk[:, :5], P[:, :5])
            for ref in ('D_parent_ref', 'pi_s1', 'pi_s2'):
                R = torch.load(DV.OUT / f'dv16_{ref}.pt', weights_only=False)
                hl = torch.stack([row['HL'] for row in R])
                o[f'auc5_minus_{ref}_rollout_HL'] = boot(a_head - auc_roots(hl[:, :5], P[:, :5]), gb)
                o[f'es5_minus_{ref}_rollout_HL'] = boot(expected_safe(rk, P, list(range(5))) - expected_safe(hl, P, list(range(5))), gb)
            out[hname] = o
        res[name] = out
        print(json.dumps({name: out}, indent=1), flush=True)
        out_path.write_text(json.dumps(res, indent=1))
        del world; torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
