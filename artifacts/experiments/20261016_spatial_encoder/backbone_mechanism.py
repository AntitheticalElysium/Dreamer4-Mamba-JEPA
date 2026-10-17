"""BACKBONE_MECHANISM_PROTOCOL.md Part A: translation consistency, static drift and absolute-position share of backbone-LN tokens.
Read-only (frozen encoders, real diagnosis frames, exact simulator labels for evaluation only).

Panel: every diagnosis root, the factual future (f = 0), consecutive frames j -> j+1 (alive). Exact camera shift from exact tiles
(objective_audit.shift_of); sources() maps next cells to world sources. Persisting cell: source in view, same tile and same mob layer
at source and destination, neither the player cell (31).
  A1 transport ratio (camera-moved pairs)  sum |z_{j+1}[c] - z_j[src c]|^2 / sum |z_{j+1}[c] - z_j[c]|^2      (lower = moves with content)
  A2 static drift (camera-static pairs)    sum |z_{j+1}[c] - z_j[c]|^2 / (cells x total per-token variance)
  A3 position share                         (RSS by (tile, mob bits) - RSS by (tile, mob bits, cell)) / TSS over all panel map tokens
Episode-cluster bootstrap (2,000 draws for A1 / A2; 200 for A3), paired across encoders.
Usage: backbone_mechanism.py repr -> artifacts/eda/backbone_mechanism_v1/repr.json
"""
import json
import os
import sys
from pathlib import Path

os.environ.setdefault('TRITON_F32_DEFAULT', 'ieee')
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import spatial_eval as EV                                                      # noqa: E402
import objective_audit as OA                                                   # noqa: E402

import torch                                                                   # noqa: E402

OUTD = EV.OUT.parent / 'backbone_mechanism_v1'
S2 = EV.OUT
ENC = {'raw': 'raw@10000',
       **{f'mop_s1@{k}': f'spatial-bb:{S2}/stage2_mop_sig_s1/step-{k:06d}.pt' for k in (500, 1000, 2000, 4000, 10000)},
       'mop_s2@10000': f'spatial-bb:{S2}/stage2_mop_sig_s2/step-010000.pt',
       'full_s1@10000': f'spatial-bb:{S2}/stage2_full_sig_s1/step-010000.pt',
       'full_s2@10000': f'spatial-bb:{S2}/stage2_full_sig_s2/step-010000.pt'}
G1, NC = 17 * 4, 63


@torch.no_grad()
def repr_main():
    encs = {k: EV.PatchState(v) for k, v in ENC.items()}
    eps = []
    acc = {k: {'mw': [], 'ms': [], 'dn': [], 'dc': [], 'sum': torch.zeros(G1 * NC, 192, dtype=torch.float64),
               'n': torch.zeros(G1 * NC, dtype=torch.float64), 'sq': 0.0, 'tot': torch.zeros(192, dtype=torch.float64), 'cnt': 0,
               'ep_sum': {}, 'ep_n': {}, 'ep_sq': {}} for k in encs}
    rows = (r for f in sorted(OA.DIAG.glob('seed-*.pt')) for r in torch.load(f, weights_only=False))
    for r in rows:
        alive = int((~r['future_dead'][0].bool()).long().cumprod(0).sum())
        if alive < 1:
            continue
        frames = torch.cat([r['context'][-1:], r['future_frames'][0][:alive]])
        vis = torch.cat([r['root_visible'][None], r['future_visible'][0][:alive]])
        T, L, _ = OA.layout(vis)
        ep = int(r['seed']); eps.append(ep)
        pairs = []
        for j in range(len(frames) - 1):
            dr, dc = OA.shift_of(T[j], T[j + 1]); src = OA.sources(dr, dc)
            ok = src >= 0
            s = src.clamp_min(0)
            pers = ok & (T[j + 1] == T[j][s]) & (L[j + 1] == L[j][s]) & (torch.arange(63) != 31) & (s != 31)
            pairs.append((j, (dr, dc) != (0, 0), s, pers))
        g = (T * 4 + (L & 3)).long() * NC + torch.arange(NC)[None]                 # (tile, mob bits, cell) group per map cell
        for k, enc in encs.items():
            z = enc(frames)[:, :63].double().cpu()                                  # [n,63,192]
            A = acc[k]
            mw = ms = dn = 0.0; dcn = 0
            for j, moved, s, pers in pairs:
                if not bool(pers.any()):
                    continue
                if moved:
                    mw += float((z[j + 1][pers] - z[j][s][pers]).pow(2).sum()); ms += float((z[j + 1][pers] - z[j][pers]).pow(2).sum())
                else:
                    dn += float((z[j + 1][pers] - z[j][pers]).pow(2).sum()); dcn += int(pers.sum())
            A['mw'].append(mw); A['ms'].append(ms); A['dn'].append(dn); A['dc'].append(dcn)
            flat = z.reshape(-1, 192); gf = g.reshape(-1)
            A['sum'].index_add_(0, gf, flat); A['n'].index_add_(0, gf, torch.ones(len(gf), dtype=torch.float64))
            A['sq'] += float(flat.pow(2).sum()); A['tot'] += flat.sum(0); A['cnt'] += len(flat)
            es = A['ep_sum'].setdefault(ep, torch.zeros(G1 * NC, 192, dtype=torch.float32))
            es.index_add_(0, gf, flat.float())
            A['ep_n'].setdefault(ep, torch.zeros(G1 * NC)).index_add_(0, gf, torch.ones(len(gf)))
            A['ep_sq'][ep] = A['ep_sq'].get(ep, 0.0) + float(flat.pow(2).sum())
    ueps = sorted(set(eps)); eo = {e: i for i, e in enumerate(ueps)}
    epi = torch.tensor([eo[e] for e in eps])
    gen = torch.Generator().manual_seed(20261111)
    draws = torch.randint(len(ueps), (2000, len(ueps)), generator=gen)
    mult = torch.zeros(2000, len(ueps)).scatter_add_(1, draws, torch.ones(2000, len(ueps)))
    res = {'roots': len(eps), 'episodes': len(ueps), 'encoders': ENC}
    boot = {}
    for k, A in acc.items():
        var_tok = (A['sq'] - float((A['tot'] ** 2).sum()) / A['cnt']) / A['cnt']   # total per-token variance (sum over dims)
        per_ep = lambda key: torch.zeros(len(ueps), dtype=torch.float64).index_add_(0, epi, torch.tensor(A[key], dtype=torch.float64))
        MW, MS, DN, DC = per_ep('mw'), per_ep('ms'), per_ep('dn'), per_ep('dc')
        a1 = float(MW.sum() / MS.sum()); a2 = float(DN.sum() / (DC.sum() * var_tok))
        boot[(k, 'A1')] = (mult.double() @ MW) / (mult.double() @ MS)
        boot[(k, 'A2')] = (mult.double() @ DN) / ((mult.double() @ DC) * var_tok)
        g1 = lambda s_, n_: (s_.view(G1, NC, 192).sum(1), n_.view(G1, NC).sum(1))
        def rss(s_, n_):
            return -float((s_.pow(2).sum(-1) / n_.clamp_min(1)).sum())
        tss = A['sq'] - float((A['tot'] ** 2).sum()) / A['cnt']
        s1, n1 = g1(A['sum'], A['n'])
        a3 = (rss(s1, n1) - rss(A['sum'], A['n'])) / tss
        res[k] = {'A1_transport_ratio': round(a1, 4), 'A2_static_drift': round(a2, 4), 'A3_position_share': round(a3, 4),
                  'moved_pair_cells_energy_denominator': round(float(MS.sum()), 1), 'static_persisting_cells': int(DC.sum())}
        # A3 bootstrap (200 draws), episode-replicated group sums
        ES = torch.stack([A['ep_sum'][e] for e in ueps]); EN = torch.stack([A['ep_n'][e] for e in ueps]); ESQ = torch.tensor([A['ep_sq'][e] for e in ueps], dtype=torch.float64)
        b3 = []
        for b in range(200):
            w = mult[b].float()
            S_ = torch.einsum('e,egd->gd', w, ES).double(); N_ = (w[:, None] * EN).sum(0).double()
            tot_ = S_.sum(0); cnt_ = float(N_.sum()); sq_ = float((w.double() * ESQ).sum())
            tss_ = sq_ - float((tot_ ** 2).sum()) / cnt_
            s1_, n1_ = g1(S_, N_)
            b3.append((rss(s1_, n1_) - rss(S_, N_)) / tss_)
        boot[(k, 'A3')] = torch.tensor(b3, dtype=torch.float64)
        print(json.dumps({k: res[k]}), flush=True)
    q = lambda v: [round(float(v.quantile(.025)), 4), round(float(v.quantile(.975)), 4)]
    for k in acc:
        if k == 'raw':
            continue
        res[f'{k}_minus_raw'] = {m: q(boot[(k, m)][:len(boot[('raw', m)])] - boot[('raw', m)]) for m in ('A1', 'A2', 'A3')}
    OUTD.mkdir(parents=True, exist_ok=True)
    (OUTD / 'repr.json').write_text(json.dumps(res, indent=1))
    print(json.dumps({k: v for k, v in res.items() if k.endswith('_minus_raw')}, indent=1))


if __name__ == '__main__':
    {'repr': repr_main}[sys.argv[1]]()
