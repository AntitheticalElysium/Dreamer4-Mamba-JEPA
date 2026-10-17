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




# ----------------------------------------------------------------------------------------------------------------- Part B
def _groups(name):
    import re
    m = re.search(r'(?:blocks|layers|layer)\.(\d+)\.', name)
    if m:
        b = int(m.group(1))
        return 'blocks_1-4' if b < 4 else ('blocks_5-8' if b < 8 else 'blocks_9-12')
    return 'embed_pos_cls_norm'


def _panel(n_pairs=256, n_frames=512, seed=20261112):
    """fixed panel: camera-moved transitions with persisting cells; labelled map frames (fit roots of split_v2)"""
    split = json.loads((EV.OUT / 'split_v2.json').read_text())
    fit = set(split['fit'])
    pairs, frames, labels = [], [], []
    ri = 0
    for f in sorted(OA.DIAG.glob('seed-*.pt')):
        for r in torch.load(f, weights_only=False):
            if ri in fit and not bool(r['future_dead'][0][0]):
                fr = torch.cat([r['context'][-1:], r['future_frames'][0][0:1]])
                vis = torch.cat([r['root_visible'][None], r['future_visible'][0][0:1]])
                T, L, _ = OA.layout(vis)
                dr, dc = OA.shift_of(T[0], T[1]); src = OA.sources(dr, dc); s = src.clamp_min(0)
                pers = (src >= 0) & (T[1] == T[0][s]) & (L[1] == L[0][s]) & (torch.arange(63) != 31) & (s != 31)
                if (dr, dc) != (0, 0) and bool(pers.any()) and len(pairs) < n_pairs:
                    pairs.append((fr, s, pers))
                if len(frames) < n_frames:
                    frames.append(fr[0]); labels.append(((L[0] & 1) > 0, (L[0] & 2) > 0, T[0] * 4 + (L[0] & 3)))
            ri += 1
    return pairs, torch.stack(frames), labels


def _metrics(enc, pairs, frames, labels, W=None):
    """differentiable A1 (transport ratio), A3 proxy (within-content position share) and mob-probe loss on the panel"""
    from torch.nn import functional as F
    def tok(x):
        _, _, t, _, _ = enc._hidden(x.cuda()[:, None])
        return F.layer_norm(t.float(), (t.shape[-1],))[:, :63]
    num = den = 0.
    for i in range(0, len(pairs), 64):
        ch = pairs[i:i + 64]
        z = tok(torch.cat([p[0] for p in ch]))                                     # [2n,63,D]
        for k, (_, s, pers) in enumerate(ch):
            z0, z1 = z[2 * k], z[2 * k + 1]
            pm = pers.cuda(); sc = s.cuda()
            num = num + (z1[pm] - z0[sc][pm]).pow(2).sum(); den = den + (z1[pm] - z0[pm]).pow(2).sum()
    a1 = num / den
    zs = torch.cat([tok(frames[i:i + 128]) for i in range(0, len(frames), 128)])  # [n,63,D]
    g1 = torch.stack([lab[2] for lab in labels]).cuda().long()                     # (tile, mob bits)
    g2 = g1 * 63 + torch.arange(63, device='cuda')[None]
    X = zs.reshape(-1, zs.shape[-1])
    def rss(g, G):
        n = torch.zeros(G, device='cuda').index_add_(0, g, torch.ones(len(g), device='cuda'))
        s_ = torch.zeros(G, X.shape[1], device='cuda').index_add_(0, g, X)
        return -(s_.pow(2).sum(-1) / n.clamp_min(1)).sum()
    tss = X.pow(2).sum() - X.sum(0).pow(2).sum() / len(X)
    a3 = (rss(g1.reshape(-1), 68) - rss(g2.reshape(-1), 68 * 63)) / tss
    y = torch.stack([torch.stack([lab[0], lab[1]], -1) for lab in labels]).reshape(-1, 2).float().cuda()
    if W is None:
        Xd = torch.cat([X.detach().double(), torch.ones(len(X), 1, device='cuda', dtype=torch.float64)], 1)
        A = Xd.T @ Xd; A += 1e-2 * torch.trace(A) / len(A) * torch.eye(len(A), device='cuda', dtype=torch.float64)
        W = torch.linalg.solve(A, Xd.T @ y.double()).float()
    probe = ((torch.cat([X, torch.ones(len(X), 1, device='cuda')], 1) @ W - y).pow(2).mean())
    return {'A1': a1, 'A3': a3, 'probe': probe}, W


def grad_main():
    import dataclasses
    import spatial_encoder as SE
    from torch.nn import functional as F
    ckpts = [(s, k) for s in (1, 2) for k in (1000, 2000, 4000, 10000)]
    base = SE.load_recipe(SE.RECIPE)
    episodes, _ = SE.load_joint_corpus(SE.CORPUS, base)
    sampler = SE.JointSampler(episodes, base, torch.Generator().manual_seed(20261110))
    batches = [sampler.sample() for _ in range(4)]
    del episodes, sampler
    pairs, pframes, plabels = _panel()
    out = {'batches': 4, 'panel_pairs': len(pairs), 'panel_frames': len(pframes)}
    dev = 'cuda'
    for s, k in ckpts:
        ck = torch.load(S2 / f'stage2_mop_sig_s{s}' / f'step-{k:06d}.pt', map_location=dev, weights_only=False)
        config = dataclasses.replace(base, seed=20260730 + s)
        torch.manual_seed(0)
        bundle = SE.ModelBundle.create(config); SE.set_phase_mode(bundle, 'joint')
        proj = SE.LeWMProjector(SE.D, config.encoder).to(dev); pred = SE.PatchPredictor(4).to(dev); idm = SE.IDMHead().to(dev)
        bundle.encoder.load_state_dict(ck['encoder']); bundle.world.load_state_dict(ck['world'])
        proj.load_state_dict(ck['patch_proj']); pred.load_state_dict(ck['predictor']); idm.load_state_dict(ck['idm'])
        opt = SE.optimizer([bundle.encoder, bundle.world, proj, pred, idm], config.joint, exclude_vectors=True)
        opt.load_state_dict(ck['optimizer'])
        for m in (bundle.encoder, bundle.world, proj, pred):
            m.train()
        named = [(n, p) for n, p in bundle.encoder.named_parameters() if p.requires_grad]
        params = [p for _, p in named]
        sig_cls = SE.SIGReg(config.joint.knots, config.joint.projections).to(dev)
        sig_patch = SE.SIGReg(config.joint.knots, config.joint.projections).to(dev)
        terms = ('cls_pred', 'cls_sigreg', 'patch_mse_input', 'patch_mse_target', 'router_ce', 'patch_sigreg')
        G = {t: [torch.zeros_like(p, dtype=torch.float32) for p in params] for t in terms}
        for bi, batch in enumerate(batches):
            batch = batch.to(dev)
            frames, acts = batch.frames, batch.actions
            B, T = frames.shape[:2]
            proj_rng = torch.Generator(device=dev).manual_seed(77 + bi); patch_rng = torch.Generator(device=dev).manual_seed(88 + bi)
            with SE.autocast_context(config):
                zc, _, tokens, _, _ = bundle.encoder._hidden(frames)
                zp = proj(tokens.reshape(-1, SE.D)).view(-1, T, SE.P, SE.D)
                z = zc.view(-1, T, 1, SE.D)
                pred_cls = bundle.world.teacher(z, acts).predicted
                prev = zp[:, 0:2].flatten(0, 1); cur = zp[:, 1:3].flatten(0, 1); tgt = zp[:, 2:4].flatten(0, 1)
                cand, router = pred(prev, cur, acts[:, 1:3].flatten(0, 1))
            with torch.autocast(device_type='cuda', enabled=False):
                cls_pred = (pred_cls.float() - z[:, 1:].float()).square().mean()
                cls_reg = sig_cls(z[:, :, 0].float().transpose(0, 1), proj_rng)
                err = (cand.float() - tgt.float()[:, :, None]).square().mean(-1)
                best = err.argmin(-1)
                e_in = (cand.float() - tgt.float().detach()[:, :, None]).square().mean(-1).gather(-1, best[..., None]).mean()
                e_tg = (cand.float().detach() - tgt.float()[:, :, None]).square().mean(-1).gather(-1, best[..., None]).mean()
                router_ce = F.cross_entropy(router.float().flatten(0, 1), best.flatten())
                flat = zp.float().permute(1, 0, 2, 3).reshape(T, B * SE.P, SE.D)
                pick = torch.randperm(B * SE.P, device=dev, generator=patch_rng)[:SE.SIG_N]
                ac = sig_patch(flat[:, pick], patch_rng)
            vals = {'cls_pred': cls_pred, 'cls_sigreg': 0.09 * cls_reg, 'patch_mse_input': e_in, 'patch_mse_target': e_tg,
                    'router_ce': 0.1 * router_ce, 'patch_sigreg': 0.09 * 128 / SE.SIG_N * ac}
            for t in terms:
                gs = torch.autograd.grad(vals[t], params, retain_graph=True, allow_unused=True)
                for acc_, g in zip(G[t], gs):
                    if g is not None:
                        acc_ += g.float()
            del zc, tokens, zp, z, pred_cls, cand, router, err, vals
        # Adam preconditioning (saved second moments)
        beta2, eps = opt.param_groups[0]['betas'][1], opt.param_groups[0]['eps']
        pre = []
        for p in params:
            st = opt.state[p]
            step = float(st['step']) if 'step' in st else float(ck['update'])
            pre.append(1.0 / ((st['exp_avg_sq'].float() / (1 - beta2 ** step)).sqrt() + eps))
        U = {t: [g * c for g, c in zip(G[t], pre)] for t in terms}
        res = {}
        for kind, D_ in (('raw', G), ('adam', U)):
            tot = [sum(D_[t][i] for t in terms) for i in range(len(params))]
            nt = sum(float(x.pow(2).sum()) for x in tot)
            r = {'total_norm': round(nt ** .5, 6)}
            for t in terms:
                dot = sum(float((D_[t][i] * tot[i]).sum()) for i in range(len(params)))
                nn_ = sum(float(D_[t][i].pow(2).sum()) for i in range(len(params)))
                r[t] = {'share': round(dot / nt, 4), 'norm_ratio': round((nn_ / nt) ** .5, 4), 'cos': round(dot / ((nn_ * nt) ** .5 + 1e-30), 4)}
            grp = {}
            for gname in sorted(set(_groups(n) for n, _ in named)):
                idx = [i for i, (n, _) in enumerate(named) if _groups(n) == gname]
                ntg = sum(float(tot[i].pow(2).sum()) for i in idx)
                grp[gname] = {t: round(sum(float((D_[t][i] * tot[i]).sum()) for i in idx) / max(ntg, 1e-30), 4) for t in terms}
            r['by_group_share'] = grp
            res[kind] = r
        # first-order effects of each term's preconditioned update (-u_t) on the panel metrics
        bundle.encoder.eval()
        M, W = _metrics(bundle.encoder, pairs, pframes, plabels)
        eff = {}
        for mname, mval in M.items():
            gm = torch.autograd.grad(mval, params, retain_graph=True, allow_unused=True)
            gm = [g.float() if g is not None else torch.zeros_like(p) for g, p in zip(gm, params)]
            e = {t: -sum(float((gm[i] * U[t][i]).sum()) for i in range(len(params))) for t in terms}
            tot_e = sum(e.values())
            eff[mname] = {'value': round(float(mval), 5), 'effect_per_term': {t: float(v) for t, v in e.items()}, 'total': float(tot_e),
                          'share_of_abs': {t: round(v / max(sum(abs(x) for x in e.values()), 1e-30), 4) for t, v in e.items()}}
        res['first_order_effects'] = eff
        out[f'mop_s{s}@{k}'] = res
        print(json.dumps({f'mop_s{s}@{k}': {'adam_share': {t: res['adam'][t]['share'] for t in terms},
                                             'raw_share': {t: res['raw'][t]['share'] for t in terms},
                                             'A1_effect_share': eff['A1']['share_of_abs'], 'A3_effect_share': eff['A3']['share_of_abs']}}), flush=True)
        del bundle, proj, pred, idm, opt, G, U, ck
        torch.cuda.empty_cache()
    OUTD.mkdir(parents=True, exist_ok=True)
    (OUTD / 'grad.json').write_text(json.dumps(out, indent=1))


if __name__ == '__main__':
    {'repr': repr_main, 'grad': grad_main}[sys.argv[1]]()
