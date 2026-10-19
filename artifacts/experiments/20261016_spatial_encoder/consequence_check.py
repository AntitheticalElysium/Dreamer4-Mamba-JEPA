"""Do a world's predicted states contain attack-related health changes? (FROZEN_STATE_CONTROL_PREREG.md, read-only consequence
check; Raw-predictor primary, joint worlds as context. Evaluation only: exact simulator labels evaluate, never train.)

Units: every diagnosis root x 17 actions with >= 1 alive key (the B1 panel). Target: dh = mean over alive keys of the next shown
health - the root's shown health; attack unit = some key loses >= 1 shown health; attack events = (unit, key) pairs.
States: predicted EXPECTED state (root's 2-frame context, the action; the world's own predictor), COPY (the root state, with the
action one-hot appended: what the input + action announce linearly) and REAL next (per key; the ceiling).
Readers (ridge lambda 1e-2, 5-fold cross-fitting grouped by episode over all roots; the target is next shown health):
  hud   the 18 HUD tokens            -> does the predicted health display change?
  near  HUD + the 3x3 map tokens around the player (27 tokens) -> is attack-related evidence near the player?
Also the world's own Tier B HUD reader (transfer, = B1).
Per-unit predictions are saved for paired episode-bootstrap comparisons (consequence_analysis in this script's --analyse mode).
Usage: consequence_check.py <label>=<encoder spec>,<predictor ckpt>,<probes2> ...  -> erasure/conseq_<label>.pt
       consequence_check.py --analyse <label> ...                              -> erasure/conseq_summary.json
"""
import json
import os
import sys
from pathlib import Path

os.environ.setdefault('TRITON_F32_DEFAULT', 'ieee')
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import spatial_encoder as SE                                                   # noqa: E402
import spatial_eval as EV                                                      # noqa: E402
import spatial_eval_v2 as EV2                                                  # noqa: E402
import tierb_eval as TB                                                        # noqa: E402

import torch                                                                   # noqa: E402

import objective_audit as OA                                                   # noqa: E402

OUTD = EV.OUT / 'erasure'
NEAR = [21, 22, 23, 30, 31, 32, 39, 40, 41]                                   # 3x3 map cells around the player (cell 31)
HUD = list(range(63, 81))
FOLDS, B = 5, 4000


def folds(seeds):
    u = sorted(set(seeds.tolist()))
    perm = torch.randperm(len(u), generator=torch.Generator().manual_seed(20261102)).tolist()
    f = {u[p]: i % FOLDS for i, p in enumerate(perm)}
    return torch.tensor([f[int(s)] for s in seeds])


def crossfit(X, y, fold, lam=1e-2, chunk=4096):
    """cross-fitted ridge = spatial_eval.ridge (intercept, lambda relative to the mean eigenvalue) per held-out fold, with the
    normal equations accumulated in float64 chunks per fold (memory-lean, same estimator); X [n,d], y [n]"""
    n, d = X.shape[0], X.shape[1] + 1
    A = torch.zeros(FOLDS, d, d, dtype=torch.float64); b = torch.zeros(FOLDS, d, dtype=torch.float64)
    for i in range(0, n, chunk):
        xc = torch.cat([X[i:i + chunk].double(), torch.ones(min(chunk, n - i), 1, dtype=torch.float64)], 1)
        yc, fc = y[i:i + chunk].double(), fold[i:i + chunk]
        for k in range(FOLDS):
            m = fc == k
            if bool(m.any()):
                A[k] += xc[m].T @ xc[m]; b[k] += xc[m].T @ yc[m]
    out = torch.empty(n)
    for k in range(FOLDS):
        Ak, bk = A.sum(0) - A[k], b.sum(0) - b[k]
        Ak = Ak + lam * torch.trace(Ak) / d * torch.eye(d, dtype=torch.float64)
        W = torch.linalg.solve(Ak, bk)
        m = torch.nonzero(fold == k)[:, 0]
        for i in range(0, len(m), chunk):
            j = m[i:i + chunk]
            out[j] = (torch.cat([X[j].double(), torch.ones(len(j), 1, dtype=torch.float64)], 1) @ W).float()
    return out


@torch.no_grad()
def collect(label, enc_spec, pred_ck, probe_file):
    enc = EV.PatchState(enc_spec)
    dw = None
    if pred_ck.startswith('d:'):                                                 # an existing D-class world (d_adapter.py)
        from d_adapter import DWorld
        dw = DWorld(pred_ck[2:])
    else:
        sd = torch.load(pred_ck, map_location=EV.DEV, weights_only=False)['predictor']
        pred = SE.PatchPredictor(sd['router.weight'].shape[0]).to(EV.DEV).eval(); pred.load_state_dict(sd)
    probes = torch.load(probe_file, weights_only=False)
    g = torch.Generator(device=EV.DEV).manual_seed(20261071)
    keep = HUD + NEAR
    U = {k: [] for k in ('seed', 'root', 'action', 'h0', 'dh', 'attack', 'keys_alive', 'key_dh', 'key_ok', 'exp', 'copy', 'real', 'transfer')}
    rows = (r for f in sorted(OA.DIAG.glob('seed-*.pt')) for r in torch.load(f, weights_only=False))   # file by file
    for ri, r in enumerate(rows):
        ok = ~r['onestep_dead'].bool()                                           # [4,17]
        if not bool(ok.any()):
            continue
        if dw is not None:                                                       # D: all 4 context frames, the 17 actions
            ctx4 = enc(r['context'])
            acts = torch.cat([r['context_actions'][None].expand(17, -1), torch.arange(17)[:, None]], 1)
            exp = dw.predict(ctx4[None].expand(17, -1, -1, -1), acts); ctx = ctx4[-2:]
        else:
            ctx = enc(r['context'][-2:])
            exp, _, _, _ = TB.step(pred, ctx[0:1].expand(17, -1, -1), ctx[1:2].expand(17, -1, -1), torch.arange(17), g)
        h_tr = TB.read(probes, exp.cpu())[0]                                     # transfer: the Tier B HUD reader
        h_cur = float(TB.read(probes, ctx[1:2].cpu())[0][0])
        real = enc(r['onestep_frames'].flatten(0, 1)).view(4, 17, 81, -1)[:, :, keep].cpu().half()
        _, _, sh0 = OA.layout(r['root_visible'][None]); _, _, sh1 = OA.layout(r['onestep_visible'])
        kdh = (sh1[..., 0] - sh0[0, 0]).float()                                  # [4,17]
        for a in range(17):
            if not bool(ok[:, a].any()):
                continue
            U['seed'].append(int(r['seed'])); U['root'].append(ri); U['action'].append(a); U['h0'].append(float(sh0[0, 0]))
            U['dh'].append(float(kdh[ok[:, a], a].mean())); U['attack'].append(bool((kdh[ok[:, a], a] <= -1).any()))
            U['keys_alive'].append(int(ok[:, a].sum())); U['key_dh'].append(kdh[:, a]); U['key_ok'].append(ok[:, a])
            U['exp'].append(exp[a, keep].cpu().half()); U['copy'].append(ctx[1, keep].cpu().half()); U['real'].append(real[:, a])
            U['transfer'].append(float(h_tr[a]) - h_cur)
    T = {k: (torch.stack(v) if isinstance(v[0], torch.Tensor) else torch.tensor(v)) for k, v in U.items()}
    return T


def fit_predict(T):
    """cross-fitted predicted dh per unit for each (state, reader)"""
    n = len(T['dh']); fold = folds(T['seed'])
    h1 = T['h0'] + T['dh']                                                       # mean next shown health (target)
    act = torch.nn.functional.one_hot(T['action'], 17).float()
    P = {}
    for rd, idx in (('hud', slice(0, 18)), ('near', slice(0, 27))):
        for st in ('exp', 'copy'):
            X = T[st][:, idx].reshape(n, -1).float()
            if st == 'copy':
                X = torch.cat([X, act], 1)                                       # what the input + action announce
            P[f'{st}_{rd}'] = crossfit(X, h1, fold) - T['h0']
        if rd != 'hud':
            continue                                                             # the real-state ceiling uses the HUD reader only
        # real: per-key rows, target per-key next health; per-unit mean of the alive keys' predictions
        kr = T['key_ok'].reshape(-1)
        Xr = T['real'][:, :, idx].reshape(n * 4, -1)[kr]
        yr = (T['h0'][:, None] + T['key_dh']).reshape(-1)[kr]
        fr = fold[:, None].expand(-1, 4).reshape(-1)[kr]
        pr = torch.zeros(n * 4); pr[kr] = crossfit(Xr, yr, fr)
        pr = pr.view(n, 4)
        P[f'real_{rd}'] = (pr * T['key_ok']).sum(1) / T['key_ok'].sum(1) - T['h0']
    P['transfer'] = T['transfer']
    return P


def auc(score, label):
    """tie-aware ROC AUC (Mann-Whitney with average ranks), vectorized"""
    s, l = score.double(), label.bool()
    n1, n0 = int(l.sum()), int((~l).sum())
    if n1 == 0 or n0 == 0:
        return float('nan')
    _, inv, cnt = torch.unique(s, return_inverse=True, return_counts=True)
    rank = (cnt.cumsum(0).double() - (cnt.double() - 1) / 2)[inv]
    return float((rank[l].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def pearson(x, y):
    x, y = x - x.mean(), y - y.mean()
    return float((x * y).sum() / (x.norm() * y.norm()).clamp_min(1e-12))


def metrics(p, T, idx=None):
    idx = torch.arange(len(p)) if idx is None else idx
    p, dh, att = p[idx], T['dh'][idx], T['attack'][idx]
    ev = (T['key_dh'][idx] <= -1) & T['key_ok'][idx]                              # attack events (unit, key)
    rec = float((p[:, None].expand_as(ev)[ev] <= -0.5).float().mean()) if bool(ev.any()) else float('nan')
    return {'pearson': pearson(p, dh), 'attack_auc': auc(-p, att) if 0 < int(att.sum()) < len(att) else float('nan'),
            'mean_pred_attack': float(p[att].mean()) if bool(att.any()) else float('nan'),
            'mean_pred_no_attack': float(p[~att].mean()), 'recall_at_-0.5': rec}


def analyse(labels):
    res = {}
    for label in labels:
        D = torch.load(OUTD / f'conseq_{label}.pt', weights_only=False)
        T, P = D['units'], D['pred']
        eps = sorted(set(T['seed'].tolist())); eo = {e: i for i, e in enumerate(eps)}
        ep = torch.tensor([eo[int(s)] for s in T['seed']])
        by = [torch.nonzero(ep == i)[:, 0] for i in range(len(eps))]
        g = torch.Generator().manual_seed(20261103)
        draws = lambda: (torch.cat([by[i] for i in torch.randint(len(eps), (len(eps),), generator=g).tolist()]) for _ in range(B))
        out = {'units': len(T['dh']), 'attack_units': int(T['attack'].sum()), 'episodes': len(eps),
               'attack_episodes': len(set(T['seed'][T['attack']].tolist())),
               'attack_events': int(((T['key_dh'] <= -1) & T['key_ok']).sum())}
        for k, p in P.items():
            out[k] = {m: round(v, 4) for m, v in metrics(p, T).items()}
        for a_, b_ in (('exp_hud', 'copy_hud'), ('exp_near', 'copy_near'), ('exp_hud', 'transfer'), ('real_hud', 'exp_hud')):
            vals = []
            for d in draws():
                ma, mb = metrics(P[a_], T, d), metrics(P[b_], T, d)
                vals.append((ma['attack_auc'] - mb['attack_auc'], ma['pearson'] - mb['pearson']))
            v = torch.tensor(vals, dtype=torch.float64)
            q = lambda c: [round(float(v[~torch.isnan(v[:, c]), c].quantile(.025)), 4), round(float(v[~torch.isnan(v[:, c]), c].quantile(.975)), 4)]
            out[f'{a_}_minus_{b_}'] = {'attack_auc_ci95': q(0), 'pearson_ci95': q(1)}
        cn = [metrics(P['copy_near'], T, d)['attack_auc'] for d in draws()]
        cn = torch.tensor(cn, dtype=torch.float64); cn = cn[~torch.isnan(cn)]
        out['copy_near_attack_auc_ci95'] = [round(float(cn.quantile(.025)), 4), round(float(cn.quantile(.975)), 4)]
        res[label] = out
        print(json.dumps({label: out}), flush=True)
    (OUTD / 'conseq_summary.json').write_text(json.dumps(res, indent=1))


def main():
    OUTD.mkdir(exist_ok=True)
    if sys.argv[1] == '--analyse':
        return analyse(sys.argv[2:])
    for arg in sys.argv[1:]:
        label, spec = arg.split('=', 1)
        enc_spec, pred_ck, probe_file = spec.split(',')
        T = collect(label, enc_spec, pred_ck, probe_file)
        P = fit_predict(T)
        torch.save({'label': label, 'encoder': enc_spec, 'predictor': pred_ck, 'units': {k: v for k, v in T.items() if k not in ('exp', 'copy', 'real')},
                    'pred': P}, OUTD / f'conseq_{label}.pt')
        print(json.dumps({label: {k: {m: round(v, 4) for m, v in metrics(p, T).items()} for k, p in P.items()}}), flush=True)


if __name__ == '__main__':
    main()
