"""Post-hoc on Codex's saved features (20261106_mamba_hostile_path/features.pt), CPU, no model: where does the
fine-only zombie signal shrink? Paired within-background differences remove the background variance an unpaired
reader must fight. PAIRS: 0 empty, 1 zombie both, 2 zombie fine, 3 zombie coarse, 4 cow fine, 5 cow both, 6 cow coarse."""
import json, numpy as np, torch
from sklearn.linear_model import LogisticRegression
H = 'artifacts/experiments/20261106_mamba_hostile_path/'
f = torch.load(H + 'features.pt', weights_only=True)
split = json.load(open(H + 'probe_split.json')); fit, test = np.array(split['fit']), np.array(split['test'])
print({k: tuple(v.shape) for k, v in f.items()})
out = {}
for feat in ('fine', 'embedding', 'belief', 'memory'):
    X = f[feat].flatten(2).double().numpy()                       # [250, 7, D]
    bg = X[:, 0].std(0).mean()                                     # background spread of the empty condition per dim
    r = {}
    for route, (z, c) in {'fine': (2, 4), 'coarse': (3, 6), 'both': (1, 5)}.items():
        dz, dc = X[:, z] - X[:, 0], X[:, c] - X[:, 0]
        nz = np.linalg.norm(dz, axis=1); nb = np.linalg.norm(X[:, 0] - X[:, 0].mean(0), axis=1)
        # consistency: cosine of each held-out zombie difference with the fit-mean zombie difference
        mu = dz[fit].mean(0); cos = (dz[test] @ mu) / (np.linalg.norm(dz[test], axis=1) * np.linalg.norm(mu) + 1e-30)
        # paired readers: zombie-vs-cow and zombie-vs-nothing from difference vectors (fit on 200 backgrounds)
        Xtr = np.concatenate([dz[fit], dc[fit]]); ytr = np.r_[np.ones(len(fit)), np.zeros(len(fit))]
        Xte = np.concatenate([dz[test], dc[test]]); yte = np.r_[np.ones(len(test)), np.zeros(len(test))]
        s = Xtr.std(0).clip(1e-12)
        acc_zc = LogisticRegression(C=1, max_iter=5000).fit(Xtr / s, ytr).score(Xte / s, yte) if nz.max() > 0 else None
        # projection on the fit-mean direction: does the sign separate zombie from cow on held-out backgrounds?
        u = mu / (np.linalg.norm(mu) + 1e-30)
        r[route] = {'diff_norm_median': float(np.median(nz)), 'background_norm_median': float(np.median(nb)),
                    'ratio': float(np.median(nz) / np.median(nb)), 'heldout_cos_with_fit_mean': float(np.median(cos)),
                    'paired_zombie_vs_cow_acc': acc_zc,
                    'proj_zombie_gt_cow_heldout': float(((dz[test] @ u) > (dc[test] @ u)).mean())}
    out[feat] = r
    print(feat, json.dumps(r))
json.dump(out, open('artifacts/experiments/20261106_mamba_hostile_path_review/paired.json', 'w'), indent=1)
