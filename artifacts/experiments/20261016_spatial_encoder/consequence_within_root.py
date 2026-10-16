"""Within-root ACTION discrimination of the consequence predictions (user + Codex audit 10-17: pooled attack AUC can identify dangerous
roots with zero action ranking; ../20261017_frozen_state_audit/ASSESSMENT.md). Read-only, from consequence_check.py's saved
per-unit cross-fitted predictions.

Population: (root, action) units with >= 2 alive keys (as tierb_eval_v2 B1 and Codex's label control).
Primary: within-root attack AUC over actions - roots having both attacking and non-attacking actions; every (attacking action,
non-attacking action) pair of the same root; score = -predicted dh; ties count 1/2. Pair-weighted (all pairs) and root-weighted
(mean of per-root AUCs).
Secondary: within-root Pearson of predicted vs real mean dh across a root's actions (roots with any spread in real dh), root-
weighted; pooled attack AUC on the same units (for contrast); the label-only control (exact root adjacent-zombie count, constant
over actions -> 0.5 within roots by construction).
Intervals: episode-cluster bootstrap (4,000) over the episodes of the comparable roots; paired differences between states.
Usage: consequence_within_root.py <label> ... -> artifacts/eda/spatial_encoder_v1/erasure/conseq_within_root.json
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import spatial_eval as EV                                                       # noqa: E402
from consequence_check import auc                                               # noqa: E402  (tie-aware, vectorized)

import torch                                                                    # noqa: E402

D = EV.OUT / 'erasure'
B = 4000
STATES = ('exp_hud', 'exp_near', 'copy_hud', 'copy_near', 'real_hud', 'transfer')


def per_root(T, P, keep):
    """per comparable root: (episode, n_pos*n_neg, correct pairs (ties 1/2) per state, pearson per state or nan)"""
    roots = sorted(set(T['root'][keep].tolist()))
    out = []
    for r in roots:
        m = keep & (T['root'] == r)
        att, dh = T['attack'][m], T['dh'][m]
        npos, nneg = int(att.sum()), int((~att).sum())
        row = {'ep': int(T['seed'][m][0]), 'pairs': npos * nneg, 'correct': {}, 'pearson': {}}
        for st in STATES:
            p = -P[st][m]
            if npos and nneg:
                d = p[att][:, None] - p[~att][None, :]
                row['correct'][st] = float((d > 0).sum() + 0.5 * (d == 0).sum())
            x, y = P[st][m], dh
            row['pearson'][st] = float(((x - x.mean()) * (y - y.mean())).sum() / ((x - x.mean()).norm() * (y - y.mean()).norm()).clamp_min(1e-12)) \
                if len(x) > 2 and float(y.std()) > 0 and float(x.std()) > 0 else float('nan')
        out.append(row)
    return out


def summarise(rows, idx=None):
    rows = rows if idx is None else [rows[i] for i in idx]
    comp = [r for r in rows if r['pairs'] > 0]
    s = {}
    for st in STATES:
        pw = sum(r['correct'][st] for r in comp) / max(sum(r['pairs'] for r in comp), 1)
        rw = sum(r['correct'][st] / r['pairs'] for r in comp) / max(len(comp), 1)
        pr = torch.tensor([r['pearson'][st] for r in rows]); pr = pr[~torch.isnan(pr)]
        s[st] = {'within_auc_pairs': pw, 'within_auc_roots': rw, 'within_pearson_roots': float(pr.mean()) if len(pr) else float('nan')}
    return s


def main():
    res = {}
    for label in sys.argv[1:]:
        Dd = torch.load(D / f'conseq_{label}.pt', weights_only=False)
        T, P = Dd['units'], Dd['pred']
        keep = T['keys_alive'] >= 2
        rows = per_root(T, P, keep)
        comp = [r for r in rows if r['pairs'] > 0]
        eps = sorted(set(r['ep'] for r in rows)); eo = {e: i for i, e in enumerate(eps)}
        by = {}
        for i, r in enumerate(rows):
            by.setdefault(eo[r['ep']], []).append(i)
        out = {'units': int(keep.sum()), 'roots': len(rows), 'comparable_roots': len(comp),
               'comparable_episodes': len(set(r['ep'] for r in comp)), 'comparable_pairs': sum(r['pairs'] for r in comp),
               'pooled_attack_auc': {st: round(auc(-P[st][keep], T['attack'][keep]), 4) for st in STATES}}
        full = summarise(rows)
        out['within_root'] = {st: {k: round(v, 4) for k, v in d.items()} for st, d in full.items()}
        g = torch.Generator().manual_seed(20261105)
        boot = []
        for _ in range(B):
            draw = torch.randint(len(eps), (len(eps),), generator=g).tolist()
            boot.append(summarise(rows, [i for e in draw for i in by.get(e, [])]))
        q = lambda v: [round(float(torch.tensor(v).quantile(.025)), 4), round(float(torch.tensor(v).quantile(.975)), 4)]
        out['ci95_within_auc_pairs'] = {st: q([b[st]['within_auc_pairs'] for b in boot]) for st in STATES}
        for a_, b_ in (('exp_hud', 'copy_hud'), ('exp_near', 'copy_near'), ('exp_hud', 'real_hud'), ('exp_near', 'exp_hud')):
            out[f'{a_}_minus_{b_}'] = {'within_auc_pairs_ci95': q([b[a_]['within_auc_pairs'] - b[b_]['within_auc_pairs'] for b in boot]),
                                       'within_pearson_roots_ci95': q([b[a_]['within_pearson_roots'] - b[b_]['within_pearson_roots'] for b in boot
                                                                      if b[a_]['within_pearson_roots'] == b[a_]['within_pearson_roots']])}
        res[label] = out
        print(json.dumps({label: {k: out[k] for k in ('comparable_roots', 'comparable_episodes', 'pooled_attack_auc', 'within_root', 'ci95_within_auc_pairs')}}), flush=True)
    (D / 'conseq_within_root.json').write_text(json.dumps(res, indent=1))


if __name__ == '__main__':
    main()
