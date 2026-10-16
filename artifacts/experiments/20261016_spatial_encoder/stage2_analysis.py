"""Stage 2 verdict (SPATIAL_ENCODER_PREREG.md): Spatial vs Raw-predictor, paired per seed, from tierb_eval.py's per-unit files.

Primary: "the patch-trained encoder gives a better predictive world state" if Spatial beats Raw-predictor on B1 attack recall and
on B3 (within-root AUC of first-step risk) at BOTH seeds with paired bootstrap intervals (4,000 resamples of roots) excluding 0,
and is not worse on B2 / B4 beyond their noise (reported); "not supported" if, at both seeds, neither B1 recall nor B3 improves
significantly (interval lower bounds <= 0); anything else as is.
Secondary: B1 Pearson (graded consequence), B3 expected safe, B2 calibration, B4 census.
Usage: stage2_analysis.py -> artifacts/eda/spatial_encoder_v1/stage2_verdict.json
"""
import json
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2]))
sys.path.insert(0, str(HERE.parents[1] / '20261015_whiteboard'))
sys.path.insert(0, str(HERE.parents[1] / '20261012_oracle'))
OUT = HERE.parents[2] / 'artifacts/eda/spatial_encoder_v1'
B = 4000


def pearson(x, y):
    x, y = x - x.mean(), y - y.mean()
    return float((x * y).sum() / (x.norm() * y.norm()).clamp_min(1e-12))


def auc_root(risk, P):
    num = den = 0.
    for a in range(5):
        for b in range(5):
            if P[a] > P[b]:
                den += 1; num += float(risk[a] > risk[b]) + .5 * float(risk[a] == risk[b])
    return num / den if den else float('nan')


def boot_clusters(stat, clusters, g):
    """stat(selected cluster ids) -> value; percentile CI over cluster resamples"""
    ids = sorted(set(clusters))
    vals = []
    for _ in range(B):
        pick = [ids[i] for i in torch.randint(len(ids), (len(ids),), generator=g).tolist()]
        vals.append(stat(pick))
    v = torch.tensor(vals)
    return round(float(v.quantile(.025)), 4), round(float(v.quantile(.975)), 4)


def main():
    import dv16 as DV
    roots, _ = DV.load_roots()
    P5 = torch.stack([r['P'] for r in roots])[..., 2][:, :5]
    g = torch.Generator().manual_seed(20261072)
    res = {}
    for s in (1, 2):
        u_sp = torch.load(OUT / f'tierB_units_spatial_s{s}.pt', weights_only=False)
        u_rw = torch.load(OUT / f'tierB_units_rawpred_s{s}.pt', weights_only=False)
        out = {}
        # B1 attack recall (events clustered by root; same events in both worlds)
        assert u_sp['att_root'] == u_rw['att_root']
        ar = u_sp['att_root']
        sp_hit = torch.tensor(u_sp['att_pred']) <= -0.5; rw_hit = torch.tensor(u_rw['att_pred']) <= -0.5
        by = {}
        for i, rid in enumerate(ar):
            by.setdefault(rid, []).append(i)

        def rec_diff(pick):
            idx = [i for rid in pick for i in by[rid]]
            return float(sp_hit[idx].float().mean() - rw_hit[idx].float().mean())
        out['B1_attack_recall'] = {'spatial': round(float(sp_hit.float().mean()), 4), 'rawpred': round(float(rw_hit.float().mean()), 4),
                                   'diff': round(rec_diff(list(by)), 4), 'ci95': boot_clusters(rec_diff, list(by), g), 'events': len(ar)}
        # B1 Pearson (graded), clustered by root
        assert u_sp['b1_root'] == u_rw['b1_root']
        br = u_sp['b1_root']
        sp_p, rw_p, real = torch.tensor(u_sp['b1_pred']), torch.tensor(u_rw['b1_pred']), torch.tensor(u_sp['b1_real'])
        by1 = {}
        for i, rid in enumerate(br):
            by1.setdefault(rid, []).append(i)

        def r_diff(pick):
            idx = torch.tensor([i for rid in pick for i in by1[rid]])
            return pearson(sp_p[idx], real[idx]) - pearson(rw_p[idx], real[idx])
        out['B1_pearson'] = {'spatial': round(pearson(sp_p, real), 4), 'rawpred': round(pearson(rw_p, real), 4),
                             'diff': round(r_diff(list(by1)), 4), 'ci95': boot_clusters(r_diff, list(by1), g)}
        # B3 within-root AUC and expected safe (DV16 roots)
        a_sp = torch.tensor([auc_root(u_sp['b3_risk'][i], P5[i]) for i in range(len(P5))])
        a_rw = torch.tensor([auc_root(u_rw['b3_risk'][i], P5[i]) for i in range(len(P5))])
        ok = ~torch.isnan(a_sp)
        d = (a_sp - a_rw)[ok]
        idx = torch.randint(len(d), (B, len(d)), generator=g)
        out['B3_auc5'] = {'spatial': round(float(a_sp[ok].mean()), 4), 'rawpred': round(float(a_rw[ok].mean()), 4),
                          'diff': round(float(d.mean()), 4), 'ci95': [round(float(d[idx].mean(1).quantile(q)), 4) for q in (.025, .975)],
                          'roots': int(ok.sum())}
        de = u_sp['b3_es'] - u_rw['b3_es']
        idx = torch.randint(len(de), (B, len(de)), generator=g)
        out['B3_expected_safe5'] = {'diff': round(float(de.mean()), 4), 'ci95': [round(float(de[idx].mean(1).quantile(q)), 4) for q in (.025, .975)]}
        for k in ('B2', 'B4'):
            out[k] = {'spatial': json.loads((OUT / f'tierB_spatial_s{s}.json').read_text())[k],
                      'rawpred': json.loads((OUT / f'tierB_rawpred_s{s}.json').read_text())[k]}
        res[f'seed{s}'] = out
    wins = all(res[f'seed{s}']['B1_attack_recall']['ci95'][0] > 0 and res[f'seed{s}']['B3_auc5']['ci95'][0] > 0 for s in (1, 2))
    no_better = all(res[f'seed{s}']['B1_attack_recall']['ci95'][0] <= 0 and res[f'seed{s}']['B3_auc5']['ci95'][0] <= 0 for s in (1, 2))
    res['verdict_primary'] = ('patch-trained encoder gives a better predictive world state (B2/B4 to be checked)' if wins
                              else 'not supported' if no_better else 'mixed: reported as is')
    print(json.dumps(res, indent=1))
    (OUT / 'stage2_verdict.json').write_text(json.dumps(res, indent=1))


if __name__ == '__main__':
    main()
