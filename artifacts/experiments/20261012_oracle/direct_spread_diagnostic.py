"""Frozen-output CPU spread intervention; never training or physical repair.

Shrink only scored semantically stable cells around each model's own sample
mean. Exact simulator-derived masks are evaluation only. Scientific criteria,
models, samples, readers, and targets remain frozen. Atomic per-root records
make the analysis resumable under an exact source/input contract.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np
import torch

import direct_eval as E
import direct_model as M
import loss_intervention as L

OUT = L.OUT / 'direct_generator/diagnostics/stable_spread'
FACTORS = (0., .25, .5, 1.)


def distance_matrix(x, y):
    """Float64 implementation of the declared first-power global+patch metric."""
    def norm(cells):
        a, b = x[:, cells].flatten(1), y[:, cells].flatten(1)
        return torch.cdist(a, b, compute_mode='use_mm_for_euclid_dist') / math.sqrt(a.shape[-1])
    return .5 * norm(M.INTERIOR) + .5 * torch.stack([norm(p) for p in M.PATCHES]).mean(0)


def score(x, y):
    cross = distance_matrix(x, y)
    pair = distance_matrix(x, x)
    pair.fill_diagonal_(0.)
    attraction = float(cross.mean())
    spread = float(pair.sum() / (len(x) * (len(x) - 1)))
    return {'attraction': attraction, 'pair_distance': spread,
            'half_pair_distance': .5 * spread, 'energy': attraction - .5 * spread}


def mean_bias(x, y, mask):
    delta = x.mean(0) - y.mean(0)
    return {'declared_distance': float(distance_matrix(x.mean(0, keepdim=True), y.mean(0, keepdim=True))[0, 0]),
            'scored_mse': float(delta[list(M.INTERIOR)].square().mean()),
            'stable_mse': float(delta[mask].square().mean()) if mask.any() else None}


def norms(x, mask):
    z = x[:, mask]
    if not len(z[0]):
        return None
    rms = z.square().mean(-1).sqrt()
    return {'mean_token_rms': float(rms.mean()), 'minimum_token_rms': float(rms.min()),
            'maximum_token_rms': float(rms.max()), 'mean_abs_rms_minus_one': float((rms - 1).abs().mean()),
            'mean_abs_token_coordinate_mean': float(z.mean(-1).abs().mean())}


def interval(values, *, seed=20261014):
    values = np.asarray(values, dtype=np.float64)
    assert values.ndim == 1 and np.isfinite(values).all()
    rng = np.random.default_rng(seed)
    boot = values[rng.integers(0, len(values), size=(2000, len(values)))].mean(1)
    return {'mean': float(values.mean()), 'ci95': np.quantile(boot, [.025, .975]).tolist(),
            'roots': len(values), 'episodes': len(values)}


def selftest():
    g = torch.Generator().manual_seed(11)
    x, y = torch.randn(6, 81, 192, generator=g).double(), torch.randn(7, 81, 192, generator=g).double()
    pair = distance_matrix(x, y)
    wanted = M.distance(x[:, None].float(), y[None].float()).double()
    assert float((pair - wanted).abs().max()) < 3e-7
    # Analytically known independent-cell counterexample, no model fitting.
    p = torch.tensor([[1., 0.], [0., 1.]], dtype=torch.float64)
    independent = torch.tensor([[0., 0.], [1., 0.], [0., 1.], [1., 1.]], dtype=torch.float64)
    def exact(q):
        return float(torch.cdist(q, p).mean() - .5 * torch.cdist(q, q).mean())
    assert math.isclose(exact(p), math.sqrt(2) / 4)
    assert exact(p) < exact(independent) < exact(torch.tensor([[.5, .5]], dtype=torch.float64))
    return {'declared_metric_float64_vs_training_maxabs': float((pair - wanted).abs().max()),
            'exact_energy_counterexample_passed': True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--threads', type=int, default=8)
    parser.add_argument('--stop-after', type=int)
    a = parser.parse_args()
    torch.set_num_threads(a.threads)
    validation = selftest()
    directory = L.OUT / 'direct_generator/eval_exploratory_u3000'
    analysis_path = directory / 'analysis.json'
    analysis = json.loads(analysis_path.read_text())
    data, inputs = E.cohort('exploratory')
    saved = {arm: torch.load(directory / f'pred_{arm}.pt', map_location='cpu', weights_only=False, mmap=True)
             for arm in ('point', 'energy')}
    assert all(x['roots'] == data['roots'] for x in saved.values())
    assert all(x['update'] == 3000 for x in saved.values())
    # Each root is a distinct episode, matching the declared episode bootstrap.
    assert len({json.dumps(r, sort_keys=True) for r in data['roots']}) == len(data['roots'])
    mask = torch.tensor(analysis['arms']['energy']['sampled_variation']['masks']['scored_semantic_stable'])
    assert mask.shape == (len(data['roots']), 81) and mask.dtype == torch.bool
    assert torch.equal(mask, torch.tensor(analysis['arms']['point']['sampled_variation']['masks']['scored_semantic_stable']))
    contract = {'format': 'direct_stable_spread_v1', 'factors': list(FACTORS), 'roots': data['roots'],
                'prediction_hashes': {arm: L.sha(directory / f'pred_{arm}.pt') for arm in saved},
                'analysis_sha256': L.sha(analysis_path), 'inputs': inputs,
                'sources': {str(Path(p).relative_to(L.ROOT)): L.sha(p)
                            for p in (__file__, E.__file__, M.__file__)},
                'metric': 'Fixed first-power .5 global35 + .5 mean9 overlapping3x3 patch distances; U-statistic spread',
                'intervention': 'Xalpha = mean_model + alpha*(X-mean_model) on scored stable cells only; no renormalization',
                'limitations': ['Exact masks are evaluation-only oracle information',
                                'Finite-fan history/full-state conditioning differs',
                                'Shrunk outputs may leave LayerNorm norm; no physical-repair claim',
                                'This post-hoc diagnostic does not alter preregistered verdicts']}
    cid = hashlib.sha256(json.dumps(contract, sort_keys=True).encode()).hexdigest()
    cp = OUT / 'contract.json'
    if cp.exists():
        assert json.loads(cp.read_text()) == contract, 'diagnostic source/input drift'
    else:
        L.atomic(contract, cp, True)
    rows = []
    for index, root in enumerate(data['roots']):
        path = OUT / 'roots' / f'{index:04d}.json'
        if path.exists():
            row = json.loads(path.read_text())
            assert row['contract_id'] == cid and row['root'] == root
        else:
            start = time.perf_counter()
            y = data['fan_tokens'][index].double()
            st = mask[index]
            row = {'contract_id': cid, 'index': index, 'root': root, 'stable_cells': int(st.sum()),
                   'true_stable_variance_sum': float(y.var(0, unbiased=False).mean(-1)[st].sum()),
                   'real_forward_split': score(y[:16], y[16:]), 'real_reverse_split': score(y[16:], y[:16]),
                   'real_norms': norms(y, st), 'arms': {}}
            for arm in saved:
                x = saved[arm]['predictions'][index].double()
                row['arms'][arm] = {'score': score(x, y), 'mean_bias': mean_bias(x, y, st),
                                     'stable_variance_sum': float(x.var(0, unbiased=False).mean(-1)[st].sum()),
                                     'norms': norms(x, st)}
                if arm == 'energy':
                    mean = x.mean(0, keepdim=True)
                    interventions = {}
                    for factor in FACTORS:
                        altered = x.clone()
                        altered[:, st] = mean[:, st] + factor * (x[:, st] - mean[:, st])
                        assert torch.equal(altered[:, ~st], x[:, ~st])
                        assert float((altered.mean(0) - x.mean(0)).abs().max()) < 2e-15
                        interventions[str(factor)] = {'score': score(altered, y), 'norms': norms(altered, st),
                                                       'mean_preservation_maxabs': float((altered.mean(0) - mean[0]).abs().max()),
                                                       'unmasked_preservation_exact': True}
                    row['arms'][arm]['interventions'] = interventions
            row['seconds'] = time.perf_counter() - start
            L.atomic(row, path, True)
        rows.append(row)
        print(json.dumps({'stage': 'stable_spread', 'completed_roots': index + 1, 'total': len(data['roots'])}), flush=True)
        if a.stop_after and index + 1 >= a.stop_after:
            return
    fields = ('energy', 'attraction', 'half_pair_distance')
    summaries = {}
    for arm in ('point', 'energy'):
        summaries[arm] = {field: interval([r['arms'][arm]['score'][field] for r in rows]) for field in fields}
        summaries[arm]['mean_bias'] = {field: interval([r['arms'][arm]['mean_bias'][field] for r in rows if r['arms'][arm]['mean_bias'][field] is not None])
                                        for field in ('declared_distance', 'scored_mse', 'stable_mse')}
        summaries[arm]['stable_variance_sum'] = interval([r['arms'][arm]['stable_variance_sum'] for r in rows])
    reference = {field: interval([.5 * (r['real_forward_split'][field] + r['real_reverse_split'][field]) for r in rows])
                 for field in fields}
    interventions = {}
    for factor in FACTORS:
        key = str(factor)
        interventions[key] = {field: interval([r['arms']['energy']['interventions'][key]['score'][field] for r in rows]) for field in fields}
        interventions[key]['energy_minus_unmodified'] = interval([r['arms']['energy']['interventions'][key]['score']['energy'] - r['arms']['energy']['score']['energy'] for r in rows])
        interventions[key]['roots_improved'] = sum(r['arms']['energy']['interventions'][key]['score']['energy'] < r['arms']['energy']['score']['energy'] for r in rows)
        interventions[key]['norms'] = {field: interval([r['arms']['energy']['interventions'][key]['norms'][field] for r in rows if r['arms']['energy']['interventions'][key]['norms']])
                                       for field in ('mean_token_rms', 'mean_abs_rms_minus_one', 'mean_abs_token_coordinate_mean')}
    result = {'format': 'direct_stable_spread_v1', 'completed': True, 'contract_id': cid,
              'validation': validation, 'summaries': summaries, 'real_split_reference': reference,
              'true_stable_variance_sum': interval([r['true_stable_variance_sum'] for r in rows]),
              'interventions': interventions, 'roots': len(rows), 'physical_repair_claim': False,
              'scope': contract['limitations'], 'total_root_compute_seconds': sum(r['seconds'] for r in rows)}
    L.atomic(result, OUT / 'result.json', True)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
