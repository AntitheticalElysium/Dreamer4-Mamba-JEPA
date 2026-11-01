"""CPU-only checks of PLAN.md's mathematical assertions; no model forward or changes.

The proposed token map is evaluated on saved data solely to test its claimed
identity property. No mapped token is fed to a sampler, belief or outcome head.
"""
import hashlib
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]
OLD = ROOT / 'artifacts/experiments/20261028_saccade_p1_dissection'
OBS = ROOT / 'artifacts/experiments/20261028_saccade_observational_diagnosis'


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(4 << 20), b''):
            h.update(block)
    return h.hexdigest()


def proposed_map(y):
    z = 3 * torch.atanh(y / 3)
    return 3 * torch.tanh(F.layer_norm(z, z.shape[-1:]) / 3)


if __name__ == '__main__':
    paths = [OUT / 'PLAN.md', ROOT / 'saccade/model.py', ROOT / 'saccade/m0.yaml']
    opening = {str(p.relative_to(ROOT)): sha(p) for p in paths}
    b = torch.load(OLD / 'baseline_samples.pt', weights_only=False, mmap=True)
    results = {}
    for k in ('p', 'g', 'v'):
        y = b['real'][k][:1024].float()
        q = proposed_map(y)
        qq = proposed_map(q)
        z = 3 * torch.atanh(y / 3)
        rm = (q - y).square().mean(-1).sqrt().flatten()
        results[k] = {
            'n_tokens': rm.numel(),
            'real_map_coordinate_rmse': float((q - y).square().mean().sqrt()),
            'real_map_max_abs_change': float((q - y).abs().max()),
            'real_map_token_rmse_quantiles': torch.quantile(rm, torch.tensor([.5, .9, .99])).tolist(),
            'repeat_map_max_abs_change': float((qq - q).abs().max()),
            'inverse_variance_median': float(z.var(-1, unbiased=False).median()),
            'generated_at_or_outside_inverse_domain': int((b['generated'][k].abs() >= 3).sum()),
        }
    d = json.loads((OBS / 'sampler_observation.json').read_text())
    dim, sig, rms, mu = d['target_dimension'], 5., d['real_rms'], d['real_mean_norm']
    results['noise'] = {
        'dimension': dim,
        'per_coordinate_energy_snr_actual': rms * rms / sig**2,
        'total_signal_to_total_noise_energy_snr_actual': (dim * rms * rms) / (dim * sig**2),
        'unit_variance_energy_snr': 1 / sig**2,
        'dimension_over_sigma_squared': dim / sig**2,
        'mean_direction_signal_to_noise_amplitude_at_sigma_5': mu / sig,
        'start_mean_shift_gaussian_kl_term_at_sigma_5': mu * mu / (2 * sig * sig),
        'observed_first_clean_mean_projection': d['stages']['0']['clean_projection']['mean'],
        'observed_end_mean_projection': d['stages']['2']['clean_projection']['mean'],
        'target_mean_projection': d['real_projection_mean'],
    }
    results['training_sigma_distribution_tail'] = {
        str(s): .5 * math.erfc((math.log(s) + .4) / (1.2 * math.sqrt(2))) for s in (5, 10, 20)
    }
    results['churn_first_evaluation_at_sigma_max_20'] = {
        str(n): 20 * (1 + min(1 / n, math.sqrt(2) - 1)) for n in (3, 32, 64)
    }
    results['heun_network_evaluations'] = {str(n): 2 * n - 1 for n in (8, 16, 32, 64)}
    assert opening == {str(p.relative_to(ROOT)): sha(p) for p in paths}
    results['plan_and_source_sha256_unchanged'] = opening
    (OUT / 'review_checks.json').write_text(json.dumps(results, indent=2) + '\n')
    print(json.dumps(results, indent=2))
