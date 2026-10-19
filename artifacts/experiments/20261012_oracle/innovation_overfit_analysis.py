"""CPU-only analysis of the completed, deliberately overfit innovation control.

No training or new model inference. The saved forecasts are compared with the
exact training cases and a previously fitted real-state mob reader. This does
not turn the four FIT roots into a generalization or sampled-distribution test.
"""
import hashlib
import json
from pathlib import Path

import torch

import generated_readout_refit as R
import loss_intervention as L


def main():
    torch.set_num_threads(4)
    directory = L.OUT / 'learned_innovation/optimization_control'
    contract = json.loads((directory / 'contract.json').read_text())
    result = json.loads((directory / 'result.json').read_text())
    cid = hashlib.sha256(json.dumps(contract, sort_keys=True).encode()).hexdigest()
    assert result['contract_id'] == cid and result['history'][-1]['update'] == 2000
    assert all(L.sha(Path(path)) == digest for path, digest in contract['sources'].items())
    panel = torch.load(L.OUT / 'panel.pt', map_location='cpu', weights_only=False)
    fan_path = L.OUT / 'learned_innovation/real_futures/fan.pt'
    assert L.sha(fan_path) == contract['fan_sha256']
    fan = torch.load(fan_path, map_location='cpu', weights_only=False)
    targets, labels, roots = [], [], []
    for j, root in enumerate(contract['selected_FIT_roots']):
        i = panel['roots'].index(root)
        unique = fan['tokens'][i].flatten(1).unique(dim=0).view(-1, 81, 192)
        for target in unique:
            match = (fan['tokens'][i] == target).flatten(1).all(-1).nonzero()[0, 0]
            targets.append(target)
            labels.append(fan['layouts'][i, match, R.CELLS][:, [0, 2, 1, 3]])
            roots.append(j)
    targets = torch.stack(targets).float()
    labels = torch.stack(labels).bool()
    roots = torch.tensor(roots)
    assert L.sha(L.OUT / 'panel.pt') == contract['panel_sha256']
    assert hashlib.sha256(targets.numpy().tobytes()).hexdigest() == contract['targets_sha256']
    head_path = L.OUT / 'learned_innovation/eval_u3000_h01/mob_heads.pt'
    head = torch.load(head_path, map_location='cpu', weights_only=False)['heads']['real']
    threshold = torch.tensor(head['thresholds']['dev_f1'])

    def mob_reading(x):
        decoded = R.head_scores(head, x[:, R.CELLS]) > threshold
        return {kind: {'truth': int(labels[..., k].sum()),
                       'tp': int((decoded[..., k] & labels[..., k]).sum()),
                       'fp': int((decoded[..., k] & ~labels[..., k]).sum()),
                       'fn': int((~decoded[..., k] & labels[..., k]).sum())}
                for k, kind in enumerate(R.TYPES)}

    means = torch.stack([targets[roots == j].double().mean(0) for j in range(4)])
    floor = float((targets.double() - means[roots]).square().mean())
    rows = {}
    for arm in contract['arms']:
        path = directory / f'{arm}_u2000.pt'
        saved = torch.load(path, map_location='cpu', weights_only=False)
        assert saved['contract_id'] == cid and saved['update'] == 2000
        pred = saved['outputs'].double()
        pred_means = torch.stack([pred[roots == j].mean(0) for j in range(4)])
        nearest, codes_per_root = 0, []
        for j in range(4):
            idx = (roots == j).nonzero().flatten()
            distances = torch.cdist(pred[idx].flatten(1), targets[idx].double().flatten(1))
            nearest += int((distances.argmin(-1) == torch.arange(len(idx))).sum())
            if saved['codes'] is not None:
                codes_per_root.append(int(saved['codes'][idx].unique(dim=0).shape[0]))
        rows[arm] = {'snapshot_sha256': L.sha(path),
                     'mse': float((pred - targets).square().mean()),
                     'mse_over_exact_fp64_floor': float((pred - targets).square().mean()) / floor,
                     'within_root_prediction_variance_over_target_variance':
                         float((pred - pred_means[roots]).square().mean()) / floor,
                     'own_case_is_nearest_real_future': nearest,
                     'joint_argmax_codes_per_root': codes_per_root or None,
                     'real_fitted_mob_reader': mob_reading(pred.float())}
    report = {'training_contract_id': cid,
              'source_sha256': L.sha(Path(__file__)), 'real_reader_sha256': L.sha(head_path),
              'cases_per_root': [(roots == j).sum().item() for j in range(4)],
              'exact_fp64_conditional_mean_mse_floor': floor,
              'models': rows, 'reader_on_real_targets': mob_reading(targets),
              'scope': 'Saved training-case forecasts only: four FIT roots, 26 equally weighted distinct futures. '
                       'Supplied IDs are not learned innovations; no generalization, probability calibration or actor claim. '
                       'Argmax code counts do not bound information carried by soft probabilities. '
                       'Negligible no-code within-root variance can include saved BF16 batch-shape effects.'}
    L.atomic(report, directory / 'completed_analysis.json', True)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
