"""CPU paired teacher-versus-prefix generated-state readout comparisons.

Uses the fixed 100/50/50 episode split and already-fitted equal-capacity heads.
Both DEV-F1 and DEV-count thresholds remain separate readings; no TEST tuning.
The original frozen head is a transfer control, not a semantic-information ceiling.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(Path(__file__).parent))
import prefix_readout_refit as G


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prefix', default='generated_prefix')
    parser.add_argument('--updates', nargs='+', type=int, default=[500, 1500, 3000])
    args = parser.parse_args()
    torch.set_num_threads(8)
    out = G.S.OUT / args.prefix / 'readout_refit'
    manifest_path = out / 'contract.json'
    manifest = json.loads(manifest_path.read_text())
    contract = {'format': 'oracle_prefix_readout_compare_v1',
                'source_sha256': G.S.sha(__file__), 'refit_source_sha256': G.S.sha(G.__file__),
                'refit_contract_sha256': G.S.sha(manifest_path),
                'arms': ['teacher', 'prefix'], 'population': 'all TEST50 episodes, depths9-16',
                'bootstrap': 'same paired episode weights as refit, no TEST selection',
                'thresholds': 'original; per-head DEV-F1 and DEV-count separately'}
    cid = hashlib.sha256(json.dumps(contract, sort_keys=True).encode()).hexdigest()
    proof = out / 'paired_contract.json'
    if proof.exists():
        assert json.loads(proof.read_text()) == contract, 'paired readout contract drift'
    else:
        G.S.atomic(contract, proof, True)
    panel = torch.load(G.S.OUT / 'panel.pt', weights_only=False, map_location='cpu')
    visible = torch.load(G.S.OUT / 'snap_visible.pt', weights_only=False, map_location='cpu')
    episodes = [row['episode_id'] for row in panel['roots']]
    test = np.array([episodes.index(e) for e in manifest['split']['TEST']])
    assert len(test) == 50 and len(set(episodes)) == 200
    labels = G.SB.AllProbes.facts(visible['visible'][:, 16:32])['mobs'].float()[test]
    del visible
    reader = G.S.Readers(torch.load(G.S.OUT / 'snap_bank_readers.pt', weights_only=False, map_location='cpu'))
    real_head = torch.load(out / 'real_head.pt', weights_only=False, map_location='cpu')['head']
    weights = np.random.default_rng(20261012).multinomial(50, np.full(50, 1 / 50), size=manifest['bootstrap_draws'])
    for update in args.updates:
        inputs = {arm: {'prediction': G.S.sha(G.S.OUT / args.prefix / f'eval_u{update:04d}' / f'pred_{arm}.pt'),
                        'head_fit_result': G.S.sha(out / f'u{update:04d}_{arm}.json')}
                  for arm in ('teacher', 'prefix')}
        destination = out / f'paired_u{update:04d}.json'
        if destination.exists():
            previous = json.loads(destination.read_text())
            assert previous['contract_id'] == cid and previous['inputs'] == inputs
            G.log(stage='paired_readout_resume', update=update)
            continue
        results, draws = {}, {}
        for arm in ('teacher', 'prefix'):
            predictions = torch.load(G.S.OUT / args.prefix / f'eval_u{update:04d}' / f'pred_{arm}.pt',
                                     weights_only=False, map_location='cpu')
            tokens = predictions['gen'][test][:, :, G.CELLS].float()
            del predictions
            heads = {'new_real': real_head,
                     'new_gen': torch.load(out / f'head_u{update:04d}_{arm}_gen.pt',
                                           weights_only=False, map_location='cpu')['head']}
            for head_name in ('original', 'new_real', 'new_gen'):
                scores = (reader.mob(tokens.reshape(-1, 192)) if head_name == 'original'
                          else G.head_scores(heads[head_name], tokens.reshape(-1, 192)).float())
                scores = scores.reshape(50, 16, 62, 4)
                options = {'original': reader.thr.tolist()} if head_name == 'original' else heads[head_name]['thresholds']
                for threshold_name, threshold in options.items():
                    key = f'{arm}/{head_name}/{threshold_name}'
                    results[key], draws[key] = G.metrics(scores, labels, threshold, weights)
                    for j, mob in enumerate(G.TYPES):
                        true = labels[:, 8:16, :, j].numpy().astype(bool)
                        detected = scores[:, 8:16, :, j].numpy() > threshold[j]
                        tp = (true & detected).sum((1, 2))
                        denominator = true.sum((1, 2)) + detected.sum((1, 2))
                        sampled_denominator = weights @ denominator
                        z = np.full(len(weights), np.nan)
                        np.divide(2 * (weights @ tp), sampled_denominator, out=z, where=sampled_denominator > 0)
                        draws[key][mob, 'f1'] = z
                        results[key][mob]['f1_interval'] = {
                            'estimate': results[key][mob]['f1'],
                            'ci95': np.quantile(z, [.025, .975]).tolist() if np.isfinite(z).all() else None,
                            'undefined_draws': int((~np.isfinite(z)).sum())}
            del tokens
        contrasts = {}
        for head_name, options in [('original', ['original']), ('new_real', ['dev_f1', 'dev_count']),
                                   ('new_gen', ['dev_f1', 'dev_count'])]:
            for option in options:
                left, right = f'prefix/{head_name}/{option}', f'teacher/{head_name}/{option}'
                contrast = {}
                for mob in G.TYPES:
                    contrast[mob] = {}
                    for metric in ('f1', 'recall', 'precision', 'fp_per_true', 'census'):
                        field = 'f1_interval' if metric == 'f1' else metric
                        a, b = results[left][mob][field]['estimate'], results[right][mob][field]['estimate']
                        samples = draws[left][mob, metric] - draws[right][mob, metric]
                        contrast[mob][metric] = {
                            'difference': a - b if a is not None and b is not None else None,
                            'ci95': np.quantile(samples, [.025, .975]).tolist() if np.isfinite(samples).all() else None,
                            'undefined_draws': int((~np.isfinite(samples)).sum())}
                contrasts[left + '_minus_' + right] = contrast
        G.S.atomic({'contract_id': cid, 'inputs': inputs, 'update': update,
                    'TEST_episodes': manifest['split']['TEST'], 'readings': results,
                    'contrasts': contrasts}, destination, True)
        G.log(stage='paired_readout_done', update=update)


if __name__ == '__main__':
    main()
