"""CPU foreground/localization controls for generated-state readouts.

Same sealed50 TEST episodes, depths9-16. Mean per-frame AUC is computed only
when that native frame contains the mob. Top1 asks whether the highest-score
cell contains it. TopK uses the TRUE count K (privileged diagnostic), so it is
not a deployable count estimator. Local AUC/FP uses cells within Chebyshev1
of a true mob, excluding player31. Paired intervals resample whole episodes.
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
import readout_spatial as BASE
from readout_spatial import masked_auc_draws, ratio, spatial_metrics

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--updates', nargs='+', type=int, default=[0, 500, 1500, 3000]); p.add_argument('--prefix', default='generated_prefix')
    a = p.parse_args(); torch.set_num_threads(8)
    out = G.S.OUT / a.prefix / 'readout_refit'; manifest = out / 'contract.json'
    original_contract = json.loads(manifest.read_text())
    contract = {'format': 'oracle_prefix_readout_spatial_v1', 'source_sha256': G.S.sha(__file__),
                'numerical_helper_sha256': G.S.sha(BASE.__file__),
                'refit_source_sha256': G.S.sha(G.__file__), 'refit_contract_sha256': G.S.sha(manifest),
                'reader_sha256': G.S.sha(G.S.OUT / 'snap_bank_readers.pt'),
                'definitions': __doc__, 'tie_policy_topK': 'stable screen-cell order; no truth tie-break',
                'split': 'sealed refit TEST50 episodes', 'thresholds': 'original unchanged; fixed-weight DEV-F1; new-head own DEV-F1'}
    cid = hashlib.sha256(json.dumps(contract, sort_keys=True).encode()).hexdigest()
    proof = out / 'spatial_contract.json'
    if proof.exists(): assert json.loads(proof.read_text()) == contract, 'spatial diagnostic contract drift'
    else: G.S.atomic(contract, proof, True)
    panel = torch.load(G.S.OUT / 'panel.pt', weights_only=False, map_location='cpu')
    visible = torch.load(G.S.OUT / 'snap_visible.pt', weights_only=False, map_location='cpu')['visible']
    reader = G.S.Readers(torch.load(G.S.OUT / 'snap_bank_readers.pt', weights_only=False, map_location='cpu'))
    labels = G.SB.AllProbes.facts(visible[:, 16:32])['mobs'].float()
    episodes = [r['episode_id'] for r in panel['roots']]
    ids = {k: np.array([episodes.index(e) for e in v]) for k, v in original_contract['split'].items()}
    test_y = labels[ids['TEST'], 8:16].numpy().astype(bool)
    weights = np.random.default_rng(20261012).multinomial(50, np.full(50, 1 / 50), size=original_contract['bootstrap_draws'])
    realhead = torch.load(out / 'real_head.pt', weights_only=False, map_location='cpu')['head']
    for update in a.updates:
        for arm in ('teacher',) if update == 0 else ('teacher', 'prefix'):
            path = G.S.OUT / a.prefix / f'eval_u{update:04d}' / f'pred_{arm}.pt'
            inputs = {'prediction_sha256': G.S.sha(path), 'refit_json_sha256': G.S.sha(out / f'u{update:04d}_{arm}.json')}
            dst = out / f'spatial_u{update:04d}_{arm}.json'
            if dst.exists():
                old = json.loads(dst.read_text()); assert old['contract_id'] == cid and old['inputs'] == inputs
                G.log(stage='spatial_resume', update=update, arm=arm); continue
            G.log(stage='spatial_start', update=update, arm=arm)
            pred = torch.load(path, weights_only=False, map_location='cpu')
            tokens = pred['gen'][:, :, G.CELLS].float(); del pred
            q = tokens[ids['TEST']].reshape(-1, 192)
            original = reader.mob(q).reshape(50, 16, 62, 4).numpy()[:, 8:16]
            dev_scores = reader.mob(tokens[ids['DEV']].reshape(-1, 192))
            calibrated = G.thresholds(dev_scores, labels[ids['DEV']].reshape(-1, 4))['dev_f1']
            cases = {'original': (original, reader.thr.tolist()), 'frozen_dev_f1': (original, calibrated)}
            for head_name in ('new_real', 'new_gen'):
                h = realhead if head_name == 'new_real' else torch.load(out / f'head_u{update:04d}_{arm}_gen.pt', weights_only=False, map_location='cpu')['head']
                cases[head_name] = (G.head_scores(h, q).float().reshape(50, 16, 62, 4).numpy()[:, 8:16], h['thresholds']['dev_f1'])
            # Real-token positive controls on exactly the same TEST frames.
            realq = panel['tokens'][ids['TEST']][:, 16:32, G.CELLS].float().reshape(-1, 192)
            cases['TRUE_original'] = (reader.mob(realq).reshape(50, 16, 62, 4).numpy()[:, 8:16], reader.thr.tolist())
            cases['TRUE_new_real'] = (G.head_scores(realhead, realq).float().reshape(50, 16, 62, 4).numpy()[:, 8:16], realhead['thresholds']['dev_f1'])
            results, samples = {}, {}
            for case, (scores, thresh) in cases.items():
                results[case] = {}
                for mob in ('cow', 'zombie'):
                    j = G.TYPES.index(mob)
                    results[case][mob], samples[case, mob] = spatial_metrics(scores[..., j], test_y[..., j], thresh[j], weights)
            contrasts = {}
            for left, right in [('frozen_dev_f1', 'original'), ('new_real', 'original'), ('new_gen', 'original'), ('new_gen', 'new_real')]:
                contrasts[left + '_minus_' + right] = {}
                for mob in ('cow', 'zombie'):
                    contrasts[left + '_minus_' + right][mob] = {}
                    for met in samples[left, mob]:
                        pa, pb = results[left][mob][met]['estimate'], results[right][mob][met]['estimate']
                        z = samples[left, mob][met] - samples[right, mob][met]
                        contrasts[left + '_minus_' + right][mob][met] = {'difference': pa - pb if pa is not None and pb is not None else None,
                                                                        'ci95': np.quantile(z, [.025, .975]).tolist() if np.isfinite(z).all() else None}
            G.S.atomic({'contract_id': cid, 'inputs': inputs, 'update': update, 'arm': arm,
                        'readings': results, 'contrasts': contrasts}, dst, True)
            G.log(stage='spatial_done', update=update, arm=arm)
    G.log(stage='done', updates=a.updates)


if __name__ == '__main__': main()
