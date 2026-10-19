"""CPU supplement: frozen-weight threshold recalibration and paired AUC.

Uses the already sealed generated_readout_refit episode split and head files.
The original reader coefficients remain fixed. DEV-only F1/count thresholds
are separate sensitivity readings; originals are never overwritten. Paired
episode-bootstrap AUC uses score ties exactly, with no TEST selection.
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
import generated_readout_refit as G


def auc_draws(scores, labels, weights):
    """Weighted empirical AUC; each bootstrap gives all cells of an episode
    the episode's multinomial weight. Exact equal-score ties count 1/2.
    """
    s, y = np.asarray(scores), np.asarray(labels).astype(bool)
    assert s.shape == y.shape and len(s) == weights.shape[1]
    episode = np.repeat(np.arange(len(s)), np.prod(s.shape[1:]))
    sf, yf = s.flatten(), y.flatten()
    order = np.argsort(sf, kind='stable')
    so, yo, eo = sf[order], yf[order], episode[order]
    ends = np.flatnonzero(np.r_[so[1:] != so[:-1], True])
    starts = np.r_[0, ends[:-1] + 1]
    result = []
    for start in range(0, len(weights), 100):
        w = weights[start:start + 100][:, eo].astype(np.float64)
        p = np.add.reduceat(w * yo, starts, axis=1)
        n = np.add.reduceat(w * ~yo, starts, axis=1)
        numerator = (p * (np.cumsum(n, axis=1) - .5 * n)).sum(1)
        denominator = p.sum(1) * n.sum(1)
        z = np.full(len(w), np.nan); np.divide(numerator, denominator, out=z, where=denominator > 0)
        result.append(z)
    return np.concatenate(result)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--updates', nargs='+', type=int, default=[0, 1500, 3000])
    p.add_argument('--prefix', default='loss_intervention')
    a = p.parse_args()
    torch.set_num_threads(8)
    out = G.S.OUT / a.prefix / 'readout_refit'
    manifest = out / 'contract.json'; original_contract = json.loads(manifest.read_text())
    contract = {'format': 'oracle_readout_calibration_v1', 'source_sha256': G.S.sha(__file__),
                'refit_contract_sha256': G.S.sha(manifest), 'refit_source_sha256': G.S.sha(G.__file__),
                'frozen_reader_sha256': G.S.sha(G.S.OUT / 'snap_bank_readers.pt'),
                'threshold_fit': 'fixed original reader weights; thresholds selected on DEV-only all16depths by F1 and count',
                'auc_bootstrap': 'paired whole TEST episodes, original refit split seed/draws; exact tie handling',
                'scope': 'post-hoc reused DEV worlds, equal test episodes and head capacity'}
    cid = hashlib.sha256(json.dumps(contract, sort_keys=True).encode()).hexdigest()
    proof = out / 'calibration_contract.json'
    if proof.exists(): assert json.loads(proof.read_text()) == contract, 'calibration source/contract drift'
    else: G.S.atomic(contract, proof, True)
    panel = torch.load(G.S.OUT / 'panel.pt', weights_only=False, map_location='cpu')
    visible = torch.load(G.S.OUT / 'snap_visible.pt', weights_only=False, map_location='cpu')['visible']
    bank = torch.load(G.S.OUT / 'snap_bank_readers.pt', weights_only=False, map_location='cpu')
    reader = G.S.Readers(bank)
    episodes = [r['episode_id'] for r in panel['roots']]
    indices = {key: np.array([episodes.index(e) for e in value]) for key, value in original_contract['split'].items()}
    labels = G.SB.AllProbes.facts(visible[:, 16:32])['mobs'].float()
    test_labels = labels[indices['TEST']]
    weights = np.random.default_rng(20261012).multinomial(50, np.full(50, 1 / 50), size=original_contract['bootstrap_draws'])
    realhead = torch.load(out / 'real_head.pt', weights_only=False, map_location='cpu')['head']
    for update in a.updates:
        for arm in ('D',) if update == 0 else ('D', 'L1', 'no_max'):
            model_result = out / f'u{update:04d}_{arm}.json'
            assert model_result.exists(), 'refit must complete first'
            predpath = G.S.OUT / a.prefix / f'eval_u{update:04d}' / f'pred_{arm}.pt'
            inputs = {'pred_sha256': G.S.sha(predpath), 'refit_result_sha256': G.S.sha(model_result)}
            dst = out / f'calibration_u{update:04d}_{arm}.json'
            if dst.exists():
                old = json.loads(dst.read_text()); assert old['contract_id'] == cid and old['inputs'] == inputs
                G.log(stage='calibration_resume', update=update, arm=arm); continue
            G.log(stage='calibration_start', update=update, arm=arm)
            pred = torch.load(predpath, weights_only=False, map_location='cpu')
            scores_by_head, results, boot, calibrated = {}, {}, {}, {}
            for mode in ('tf', 'gen'):
                tokens = pred[mode][:, :, G.CELLS].float()
                dev_original = reader.mob(tokens[indices['DEV']].reshape(-1, 192))
                thresholds = G.thresholds(dev_original, labels[indices['DEV']].reshape(-1, 4))
                original = reader.mob(tokens[indices['TEST']].reshape(-1, 192)).reshape(50, 16, 62, 4)
                scores_by_head[mode + '/original'] = original
                r, b = G.metrics(original, test_labels, reader.thr.tolist(), weights)
                results[mode + '/original'] = r; boot[mode + '/original'] = b
                for option, thresh in thresholds.items():
                    key = mode + '/frozen_' + option
                    calibrated[key], bb = G.metrics(original, test_labels, thresh, weights)
                    boot[key] = bb
                for headname in ('new_real', 'new_tf', 'new_gen'):
                    head = realhead if headname == 'new_real' else torch.load(out / f'head_u{update:04d}_{arm}_{headname[4:]}.pt', weights_only=False, map_location='cpu')['head']
                    scores = G.head_scores(head, tokens[indices['TEST']].reshape(-1, 192)).float().reshape(50, 16, 62, 4)
                    key = mode + '/' + headname
                    scores_by_head[key] = scores
                    results[key], boot[key] = G.metrics(scores, test_labels, head['thresholds']['dev_f1'], weights)
                del tokens
            del pred
            intervals, auc_boot = {}, {}
            for key, scores in scores_by_head.items():
                intervals[key] = {}
                for mob in ('cow', 'zombie'):
                    j = G.TYPES.index(mob)
                    z = auc_draws(scores[:, 8:16, :, j].numpy(), test_labels[:, 8:16, :, j].numpy(), weights)
                    auc_boot[key, mob] = z
                    intervals[key][mob] = {'estimate': results[key][mob]['auc'],
                                          'ci95': np.quantile(z, [.025, .975]).tolist() if np.isfinite(z).all() else None,
                                          'undefined_draws': int((~np.isfinite(z)).sum())}
            auc_contrasts, threshold_contrasts = {}, {}
            for mode in ('tf', 'gen'):
                for left, right in [('new_gen', 'new_real'), ('new_gen', 'original'), ('new_real', 'original')]:
                    key = mode + '/' + left + '_minus_' + mode + '/' + right
                    auc_contrasts[key] = {}
                    for mob in ('cow', 'zombie'):
                        z = auc_boot[mode + '/' + left, mob] - auc_boot[mode + '/' + right, mob]
                        auc_contrasts[key][mob] = {'difference': results[mode + '/' + left][mob]['auc'] - results[mode + '/' + right][mob]['auc'],
                                                   'ci95': np.quantile(z, [.025, .975]).tolist() if np.isfinite(z).all() else None}
                for option in ('dev_f1', 'dev_count'):
                    key, baseline = mode + '/frozen_' + option, mode + '/original'
                    threshold_contrasts[key + '_minus_' + baseline] = {}
                    for mob in G.TYPES:
                        threshold_contrasts[key + '_minus_' + baseline][mob] = {}
                        for met in ('recall', 'precision', 'fp_per_true', 'census'):
                            pa, pb = calibrated[key][mob][met]['estimate'], results[baseline][mob][met]['estimate']
                            z = boot[key][mob, met] - boot[baseline][mob, met]
                            threshold_contrasts[key + '_minus_' + baseline][mob][met] = {
                                'difference': pa - pb if pa is not None and pb is not None else None,
                                'ci95': np.quantile(z, [.025, .975]).tolist() if np.isfinite(z).all() else None}
            payload = {'contract_id': cid, 'inputs': inputs, 'update': update, 'arm': arm,
                       'original_and_new_heads': results, 'frozen_weight_threshold_sensitivity': calibrated,
                       'auc_intervals': intervals, 'auc_contrasts': auc_contrasts, 'threshold_contrasts': threshold_contrasts}
            G.S.atomic(payload, dst, True)
            G.log(stage='calibration_done', update=update, arm=arm)
    G.log(stage='done', updates=a.updates)


if __name__ == '__main__': main()
