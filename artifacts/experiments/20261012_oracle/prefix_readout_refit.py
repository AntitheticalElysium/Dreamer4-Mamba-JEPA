"""Matched generated-prefix readout diagnostic, fixed FIT/DEV/TEST split.

Reuses the prior capacity, ridge family and numerical routines without modifying
source-bound helpers. TEST excludes head fitting; the world roots are reused
DEV, not fresh sealed judgment. Both training arms keep D loss fixed.
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
import generated_readout_refit as BASE
S, SB = BASE.S, BASE.SB
ALPHAS, TYPES, CELLS = BASE.ALPHAS, BASE.TYPES, BASE.CELLS
log, head_scores, thresholds, fit = BASE.log, BASE.head_scores, BASE.thresholds, BASE.fit
auc, metrics = BASE.auc, BASE.metrics

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--updates', type=int, nargs='+', default=[0, 500, 1500, 3000])
    p.add_argument('--prefix', default='generated_prefix')
    p.add_argument('--draws', type=int, default=2000)
    a = p.parse_args()
    torch.set_num_threads(8)
    out = S.OUT / a.prefix / 'readout_refit'; out.mkdir(parents=True, exist_ok=True)
    panel = torch.load(S.OUT / 'panel.pt', weights_only=False, map_location='cpu')
    visible = torch.load(S.OUT / 'snap_visible.pt', weights_only=False, map_location='cpu')
    bank = torch.load(S.OUT / 'snap_bank_readers.pt', weights_only=False, map_location='cpu')
    reader = S.Readers(bank)
    assert panel['roots'] == visible['roots']
    eps = [r['episode_id'] for r in panel['roots']]; assert len(set(eps)) == len(eps) == 200
    perm = np.random.default_rng(20261012).permutation(200)
    indices = {'FIT': perm[:100], 'DEV': perm[100:150], 'TEST': perm[150:]}
    source = {str(Path(__file__).relative_to(ROOT)): S.sha(__file__),
              str(Path(BASE.__file__).relative_to(ROOT)): S.sha(BASE.__file__),
              str(Path(S.__file__).relative_to(ROOT)): S.sha(S.__file__),
              str(Path(SB.__file__).relative_to(ROOT)): S.sha(SB.__file__)}
    contract = {'format': 'oracle_prefix_generated_readout_refit_v1', 'sources': source,
                'inputs': {name: S.sha(S.OUT / name) for name in ('panel.pt', 'snap_visible.pt', 'snap_bank_readers.pt')},
                'split_seed': 20261012, 'split': {k: [eps[i] for i in v] for k, v in indices.items()},
                'capacity': 'FIT-only standardized 192 features plus regularized intercept ->4flags; no depth/location feature',
                'fit_objective': 'squared native flag error', 'alpha_family': ALPHAS,
                'alpha_selection': 'minimum joint four-flag DEV MSE', 'threshold_primary': 'per-type DEV maximum F1',
                'threshold_secondary': 'DEV count matching as AllProbes-style sensitivity', 'fit_depths': '1-16',
                'test_depths': '9-16', 'bootstrap_unit': 'episode', 'bootstrap_draws': a.draws,
                'scope': 'post-hoc reused DEV world roots; TEST separate from head fitting, not a fresh world judgment set'}
    contract = json.loads(json.dumps(contract)); cid = hashlib.sha256(json.dumps(contract, sort_keys=True).encode()).hexdigest()
    manifest = out / 'contract.json'
    if manifest.exists(): assert json.loads(manifest.read_text()) == contract, 'head-fit contract drift'
    else: S.atomic(contract, manifest, True)
    labels = SB.AllProbes.facts(visible['visible'][:, 16:32])['mobs'].float()
    real = panel['tokens'][:, 16:32, CELLS].float()
    real_path = out / 'real_head.pt'
    if real_path.exists():
        st = torch.load(real_path, weights_only=False, map_location='cpu'); assert st['contract_id'] == cid; real_head = st['head']
    else:
        real_head = fit(real, labels, indices['FIT'], indices['DEV'])
        S.atomic({'contract_id': cid, 'head': real_head}, real_path)
    weights = np.random.default_rng(20261012).multinomial(50, np.full(50, 1 / 50), size=a.draws)
    test_labels = labels[indices['TEST']]
    heads_base = {'new_real': real_head}
    sources_predict = {u: S.OUT / a.prefix / f'eval_u{u:04d}' for u in a.updates}
    for u, directory in sources_predict.items():
        arms = ('teacher',) if u == 0 else ('teacher', 'prefix')
        for arm in arms:
            path = directory / f'pred_{arm}.pt'
            if not path.exists(): raise FileNotFoundError(path)
            pred_hash = S.sha(path)
            dst = out / f'u{u:04d}_{arm}.json'
            if dst.exists():
                old = json.loads(dst.read_text()); assert old['contract_id'] == cid and old['prediction_sha256'] == pred_hash
                log(stage='result_resume', update=u, arm=arm); continue
            log(stage='model_start', update=u, arm=arm)
            st = torch.load(path, weights_only=False, map_location='cpu')
            x = {'real': real, 'tf': st['tf'][:, :, CELLS].float(), 'gen': st['gen'][:, :, CELLS].float()}
            del st
            heads = dict(heads_base)
            for mode in ('tf', 'gen'):
                hp = out / f'head_u{u:04d}_{arm}_{mode}.pt'
                if hp.exists():
                    hs = torch.load(hp, weights_only=False, map_location='cpu')
                    assert hs['contract_id'] == cid and hs['prediction_sha256'] == pred_hash
                    heads['new_' + mode] = hs['head']
                else:
                    h = fit(x[mode], labels, indices['FIT'], indices['DEV'])
                    heads['new_' + mode] = h
                    S.atomic({'contract_id': cid, 'prediction_sha256': pred_hash, 'head': h}, hp)
                log(stage='head_fit', update=u, arm=arm, mode=mode, alpha=heads['new_' + mode]['alpha'])
            results, boot = {}, {}
            for input_mode, tokens in x.items():
                q = tokens[indices['TEST']].reshape(-1, 192)
                for head_name in ('original', 'new_real', 'new_tf', 'new_gen'):
                    if head_name == 'original':
                        scores = reader.mob(q).reshape(50, 16, 62, 4)
                        threshold_options = {'original': reader.thr.tolist()}
                    else:
                        h = heads[head_name]
                        scores = head_scores(h, q).float().reshape(50, 16, 62, 4)
                        threshold_options = h['thresholds']
                    for option, thresh in threshold_options.items():
                        key = input_mode + '/' + head_name + '/' + option
                        results[key], b = metrics(scores, test_labels, thresh, weights)
                        boot[key] = b
            contrasts = {}
            for input_mode in ('real', 'tf', 'gen'):
                baseline = input_mode + '/original/original'
                for head_name in ('new_real', 'new_tf', 'new_gen'):
                    for option in ('dev_f1', 'dev_count'):
                        key = input_mode + '/' + head_name + '/' + option
                        contrast = {}
                        for mob in TYPES:
                            contrast[mob] = {}
                            for met in ('recall', 'precision', 'fp_per_true', 'census'):
                                pa, pb = results[key][mob][met]['estimate'], results[baseline][mob][met]['estimate']
                                z = boot[key][mob, met] - boot[baseline][mob, met]
                                contrast[mob][met] = {'difference': pa - pb if pa is not None and pb is not None else None,
                                                      'ci95': np.quantile(z, [.025, .975]).tolist() if np.isfinite(z).all() else None}
                        contrasts[key + '_minus_' + baseline] = contrast
            payload = {'contract_id': cid, 'prediction_sha256': pred_hash, 'updates': u, 'arm': arm,
                       'TEST_episodes': [eps[i] for i in indices['TEST']], 'fixed_original_thresholds': reader.thr.tolist(),
                       'heads': {k: {j: h[j] for j in ('alpha', 'dev_mse', 'dev_mse_by_alpha', 'thresholds')} for k, h in heads.items()},
                       'readings': results, 'contrasts': contrasts}
            S.atomic(payload, dst, True)
            log(stage='model_done', update=u, arm=arm, output=str(dst))
            del x
    log(stage='done', updates=a.updates)


if __name__ == '__main__': main()
