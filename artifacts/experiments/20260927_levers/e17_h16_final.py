"""E17 final H16 diagnosis, predeclared 2026-10-07 before execution.

Reproduce saved risk heads and action choices for both backbones/world seeds.
Reuse completed seed7 CPU hazards only with source/input contract verification;
compute seed8 on the exact frozen feature caches and original FIT normalization.
Measure paired prior/snapshot/backbone contrasts, temporal hazard-term subsets,
death timing and within-root action-effect compression. Repeat the existing
true-hazard deletion control. Roots/labels/head weights/thresholds fixed; CPU
threads2, no world/head fitting or GPU. Later-only failure does not prove missing
latent information. Reused DEV-B panel; intervals cluster149 episode seeds and
are conditional on these trained worlds, not uncertainty over training seeds.
Source/input-bound atomic batch checkpoints; logs only to EDA, no notebook writes.
"""
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import h16_resume as R
import check_h16_traj as H
import e17_h16_diagnose as D
import e17_h16_error_split as ES

HERE = Path(__file__).resolve().parent


def main():
    torch.set_num_threads(2)
    previous_dir = HERE / 'evals/resume/e17_h16_s7_reader_diagnosis'
    previous_spec = json.loads((previous_dir / 'contract.json').read_text())
    assert previous_spec['script'] == R.file_hash(D.__file__)
    old = R.Store(previous_dir, previous_spec)
    assert old.load('result') is not None
    stores, inputs = {}, {}
    for seed in (7, 8):
        for bb in ('attention', 'mamba'):
            name = f'corrt_rawlong_teacher_s{seed}' + ('_fmamba' if bb == 'mamba' else '') + '_L16b40_from36000'
            matches = list(D.RESUME.glob(name + '__w15__det__*'))
            assert len(matches) == 1, matches
            p = matches[0]
            spec = json.loads((p / 'contract.json').read_text())
            for kind in ('sources', 'checkpoints'):
                for f, digest in spec[kind].items():
                    assert R.file_hash(f) == digest, f
            source = R.Store(p, spec)
            official = source.load('result')
            assert official is not None
            stores[f'{bb}{seed}'] = (p, source, official)
            for f in ('contract.json', 'features_fit.f16', 'features_dev.f16'):
                inputs[str(p / f)] = R.file_hash(p / f)
            for f in ['result'] + [f'head_trajectory_{i}' for i in range(3)]:
                record = json.loads((p / (f + '.json')).read_text())
                inputs[str(p / record['file'])] = R.file_hash(p / record['file'])
    for sp in ('fit', 'dev'):
        inputs[str(D.CACHE / (sp + '_meta.pt'))] = R.file_hash(D.CACHE / (sp + '_meta.pt'))
    for f in ['rows'] + [f'{bb}_batch_{lo}' for bb in D.WORLDS for lo in range(0, 1366, 32)]:
        record = json.loads((previous_dir / (f + '.json')).read_text())
        inputs[str(previous_dir / record['file'])] = R.file_hash(previous_dir / record['file'])
    contract = {'scope': __doc__, 'sources': {str(Path(p).resolve()): R.file_hash(p)
        for p in (__file__, R.__file__, H.__file__, D.__file__, ES.__file__)},
        'inputs': inputs, 'previous_contract': old.contract,
        'world_contracts': {k: s.contract for k, (_, s, _) in stores.items()},
        'bootstrap': {'draws': 4000, 'seed': 20261007},
        'runtime': {'torch': str(torch.__version__), 'threads': 2, 'precision': 'CPU FP32 / original FP16 normalized inputs'}}
    out = R.Store(HERE / 'evals/resume/e17_h16_final', contract)
    with out.lock():
        result = out.load('result')
        if result is not None:
            print(json.dumps(result), flush=True)
            return
        fit_meta = torch.load(D.CACHE / 'fit_meta.pt', mmap=True, weights_only=False)
        meta = torch.load(D.CACHE / 'dev_meta.pt', mmap=True, weights_only=False)
        indices = torch.where(meta['seed'] % 2 == 1)[0]
        base = stores['mamba7'][2]
        P, opp = base['P_devB'], base['opportunity_devB']
        assert len(P) == 1366 and int(opp.sum()) == 1139
        assert torch.equal(P[..., -1], meta['p16'][indices])
        seed_array = base['seed_devB'][opp].numpy()
        oldrows = old.load('rows')
        assert np.array_equal(seed_array, oldrows['seed'])
        prior_action = old.load('result')['prior_action']
        prior_rows = (1 - P[opp, prior_action, -1]).numpy()
        late = ~(P[opp, :, :8].amax(1) > P[opp, :, :8].amin(1)).any(-1).numpy()
        groups = {'all': np.ones(int(opp.sum()), bool), 'adjacent_zombie': oldrows['strata']['adjacent_zombie'], 'only_after8': late}
        summary, rows = {}, {}
        truth = {'early8': P[..., 7], 'late_increment': P[..., -1] - P[..., 7], 'full16': P[..., -1]}
        for name, (directory, source, official) in stores.items():
            assert torch.equal(P, official['P_devB']) and torch.equal(opp, official['opportunity_devB'])
            assert torch.equal(base['seed_devB'], official['seed_devB'])
            bb, seed = name[:-1], int(name[-1])
            norm = out.load(name + '_norm')
            if norm is None:
                if seed == 7:
                    norm = old.load(bb + '_norm')
                else:
                    progress = json.loads((directory / 'features_fit.progress.json').read_text())
                    assert progress['next'] == len(fit_meta['seed'])
                    mm = torch.from_numpy(np.memmap(directory / 'features_fit.f16', dtype=np.float16, mode='c', shape=progress['layout']['shape'])).flatten(0, 2)
                    norm = {'mu': torch.stack([mm[i:i + 65536].float().mean(0) for i in range(0, len(mm), 65536)]).mean(0),
                            'sd': torch.stack([mm[i:i + 65536].float().std(0) for i in range(0, len(mm), 65536)]).mean(0).clamp_min(1e-3)}
                    del mm
                out.save(name + '_norm', norm, 1)
            progress = json.loads((directory / 'features_dev.progress.json').read_text())
            assert progress['next'] == len(meta['seed'])
            dev = torch.from_numpy(np.memmap(directory / 'features_dev.f16', dtype=np.float16, mode='c', shape=progress['layout']['shape']))
            models = []
            for i in range(3):
                saved = source.load(f'head_trajectory_{i}')
                assert saved['step'] == 4000
                model = H.Hazard(768, 16)
                model.load_state_dict(saved['best_state']); models.append(model.eval())
            pieces = []
            for lo in range(0, len(P), 32):
                key = name + f'_batch_{lo}'
                part = out.load(key)
                if part is None:
                    if seed == 7:
                        part = old.load(f'{bb}_batch_{lo}')
                    else:
                        idx = indices[lo:lo + 32]
                        x = ((dev[idx].float() - norm['mu']) / norm['sd']).half()
                        with torch.no_grad():
                            hazard = torch.stack([F.softplus(m(x.flatten(0, 1).float())).view(len(idx), 17, 16) for m in models])
                        part = {'hazard': hazard, 'indices': idx}
                    out.save(key, part, lo + len(part['indices']))
                assert torch.equal(part['indices'], indices[lo:lo + 32])
                pieces.append(part['hazard'])
                print(json.dumps({'world': name, 'roots_done': lo + len(part['indices']), 'total': len(P)}), flush=True)
            h = torch.cat(pieces, dim=1)
            risks = {k: 1 - torch.exp(-h[..., steps].sum(-1)) for k, steps in D.BLOCKS.items()}
            error = max(float((risks['full16'][i] - official['raw_scores'][f'trajectory_{i}']).abs().max()) for i in range(3))
            mismatch = sum(int((risks['full16'][i].argmin(1) != official['raw_scores'][f'trajectory_{i}'].argmin(1)).sum()) for i in range(3))
            assert error <= 2e-5 and mismatch == 0, (name, error, mismatch)
            safe_rows = {k: torch.stack([1 - P[torch.arange(len(P)), p, -1] for p in risk.argmin(-1)])[:, opp].mean(0).numpy() for k, risk in risks.items()}
            assert np.max(np.abs(safe_rows['full16'] - official['safe_rows']['trajectory'].numpy())) <= 1e-6
            cumulative = 1 - torch.exp(-h.cumsum(-1))
            predicted = {'early8': cumulative[..., 7], 'late_increment': cumulative[..., -1] - cumulative[..., 7], 'full16': cumulative[..., -1]}
            late_full = torch.zeros_like(opp); late_full[opp] = torch.from_numpy(late)
            cohorts = {'all_DEV_B': torch.ones_like(opp), 'opportunity16': opp, 'only_after8': late_full}
            choices = {k: v.argmin(-1)[:, opp] for k, v in risks.items()}
            actual = P[opp, :, -1].unsqueeze(0).expand(3, -1, -1)
            chosen = lambda k: actual.gather(-1, choices[k][..., None])[..., 0]
            delta = chosen('first8') - chosen('full16'); changed = choices['first8'] != choices['full16']
            summary[name] = {'official': official['res'], 'reproduction': {'max_score_error': error, 'choice_mismatches': mismatch},
                'groups': {g: {'roots': int(use.sum()), 'scores': {k: float(v[use].mean()) for k, v in safe_rows.items()},
                    'full_minus_prior': D.paired(safe_rows['full16'][use], prior_rows[use], seed_array[use]),
                    'full_minus_first8': D.paired(safe_rows['full16'][use], safe_rows['first8'][use], seed_array[use])} for g, use in groups.items()},
                'risk_error': {g: {'roots': int(use.sum()), 'metrics': {k: ES.metrics(predicted[k], truth[k], use) for k in truth}} for g, use in cohorts.items()},
                'late_choice_changes': {'changed': int(changed.sum()), 'improved': int((changed & delta.gt(0)).sum()),
                    'worsened': int((changed & delta.lt(0)).sum()), 'equal_outcome': int((changed & delta.eq(0)).sum()), 'safe_change': float(delta.mean())}}
            rows[name] = safe_rows
            del dev, models, h, cumulative, pieces
        contrasts = {str(seed): {g: D.paired(rows[f'mamba{seed}']['full16'][use], rows[f'attention{seed}']['full16'][use], seed_array[use]) for g, use in groups.items()} for seed in (7, 8)}
        out.save('rows', {'safe': rows, 'seeds': seed_array, 'prior': prior_rows, 'groups': groups}, 1)
        result = {'scope': __doc__, 'contract': contract, 'worlds': summary, 'mamba_minus_attention': contrasts,
            'readings': {'trajectory_gain_both_seeds': {bb: all(stores[bb + str(s)][2]['res']['traj_minus_snapshot']['excludes_zero'] and stores[bb + str(s)][2]['res']['traj_minus_snapshot']['difference'] > 0 for s in (7, 8)) for bb in ('mamba', 'attention')},
                'one_real_future_minus_001_both_seeds': {bb: all(stores[bb + str(s)][2]['res']['trajectory'] >= base['references']['one_real_future'] - .01 for s in (7, 8)) for bb in ('mamba', 'attention')},
                'original_mamba_H16_edge_002_both_seeds': all(contrasts[str(s)]['all']['difference'] >= .02 for s in (7, 8))}}
        out.save('result', result, 1)
        R.atomic_json(HERE / 'evals/e17_h16_final.json', result)
        print(json.dumps({'readings': result['readings'], 'mamba_minus_attention': contrasts,
            'worlds': {k: {'groups': v['groups'], 'late_error': v['risk_error']['opportunity16']['metrics']['late_increment'],
                'late_only_early_risk': v['risk_error']['only_after8']['metrics']['early8']} for k, v in summary.items()}}), flush=True)


if __name__ == '__main__':
    main()
