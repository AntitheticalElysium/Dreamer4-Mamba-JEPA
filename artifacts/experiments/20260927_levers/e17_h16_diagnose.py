"""Post-hoc CPU diagnosis of completed seed7 E17 H16 readers.

Rules fixed before execution: independently reproduce saved per-root safe choices;
paired episode-cluster intervals against prior, attention, and snapshot; root
zombie-adjacency strata. Then replay the selected trajectory heads on their exact
saved features, retaining hazard terms at steps1,1-4,1-8,9-16,5-16,16 or all16.
No head fitting, world forward, GPU, or parameter changes. This intervenes on the
existing reader's aggregation, not on world memory. It cannot locate information
absence or prove that a trained short-context world is equivalent to a long one.
CPU full-reader reproduction (scores <=2e-5, safe mean <=1e-6, choices identical)
is required before interpreting temporal interventions. Preserved source/input-
bound batches and norms support resume. Logs are EDA; agents update NOTEBOOK.
"""
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import h16_resume as R
import check_h16_traj as H

HERE = Path(__file__).resolve().parent
CACHE = Path('artifacts/eda/deepeval_v1')
RESUME = CACHE / 'h16-resume-v1'
WORLDS = {'attention': 'corrt_rawlong_teacher_s7_L16b40_from36000',
          'mamba': 'corrt_rawlong_teacher_s7_fmamba_L16b40_from36000'}
BLOCKS = {'full16': list(range(16)), 'step1': [0], 'first4': list(range(4)),
          'first8': list(range(8)), 'last8': list(range(8, 16)),
          'omit_first4': list(range(4, 16)), 'step16': [15]}


def paired(left, right, seeds):
    groups = [np.flatnonzero(seeds == s) for s in np.unique(seeds)]
    counts = np.array([len(g) for g in groups], dtype=float)
    sums = np.array([(left[g] - right[g]).sum() for g in groups])
    rng = np.random.default_rng(20261007)
    ii = rng.integers(len(groups), size=(4000, len(groups)))
    delta = sums[ii].sum(1) / counts[ii].sum(1)
    lo, hi = np.quantile(delta, [.025, .975])
    return {'difference': float(np.mean(left - right)), 'interval95': [float(lo), float(hi)],
            'episode_seed_clusters': len(groups), 'roots': len(seeds)}


def main():
    torch.set_num_threads(2)
    directories = {k: next(RESUME.glob(v + '__w15__det__*')) for k, v in WORLDS.items()}
    source_inputs = {}
    stores = {}
    for label, p in directories.items():
        c = json.loads((p / 'contract.json').read_text())
        for category in ('sources', 'checkpoints'):
            for name, digest in c[category].items():
                assert R.file_hash(name) == digest, name
        stores[label] = R.Store(p, c)
        assert stores[label].load('result') is not None
        for name in ['contract.json', 'result.json', 'features_fit.f16', 'features_dev.f16']:
            source_inputs[str(p / name)] = R.file_hash(p / name)
        ri = json.loads((p / 'result.json').read_text())
        source_inputs[str(p / ri['file'])] = R.file_hash(p / ri['file'])
        for seed in range(3):
            hi = json.loads((p / f'head_trajectory_{seed}.json').read_text())
            source_inputs[str(p / hi['file'])] = R.file_hash(p / hi['file'])
    for name in ('fit_meta.pt', 'dev_meta.pt'):
        source_inputs[str(CACHE / name)] = R.file_hash(CACHE / name)
    spec = {'scope': __doc__, 'script': R.file_hash(__file__), 'storage': R.file_hash(R.__file__),
            'reader_source': R.file_hash(H.__file__), 'inputs': source_inputs, 'blocks': BLOCKS,
            'bootstrap': {'draws': 4000, 'seed': 20261007},
            'runtime': {'torch': str(torch.__version__), 'numpy': np.__version__, 'threads': 2,
                        'precision': 'CPU FP32 with original FP16 feature standardization'}}
    out_store = R.Store(HERE / 'evals/resume/e17_h16_s7_reader_diagnosis', spec)
    with out_store.lock():
        fit_meta = torch.load(CACHE / 'fit_meta.pt', mmap=True, weights_only=False)
        meta = torch.load(CACHE / 'dev_meta.pt', mmap=True, weights_only=False)
        b_indices = torch.where(meta['seed'] % 2 == 1)[0]
        fit_opp = fit_meta['p16'].amax(1) > fit_meta['p16'].amin(1)
        prior = int(fit_meta['p16'][fit_opp].mean(0).argmin())
        base = stores['attention'].load('result')
        opp = base['opportunity_devB']
        truth = base['P_devB'][..., -1]
        assert torch.equal(meta['seed'][b_indices], base['seed_devB'])
        assert torch.equal(meta['p16'][b_indices], truth)
        seeds = base['seed_devB'][opp].numpy()
        prior_rows = (1 - truth[:, prior])[opp].numpy()
        z = meta['visible'][b_indices, 1071:1512].reshape(-1, 7, 9, 7)[..., 0]
        adjacent = (z[:, 2, 4] + z[:, 4, 4] + z[:, 3, 3] + z[:, 3, 5] > 0)[opp].numpy()
        strata = {'all': np.ones(len(seeds), bool), 'adjacent_zombie': adjacent,
                  'no_adjacent_zombie': ~adjacent}
        rows, summaries = {}, {}
        for label, path in directories.items():
            official = stores[label].load('result')
            assert torch.equal(official['P_devB'], base['P_devB'])
            assert torch.equal(official['seed_devB'], base['seed_devB'])
            assert torch.equal(official['opportunity_devB'], opp)
            official_rows = {}
            for arm in ('trajectory', 'snapshot'):
                values = []
                for seed in range(3):
                    picks = official['raw_scores'][f'{arm}_{seed}'].argmin(1)
                    values.append((1 - truth[torch.arange(len(truth)), picks])[opp])
                official_rows[arm] = torch.stack(values).mean(0).numpy()
                assert np.max(np.abs(official_rows[arm] - official['safe_rows'][arm].numpy())) == 0
            norm = out_store.load(label + '_norm')
            if norm is None:
                info = json.loads((path / 'features_fit.progress.json').read_text())
                assert info['next'] == len(fit_meta['seed'])
                shape = info['layout']['shape']
                mm = np.memmap(path / 'features_fit.f16', dtype=np.float16, mode='c', shape=shape)
                flat = torch.from_numpy(mm).flatten(0, 2)
                # Literal normalization from the immutable evaluator, including its last partial chunk.
                mu = torch.stack([flat[i:i + 65536].float().mean(0) for i in range(0, len(flat), 65536)]).mean(0)
                sd = torch.stack([flat[i:i + 65536].float().std(0) for i in range(0, len(flat), 65536)]).mean(0).clamp(min=1e-3)
                norm = {'mu': mu, 'sd': sd}
                out_store.save(label + '_norm', norm, 1)
                del flat, mm
            di = json.loads((path / 'features_dev.progress.json').read_text())
            assert di['next'] == len(meta['seed'])
            dev = torch.from_numpy(np.memmap(path / 'features_dev.f16', dtype=np.float16, mode='c', shape=di['layout']['shape']))
            models = []
            for seed in range(3):
                state = stores[label].load(f'head_trajectory_{seed}')
                assert state['step'] == state['steps'] == 4000
                model = H.Hazard(768, 16)
                model.load_state_dict(state['best_state']); models.append(model.eval())
            hazards = []
            with torch.no_grad():
                for lo in range(0, len(b_indices), 32):
                    key = f'{label}_batch_{lo}'
                    saved = out_store.load(key)
                    if saved is None:
                        idx = b_indices[lo:lo + 32]
                        x = ((dev[idx].float() - norm['mu']) / norm['sd']).half()
                        # Full16 first, then delete hazard terms without changing feature inputs or step embeddings.
                        h = torch.stack([F.softplus(m(x.flatten(0, 1).float())).view(len(idx), 17, 16) for m in models])
                        saved = {'hazard': h, 'indices': idx}
                        out_store.save(key, saved, lo + len(idx))
                        print(json.dumps({'world': label, 'cpu_head_roots': lo + len(idx), 'total': len(b_indices)}), flush=True)
                    assert torch.equal(saved['indices'], b_indices[lo:lo + 32])
                    hazards.append(saved['hazard'])
            hazard = torch.cat(hazards, dim=1)
            risks = {block: 1 - torch.exp(-hazard[..., steps].sum(-1)) for block, steps in BLOCKS.items()}
            score_delta = max(float((risks['full16'][s] - official['raw_scores'][f'trajectory_{s}']).abs().max()) for s in range(3))
            choice_mismatch = sum(int((risks['full16'][s].argmin(1) != official['raw_scores'][f'trajectory_{s}'].argmin(1)).sum()) for s in range(3))
            assert score_delta <= 2e-5 and choice_mismatch == 0, (label, score_delta, choice_mismatch)
            choice, safe = {}, {}
            for block, risk in risks.items():
                picked = risk.argmin(-1)
                choice[block] = picked[:, opp]
                safe[block] = torch.stack([1 - truth[torch.arange(len(truth)), p] for p in picked])[:, opp].mean(0).numpy()
            assert np.max(np.abs(safe['full16'] - official_rows['trajectory'])) <= 1e-6
            rows[label] = {'safe': safe, 'choices': choice, 'official': official_rows}
            summaries[label] = {'official': official['res'], 'reproduction_max_score_delta': score_delta,
                'reproduction_choice_mismatches': choice_mismatch, 'strata': {}}
            for group, use in strata.items():
                summaries[label]['strata'][group] = {'roots': int(use.sum()), 'prior': float(prior_rows[use].mean()),
                    'trajectory_minus_prior': paired(safe['full16'][use], prior_rows[use], seeds[use]),
                    'trajectory_minus_snapshot': paired(safe['full16'][use], official_rows['snapshot'][use], seeds[use]),
                    'blocks': {block: {'safe': float(val[use].mean()),
                        'minus_full16': paired(val[use], safe['full16'][use], seeds[use]),
                        'chosen_actions_counts_over_3_head_seeds': torch.bincount(choice[block][:, use].flatten(), minlength=17).tolist()}
                        for block, val in safe.items()}}
        contrasts = {group: paired(rows['mamba']['safe']['full16'][use], rows['attention']['safe']['full16'][use], seeds[use]) for group, use in strata.items()}
        raw = {'seed': seeds, 'prior_rows': prior_rows, 'strata': strata, 'rows': rows}
        out_store.save('rows', raw, 1)
        result = {'scope': __doc__, 'contract': spec, 'prior_action': prior, 'worlds': summaries,
                  'mamba_minus_attention': contrasts, 'head_seed_averaging': 'mean safe score of each seed own choice; not an ensemble choice'}
        out_store.save('result', result, 1)
        R.atomic_json(HERE / 'evals/e17_h16_s7_reader_diagnosis.json', result)
        print(json.dumps({'mamba_minus_attention': contrasts, 'worlds': {k: v['strata']['all'] for k,v in summaries.items()}}), flush=True)


if __name__ == '__main__':
    main()
