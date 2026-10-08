"""Post-hoc CPU true-hazard control for E17 reader temporal interventions.

Rules fixed before execution: identical hazard-term subsets as e17_h16_diagnose;
apply them to true 32-key conditional hazards and score the chosen action at H16.
Report first-argmin and uniform selection among exact ties. Classify roots as
action-dependent by4, newly dependent by8, or dependent only after8 using the
saved P arrays, not a fitted model. Within those strata compare existing head
full16/first8 choices with episode-cluster intervals. Calibration uses each
selected head's cumulative risks, averaged metrics over the three head seeds;
it is not an ensemble choosing a new action. No world/head forward, GPU or fitting.
Raw rows/source-bound atomic result; reused DEV-B panel, exploratory only.
"""
import json
from pathlib import Path

import numpy as np
import torch

import h16_resume as R
import e17_h16_diagnose as D

HERE = Path(__file__).resolve().parent


def main():
    torch.set_num_threads(2)
    path = HERE / 'evals/resume/e17_h16_s7_reader_diagnosis'
    c = json.loads((path / 'contract.json').read_text())
    source = R.Store(path, c)
    previous = source.load('result')
    saved_rows = source.load('rows')
    spec = {'scope': __doc__, 'script': R.file_hash(__file__), 'paired_source': R.file_hash(D.__file__),
            'storage_source': R.file_hash(R.__file__), 'previous_contract': source.contract,
            'previous_result': R.file_hash(path / json.loads((path / 'result.json').read_text())['file']),
            'blocks': D.BLOCKS, 'bootstrap': {'seed': 20261007, 'draws': 4000}}
    store = R.Store(HERE / 'evals/resume/e17_h16_s7_hazard_control', spec)
    with store.lock():
        output = store.load('result')
        if output is None:
            world_path = next(D.RESUME.glob(D.WORLDS['attention'] + '__w15__det__*'))
            wc = json.loads((world_path / 'contract.json').read_text())
            official = R.Store(world_path, wc).load('result')
            P = official['P_devB']
            opp = official['opportunity_devB']
            seed = official['seed_devB'][opp].numpy()
            assert np.array_equal(seed, saved_rows['seed'])
            prev = torch.nn.functional.pad(P, (1, 0))[..., :-1]
            q = ((P - prev) / (1 - prev).clamp_min(1e-6)).clamp(0, 1)
            p16 = P[opp, :, -1]
            references = {}
            for block, steps in D.BLOCKS.items():
                risk = (1 - (1 - q[..., steps]).prod(-1))[opp]
                if block == 'full16':
                    assert float((risk - p16).abs().max()) <= 1e-6
                chosen = risk.argmin(1)
                ties = (risk == risk.amin(1, keepdim=True)).float()
                uniform_tie_safe = (ties * (1 - p16)).sum(1) / ties.sum(1)
                references[block] = {'H16_safe_first_argmin': float((1 - p16[torch.arange(len(chosen)), chosen]).mean()),
                    'H16_safe_uniform_exact_ties': float(uniform_tie_safe.mean()),
                    'mean_minimum_risk_tie_count': float(ties.sum(1).mean())}
            varies = lambda k: (P[opp, :, k - 1].amax(1) > P[opp, :, k - 1].amin(1)).numpy()
            v4, v8 = varies(4), varies(8)
            groups = {'all': np.ones(len(seed), bool), 'varies_by4': v4,
                      'newly_varies_by8': ~v4 & v8, 'varies_only_after8': ~v8}
            assert np.all(v4 <= v8), 'Non-nested action-variation groups need explicit interpretation'
            assert sum(int(groups[k].sum()) for k in ('varies_by4', 'newly_varies_by8', 'varies_only_after8')) == len(seed)
            worlds = {}
            for label in D.WORLDS:
                h = torch.cat([source.load(f'{label}_batch_{lo}')['hazard'] for lo in range(0, len(P), 32)], dim=1)
                cumulative = 1 - torch.exp(-h.cumsum(-1))
                cal = {}
                for k in (1, 4, 8, 16):
                    pred, truth = cumulative[..., k - 1], P[..., k - 1]
                    centered_truth = truth - truth.mean(1, keepdim=True)
                    centered_pred = pred - pred.mean(2, keepdim=True)
                    cov = (centered_pred * centered_truth).sum((1, 2))
                    corr = cov / ((centered_pred.square().sum((1, 2)) * centered_truth.square().sum()).sqrt().clamp_min(1e-12))
                    cal[str(k)] = {'true_death_mean_all_DEV_B_actions': float(truth.mean()),
                        'predicted_death_mean': float(pred.mean()), 'Brier_average_head_seed': float((pred - truth).square().mean()),
                        'within_root_centered_correlation_mean_head_seed': float(corr.mean()),
                        'within_root_predicted_effect_SD': float(centered_pred.square().mean().sqrt()),
                        'within_root_true_effect_SD': float(centered_truth.square().mean().sqrt())}
                safe = saved_rows['rows'][label]['safe']
                worlds[label] = {'calibration': cal, 'groups': {}}
                for name, use in groups.items():
                    worlds[label]['groups'][name] = {'roots': int(use.sum()),
                        'full16': float(safe['full16'][use].mean()), 'first8': float(safe['first8'][use].mean()),
                        'full16_minus_first8': D.paired(safe['full16'][use], safe['first8'][use], seed[use]),
                        'full16_minus_prior': D.paired(safe['full16'][use], saved_rows['prior_rows'][use], seed[use])}
            output = {'scope': __doc__, 'contract': spec, 'true_hazard_references': references, 'worlds': worlds,
                      'scope_limit': 'Deletion changes event interval for true hazards too; late-only learned-head failure does not establish missing late state information.'}
            store.save('result', output, 1)
        R.atomic_json(HERE / 'evals/e17_h16_s7_hazard_control.json', output)
        print(json.dumps({'true_hazard_references': output['true_hazard_references'], 'worlds': output['worlds']}), flush=True)


if __name__ == '__main__':
    main()
