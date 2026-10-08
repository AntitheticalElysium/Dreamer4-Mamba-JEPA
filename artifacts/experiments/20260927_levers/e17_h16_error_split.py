"""Post-hoc CPU error decomposition for E17's completed seed7 hazard readers.

Rules fixed before execution: use preserved, reproduced head hazards only;
compare cumulative death by8, additional unconditional death mass from9-16,
and cumulative death by16. Decompose MSE exactly into episode/root mean error
and within-root action-contrast error; report centered SD/correlation across
all DEV-B roots, H16 opportunities, and opportunities with no true action
variation through8. Also count improved/worsened/tied selected-action changes
when late head terms are included. Average metrics over three heads; no
ensemble decision, world/head fitting or GPU. This diagnoses outputs, not the
cause of feature loss. Source/input-bound atomic evidence; EDA logging only.
"""
import json
from pathlib import Path

import numpy as np
import torch

import h16_resume as R
import e17_h16_diagnose as D

HERE = Path(__file__).resolve().parent


def metrics(pred, truth, use):
    p, t = pred[:, use].double(), truth[use].double()
    error = p - t
    pm, tm = p.mean(-1, keepdim=True), t.mean(-1, keepdim=True)
    pc, tc = p - pm, t - tm
    mse = error.square().mean()
    common = (pm - tm).square().mean()
    contrast = (pc - tc).square().mean()
    assert abs(float(mse - common - contrast)) <= 1e-12
    denom = (pc.square().sum((1, 2)) * tc.square().sum()).sqrt()
    corr = (pc * tc).sum((1, 2)) / denom.clamp_min(1e-12)
    return {'MSE': float(mse), 'root_mean_MSE': float(common),
            'within_root_action_contrast_MSE': float(contrast),
            'predicted_action_contrast_SD': float(pc.square().mean().sqrt()),
            'true_action_contrast_SD': float(tc.square().mean().sqrt()),
            'centered_correlation_mean_head_seed': float(corr.mean()) if bool(denom.gt(1e-12).all()) else None,
            'prediction_mean': float(p.mean()), 'truth_mean': float(t.mean())}


def main():
    torch.set_num_threads(2)
    directory = HERE / 'evals/resume/e17_h16_s7_reader_diagnosis'
    old = R.Store(directory, json.loads((directory / 'contract.json').read_text()))
    raw = old.load('rows')
    worlddir = next(D.RESUME.glob(D.WORLDS['mamba'] + '__w15__det__*'))
    official = R.Store(worlddir, json.loads((worlddir / 'contract.json').read_text())).load('result')
    P, opp = official['P_devB'], official['opportunity_devB']
    assert np.array_equal(official['seed_devB'][opp].numpy(), raw['seed'])
    named_inputs = {str(directory / 'contract.json'): R.file_hash(directory / 'contract.json')}
    for name in ['rows'] + [f'{w}_batch_{i}' for w in D.WORLDS for i in range(0, len(P), 32)]:
        record = json.loads((directory / f'{name}.json').read_text())
        f = directory / record['file']
        named_inputs[str(f)] = R.file_hash(f)
    spec = {'scope': __doc__, 'script': R.file_hash(__file__), 'storage': R.file_hash(R.__file__),
            'source_contract': old.contract, 'world_contract': official['res'].get('resume_contract'),
            'inputs': named_inputs, 'torch': str(torch.__version__), 'threads': 2}
    store = R.Store(HERE / 'evals/resume/e17_h16_s7_error_split', spec)
    with store.lock():
        output = store.load('result')
        if output is None:
            late_only = opp & ~(P[..., :8].amax(1) > P[..., :8].amin(1)).any(-1)
            assert int(late_only.sum()) == 137
            truth = {'early8': P[..., 7], 'late_increment': P[..., 15] - P[..., 7], 'full16': P[..., 15]}
            cohorts = {'all_DEV_B': torch.ones_like(opp), 'opportunity16': opp, 'only_after8': late_only}
            worlds = {}
            for label in D.WORLDS:
                h = torch.cat([old.load(f'{label}_batch_{i}')['hazard'] for i in range(0, len(P), 32)], dim=1)
                cumulative = 1 - torch.exp(-h.cumsum(-1))
                pred = {'early8': cumulative[..., 7], 'late_increment': cumulative[..., 15] - cumulative[..., 7], 'full16': cumulative[..., 15]}
                choices = raw['rows'][label]['choices']
                full, early = choices['full16'], choices['first8']
                actual = P[opp, :, -1]
                chosen_truth = lambda picks: actual.unsqueeze(0).expand(3, -1, -1).gather(-1, picks[..., None])[..., 0]
                change = full != early
                delta = chosen_truth(early) - chosen_truth(full)
                worlds[label] = {'cohorts': {name: {'roots': int(use.sum()),
                    'metrics': {key: metrics(pred[key], truth[key], use) for key in truth}}
                    for name, use in cohorts.items()},
                    'late_choice_changes_over_3_head_seeds': {'changed': int(change.sum()),
                        'improved': int((change & delta.gt(0)).sum()), 'worsened': int((change & delta.lt(0)).sum()),
                        'equal_outcome': int((change & delta.eq(0)).sum()), 'mean_safe_change': float(delta.mean())}}
            output = {'scope': __doc__, 'contract': spec, 'worlds': worlds}
            store.save('result', output, 1)
        R.atomic_json(HERE / 'evals/e17_h16_s7_error_split.json', output)
        print(json.dumps(output), flush=True)


if __name__ == '__main__':
    main()
