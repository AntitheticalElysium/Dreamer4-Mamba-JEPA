"""Predeclared CPU numerical controls for completed E17 H16 readers.

Hold selected heads and every saved feature fixed. Test whether FP32 cumulative
death conversion changes choices versus mathematically equivalent negative-log
survival argmin; count exact ties and saturation. Repeat conditional-hazard
subset deletion on true32-key curves, score H16 with uniform ties. No fitting,
GPU, threshold changes or outcome-selected model intervention. Atomic source/
input-bound output, reused DEV-B roots, paired episode-cluster intervals.
"""
import json
from pathlib import Path
import torch
import torch.nn.functional as F
import h16_resume as R
import e17_h16_diagnose as D

HERE = Path(__file__).resolve().parent


def main():
    torch.set_num_threads(2)
    directory = HERE / 'evals/resume/e17_h16_final'
    source = R.Store(directory, json.loads((directory / 'contract.json').read_text()))
    source_result = source.load('result'); rows = source.load('rows')
    assert source_result is not None
    p = next(D.RESUME.glob(D.WORLDS['mamba'] + '__w15__det__*'))
    original = R.Store(p, json.loads((p / 'contract.json').read_text())).load('result')
    P, opp = original['P_devB'], original['opportunity_devB']
    truth = P[opp, :, -1]
    names = list(source_result['worlds'])
    files = {}
    for name in ['result', 'rows'] + [f'{w}_batch_{i}' for w in names for i in range(0, len(P), 32)]:
        record = json.loads((directory / (name + '.json')).read_text())
        files[str(directory / record['file'])] = R.file_hash(directory / record['file'])
    spec = {'scope': __doc__, 'sources': {str(Path(f).resolve()): R.file_hash(f) for f in (__file__, R.__file__, D.__file__)},
            'inputs': files, 'source_contract': source.contract, 'truth_hash': R.tensor_hash(P),
            'runtime': {'torch': str(torch.__version__), 'threads': 2}}
    output = R.Store(HERE / 'evals/resume/e17_h16_numeric_control', spec)
    with output.lock():
        result = output.load('result')
        if result is None:
            worlds = {}
            for name in names:
                h = torch.cat([source.load(f'{name}_batch_{i}')['hazard'] for i in range(0, len(P), 32)], dim=1)
                energy = h.sum(-1)[:, opp]
                risk = 1 - torch.exp(-energy)
                picks_p, picks_e = risk.argmin(-1), energy.argmin(-1)
                T = truth[None].expand(3, -1, -1)
                safe_p = 1 - T.gather(-1, picks_p[..., None])[..., 0]
                safe_e = 1 - T.gather(-1, picks_e[..., None])[..., 0]
                assert (safe_p.mean(0).numpy() == rows['safe'][name]['full16']).all()
                worlds[name] = {'probability_safe': float(safe_p.mean()), 'stable_energy_safe': float(safe_e.mean()),
                    'choice_changes_over_3_heads': int((picks_p != picks_e).sum()),
                    'saturated_action_scores': int((risk == 1).sum()), 'action_scores': risk.numel(),
                    'minimum_risk_tie_count_mean': float((risk == risk.amin(-1, keepdim=True)).sum(-1).float().mean()),
                    'stable_minus_probability': D.paired(safe_e.mean(0).numpy(), safe_p.mean(0).numpy(), rows['seeds'])}
            prev = F.pad(P, (1, 0))[..., :-1]
            q = ((P - prev) / (1 - prev).clamp_min(1e-6)).clamp(0, 1)
            controls = {}
            for name, steps in D.BLOCKS.items():
                risk = (1 - (1 - q[..., steps]).prod(-1))[opp]
                if name == 'full16':
                    assert float((risk - truth).abs().max()) <= 1e-6
                ties = (risk == risk.amin(-1, keepdim=True)).float()
                controls[name] = {'H16_safe_uniform_exact_ties': float((ties * (1 - truth)).sum(1).div(ties.sum(1)).mean()),
                    'H16_safe_first_argmin': float((1 - truth[torch.arange(len(truth)), risk.argmin(-1)]).mean())}
            result = {'scope': __doc__, 'contract': spec, 'worlds': worlds, 'true_hazard_control': controls}
            output.save('result', result, 1)
        R.atomic_json(HERE / 'evals/e17_h16_numeric_control.json', result)
        print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
