"""Post-hoc matched CPU head-objective diagnostic on frozen Mamba7 E17 step16 features.

Predeclared: same 768 input features, FIT normalization, H.Hazard(snapshot)
architecture, seeds0/1/2, coherent batches32 roots x17 actions, AdamW1e-3/wd1e-4,
4000 updates, DEV-A selection every200. Only loss changes: soft-target BCE on
P(dead16), or within-root pairwise softplus weighted by true P differences.
Select and score by raw death logits argmin; rank scores are not calibrated
death probabilities. All-action diagnostic labels; no world/actor training.
Reused DEV-B results are exploratory, not a sealed repair verdict. Checkpoint
every200 includes optimizer/model/best state/sampler/CPU RNG; exclusive hash-bound
atomic resume. CPU threads2; measurements go to EDA logs, never NOTEBOOK.
"""
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import h16_resume as R
import e17_h16_diagnose as D
import check_h16_traj as H

HERE = Path(__file__).resolve().parent
STEPS = 4000


@torch.no_grad()
def logits(model, x):
    return torch.cat([model(x[i:i + 128].float().flatten(0, 1)[:, None])[:, 0].view(-1, 17)
                      for i in range(0, len(x), 128)])


def safe(scores, p):
    return 1 - p[torch.arange(len(p)), scores.argmin(1)]


def main():
    torch.set_num_threads(2)
    previous_path = HERE / 'evals/resume/e17_h16_s7_reader_diagnosis'
    c = json.loads((previous_path / 'contract.json').read_text())
    old = R.Store(previous_path, c)
    report = old.load('result')
    directory = next(D.RESUME.glob(D.WORLDS['mamba'] + '__w15__det__*'))
    wc = json.loads((directory / 'contract.json').read_text())
    ws = R.Store(directory, wc)
    official = ws.load('result')
    meta = {sp: torch.load(D.CACHE / f'{sp}_meta.pt', mmap=True, weights_only=False) for sp in ('fit', 'dev')}
    spec = {'scope': __doc__, 'script': R.file_hash(__file__), 'storage_source': R.file_hash(R.__file__),
            'architecture_source': R.file_hash(H.__file__), 'paired_source': R.file_hash(D.__file__),
            'previous_contract': old.contract, 'world_contract': ws.contract,
            'previous_result_hash': R.file_hash(previous_path / json.loads((previous_path / 'result.json').read_text())['file']),
            'normalization_hash': R.file_hash(previous_path / json.loads((previous_path / 'mamba_norm.json').read_text())['file']),
            'bootstrap': {'draws': 4000, 'seed': 20261007},
            'runtime': {'torch': str(torch.__version__), 'threads': 2, 'precision': 'CPU FP32, inputs FP16'}}
    store = R.Store(HERE / 'evals/resume/e17_h16_m7_head_objective', spec)
    with store.lock():
        norm = old.load('mamba_norm')
        features = {}
        for sp in ('fit', 'dev'):
            progress = json.loads((directory / f'features_{sp}.progress.json').read_text())
            assert progress['next'] == len(meta[sp]['seed'])
            mm = torch.from_numpy(np.memmap(directory / f'features_{sp}.f16', dtype=np.float16, mode='c', shape=progress['layout']['shape']))
            x = torch.empty((len(mm), 17, 768), dtype=torch.float16)
            for i in range(0, len(mm), 64):
                x[i:i + 64] = ((mm[i:i + 64, :, -1].float() - norm['mu']) / norm['sd']).half()
            features[sp] = x
        a, b = meta['dev']['seed'] % 2 == 0, meta['dev']['seed'] % 2 == 1
        Pfit, Pdev = meta['fit']['p16'], meta['dev']['p16']
        assert torch.equal(Pdev[b], official['P_devB'][..., -1])
        oppa, oppb = Pdev[a].amax(1) > Pdev[a].amin(1), official['opportunity_devB']
        xa, xb, pa, pb = features['dev'][a], features['dev'][b], Pdev[a], Pdev[b]
        results = {}
        for arm in ('bce', 'rank'):
            runs, predictions = [], []
            for seed in range(3):
                key = f'{arm}_{seed}'
                completed = store.load(key + '_result')
                if completed is None:
                    torch.manual_seed(seed)
                    g = torch.Generator().manual_seed(seed)
                    model = H.Hazard(768, 1)
                    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
                    best, best_state, start = -1., None, 0
                    state = store.load(key)
                    if state is not None:
                        model.load_state_dict(state['model']); opt.load_state_dict(state['optimizer'])
                        best, best_state, start = state['best'], state['best_state'], state['step']
                        g.set_state(state['generator']); torch.set_rng_state(state['cpu_rng'])
                    for step in range(start, STEPS):
                        idx = torch.randint(len(Pfit), (32,), generator=g)
                        xx = features['fit'][idx].float()
                        y = Pfit[idx]
                        z = model(xx.flatten(0, 1)[:, None])[:, 0].view(-1, 17)
                        if arm == 'bce':
                            loss = F.binary_cross_entropy_with_logits(z, y)
                        else:
                            weight = (y[:, None, :] - y[:, :, None]).clamp_min(0)
                            loss = (weight * F.softplus(z[:, :, None] - z[:, None, :])).sum() / weight.sum().clamp_min(1e-9)
                        assert torch.isfinite(loss)
                        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
                        if (step + 1) % 200 == 0:
                            value = float(safe(logits(model, xa), pa)[oppa].mean())
                            if value > best:
                                best = value; best_state = {k: v.detach().clone() for k,v in model.state_dict().items()}
                            store.save(key, {'step': step + 1, 'model': model.state_dict(), 'optimizer': opt.state_dict(),
                                'best': best, 'best_state': best_state, 'generator': g.get_state(), 'cpu_rng': torch.get_rng_state()}, step + 1)
                            print(json.dumps({'arm': arm, 'seed': seed, 'update': step + 1, 'loss': float(loss.detach()), 'devA': value}), flush=True)
                    model.load_state_dict(best_state)
                    score = logits(model.eval(), xb)
                    completed = {'devA': best, 'raw_logits': score, 'safe': safe(score, pb)[oppb]}
                    store.save(key + '_result', completed, 1)
                runs.append(completed['safe']); predictions.append(completed['raw_logits'])
            rows = torch.stack(runs)
            prob = torch.stack(predictions).sigmoid()
            results[arm] = {'safe': rows.mean(0), 'raw_logits': torch.stack(predictions),
                           'score': float(rows.mean()), 'per_head_seed': rows.mean(1).tolist(),
                           'brier_mean_head_seed': float((prob - pb).square().mean()),
                           'selected_actions_counts': torch.bincount(torch.stack(predictions)[:, oppb].argmin(-1).flatten(), minlength=17).tolist()}
        seed = meta['dev']['seed'][b][oppb].numpy()
        prior_rows = 1 - pb[:, report['prior_action']][oppb]
        contrasts = {'rank_minus_bce': D.paired(results['rank']['safe'].numpy(), results['bce']['safe'].numpy(), seed),
                     'rank_minus_prior': D.paired(results['rank']['safe'].numpy(), prior_rows.numpy(), seed)}
        raw = {'results': results, 'seeds': seed, 'prior_rows': prior_rows}
        store.save('rows', raw, 1)
        result = {'scope': __doc__, 'contract': spec, 'world': D.WORLDS['mamba'], 'roots': len(seed),
            'scores': {arm:{k:v for k,v in item.items() if k not in ('safe', 'raw_logits')} for arm,item in results.items()},
            'contrasts': contrasts, 'prior': float(prior_rows.mean())}
        store.save('result', result, 1)
        R.atomic_json(HERE / 'evals/e17_h16_m7_head_objective.json', result)
        print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
