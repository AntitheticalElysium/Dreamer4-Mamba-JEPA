"""Short E20 fixed-TRAIN-panel diagnostic; no optimization and no CUDA.

Exploratory post-hoc panel, not a population estimate or convergence certificate.
Reference FP32 recurrence is checked against stored CUDA predictions before the
learning curve is interpreted. Read only selected, hash-verified feature chunks.
Evidence and logs belong under EDA; this process never writes NOTEBOOK.md.
"""
import dataclasses
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import e20_train as E
from d4mj.mamba_recurrence import FunctionalMamba2


def hud(tokens, reader):
    features = tokens[..., 63:81, :].float().flatten(-2)
    out = ((features - reader['mu']) / reader['sd']).double() @ reader['w'][:-1] + reader['w'][-1]
    return out[..., 0].float() * 9


def main():
    torch.set_num_threads(3)
    started = time.monotonic()
    out = E.D.OUT
    result_path = out / 'analysis_checkpoint_curve_cpu.json'
    if result_path.exists():
        raise RuntimeError('Preserve the previous diagnostic; do not overwrite')
    labels_path = out / 'pool/labels.pt'
    labels = torch.load(labels_path, map_location='cpu', weights_only=False)
    manifest = json.loads((out / 'pool.json').read_text())
    assert E.R.file_hash(labels_path) == manifest['labels_sha256']
    spec = json.loads((out / 'pool/contract.json').read_text())
    progress = json.loads((out / 'pool/tokens.progress.json').read_text())
    assert progress['next'] == spec['shape'][0]
    mm = np.memmap(out / 'pool/tokens.f16', dtype=np.float16, mode='r', shape=tuple(spec['shape']))
    generator = torch.Generator().manual_seed(20261007)
    ordinary = labels['cohorts']['C'][1]
    groups = {
        'death': labels['cohorts']['C'][0],
        'ordinary_nonfresh': ordinary[~labels['fresh'][ordinary]],
        'fresh_ordinary': ordinary[labels['fresh'][ordinary]],
        'unchanged': labels['cohorts']['C'][2],
    }
    selected = {k: ids[torch.randperm(len(ids), generator=generator)[:8]] for k, ids in groups.items()}
    ids = torch.cat(list(selected.values()))
    verified = []
    for chunk in progress['chunks']:
        a, b = chunk['start'], chunk['stop']
        if ((ids >= a) & (ids < b)).any():
            assert hashlib.sha256(memoryview(mm[a:b]).cast('B')).hexdigest() == chunk['sha256']
            verified.append(chunk)
    token_values = torch.from_numpy(np.array(mm[ids.numpy()])).float()
    actions = labels['actions'][ids]
    truth = labels['dh'][ids, -1].float()
    loss_reader = torch.load(out / 'health_reader.pt', map_location='cpu', weights_only=False)
    hud_reader = torch.load(out / 'analysis_hud_reader.pt', map_location='cpu', weights_only=False)
    world, parent = E.make_world(7, 'cpu')
    world.eval()
    for module in world.modules():
        if isinstance(module, FunctionalMamba2):
            module.settings = dataclasses.replace(module.settings, backend='reference')
    final = E.E.T.OUT / 'e20_C_s7_fmamba_fromM16.pt'
    world.load_state_dict(torch.load(final, map_location='cpu', weights_only=False)['world'])
    parity = torch.load(out / 'analysis_cpu_parity_slice.pt', map_location='cpu', weights_only=False)
    with torch.no_grad():
        cp = world(parity['src'].float(), parity['actions'])[0][:, -1].float()
    gp = parity['teacher_prediction'].float()
    parity_record = {
        'token_mean_abs': float((cp - gp).abs().mean()),
        'token_max_abs': float((cp - gp).abs().max()),
        'hud_health_max_abs': float((hud(cp, hud_reader) - hud(gp, hud_reader)).abs().max()),
        'loss_health_max_abs': float((E.read_health(cp[:, 63], loss_reader) - E.read_health(gp[:, 63], loss_reader)).abs().max() * 9),
        'n': len(cp), 'backend': 'FP32 CPU reference versus stored BF16 CUDA/fp16 output',
    }
    # This is a diagnostic numerical guard, not a changed scientific gate.
    assert max(parity_record['hud_health_max_abs'], parity_record['loss_health_max_abs']) < .1, parity_record
    variants = [('parent', 0, parent)]
    state_root = out / 'states/e20_C_s7_fmamba_fromM16'
    variants += [('C', update, state_root / f'train-{update}.pt') for update in (1500, 3000, 4500, 6000)]
    variants += [(arm, 6000, E.E.T.OUT / f'e20_{arm}_s7_fmamba_fromM16.pt') for arm in ('A', 'B')]
    sources = {str(p): E.R.file_hash(p) for p in (Path(__file__), HERE/'e20_train.py', HERE/'tworld.py', E.D.ROOT/'d4mj/mamba_recurrence.py')}
    records, predictions = [], []
    with torch.no_grad():
        for arm, update, path in variants:
            saved = torch.load(path, map_location='cpu', weights_only=False)
            world.load_state_dict(saved['world'])
            pieces = []
            for start in range(0, len(ids), 8):
                if time.monotonic() - started > 170:
                    raise RuntimeError('Short diagnostic runtime ceiling reached; no long job')
                pieces.append(world(token_values[start:start+8, 11:15], actions[start:start+8, 11:15])[0][:, -1].float())
            pred = torch.cat(pieces)
            dl = (E.read_health(pred[:,63], loss_reader) - E.read_health(token_values[:,14,63], loss_reader)) * 9
            dh = hud(pred, hud_reader) - hud(token_values[:,14], hud_reader)
            row = {'arm': arm, 'update': update, 'checkpoint_sha256': E.R.file_hash(path), 'groups': {}}
            for j, group in enumerate(groups):
                q = slice(j*8, (j+1)*8)
                r = {'n': 8, 'true_delta_mean': float(truth[q].mean())}
                for name, delta in (('loss_reader', dl), ('independent_hud', dh)):
                    r[name] = {'delta_mean': float(delta[q].mean()), 'delta_median': float(delta[q].median()),
                               'change_mae': float((delta[q]-truth[q]).abs().mean()),
                               'drops_gt_1_5': int((delta[q] < -1.5).sum()), 'drops_gt_0_5': int((delta[q] < -.5).sum())}
                row['groups'][group] = r
            records.append(row)
            predictions.append({'arm':arm,'update':update,'loss_delta':dl,'hud_delta':dh})
            print(json.dumps({'arm':arm,'update':update,'groups':row['groups']}), flush=True)
    result = {'scope':'Exploratory fixed 32-target factual TRAIN panel; four groups of eight. Not representative performance or sufficient convergence proof.',
              'window':4,'no_optimization':True,'device':'cpu','numerical_parity':parity_record,'selection_seed':20261007,
              'selected_ids':ids.tolist(),'selected_source_ids':[labels['ids'][int(i)] for i in ids],
              'verified_feature_chunks':verified,'pool_labels_sha256':E.R.file_hash(labels_path),'sources':sources,
              'health_reader_sha256':E.R.file_hash(out/'health_reader.pt'),'hud_reader_sha256':E.R.file_hash(out/'analysis_hud_reader.pt'),
              'records':records,'seconds':time.monotonic()-started}
    E.R.atomic_torch(out/'analysis_checkpoint_curve_cpu_rows.pt', {'ids':ids,'truth':truth,'predictions':predictions})
    E.R.atomic_json(result_path,result,immutable=True)
    print(json.dumps({'stage':'complete','seconds':result['seconds'],'parity':parity_record}),flush=True)


if __name__ == '__main__':
    main()
