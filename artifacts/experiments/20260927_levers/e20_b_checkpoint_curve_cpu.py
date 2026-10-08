"""Short B-only checkpoint diagnostic on the previous fixed factual TRAIN panel.

No optimization, CUDA, new data collection, or notebook writes. Preserve all
previous evidence. Changes of context are reported separately, never pooled.
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
import e20_checkpoint_curve_cpu as Q
E = Q.E


def main():
    torch.set_num_threads(3)
    out = E.D.OUT
    destination = out / 'analysis_B_checkpoint_curve_cpu.json'
    if destination.exists():
        raise RuntimeError('Preserve previous B diagnostic')
    previous_path = out / 'analysis_checkpoint_curve_cpu.json'
    previous = json.loads(previous_path.read_text())
    assert max(previous['numerical_parity']['hud_health_max_abs'], previous['numerical_parity']['loss_health_max_abs']) < .1
    labels = torch.load(out/'pool/labels.pt', map_location='cpu', weights_only=False)
    assert E.R.file_hash(out/'pool/labels.pt') == previous['pool_labels_sha256']
    ids = torch.tensor(previous['selected_ids'][8:24])
    spec = json.loads((out/'pool/contract.json').read_text())
    mm = np.memmap(out/'pool/tokens.f16', dtype=np.float16, mode='r', shape=tuple(spec['shape']))
    for chunk in previous['verified_feature_chunks']:
        a, b = chunk['start'], chunk['stop']
        if ((ids >= a) & (ids < b)).any():
            assert hashlib.sha256(memoryview(mm[a:b]).cast('B')).hexdigest() == chunk['sha256']
    s = torch.from_numpy(np.array(mm[ids.numpy()])).float()
    acts = labels['actions'][ids]
    truth = labels['dh'][ids, -1].float()
    assert (truth == -2).all()
    loss_reader = torch.load(out/'health_reader.pt', map_location='cpu', weights_only=False)
    hud_reader = torch.load(out/'analysis_hud_reader.pt', map_location='cpu', weights_only=False)
    world, parent = E.make_world(7, 'cpu')
    world.eval()
    for module in world.modules():
        if isinstance(module, Q.FunctionalMamba2):
            module.settings = dataclasses.replace(module.settings, backend='reference')
    root = out/'states/e20_B_s7_fmamba_fromM16'
    variants = [(0, parent)] + [(u, root/f'train-{u}.pt') for u in (1500, 3000, 4500, 6000)]
    rows, raw = [], []
    started = time.monotonic()
    with torch.no_grad():
        for update, path in variants:
            world.load_state_dict(torch.load(path, map_location='cpu', weights_only=False)['world'])
            for length in (4, 15):
                if length == 15 and update not in (0, 6000):
                    continue
                if time.monotonic()-started > 120:
                    raise RuntimeError('Short-diagnostic time limit')
                p = torch.cat([world(s[i:i+8, 15-length:15], acts[i:i+8, 15-length:15])[0][:,-1].float() for i in (0,8)])
                dl = (E.read_health(p[:,63], loss_reader)-E.read_health(s[:,14,63], loss_reader))*9
                dh = Q.hud(p,hud_reader)-Q.hud(s[:,14],hud_reader)
                row = {'update':update,'window':length,'checkpoint_sha256':E.R.file_hash(path),'groups':{}}
                for j, group in enumerate(('ordinary_nonfresh','fresh_ordinary')):
                    q = slice(j*8,(j+1)*8)
                    row['groups'][group] = {name:{'n':8,'delta_mean':float(delta[q].mean()),
                        'change_mae':float((delta[q]-truth[q]).abs().mean()),'strict_hits':int((delta[q]<-1.5).sum()),
                        'sensitivity_hits':int((delta[q]<-.5).sum())} for name,delta in (('loss_reader',dl),('independent_hud',dh))}
                rows.append(row);raw.append({'update':update,'window':length,'loss_delta':dl,'hud_delta':dh})
                print(json.dumps(row),flush=True)
    report = {'scope':'Post-hoc fixed factual TRAIN micro-panel; 8 ordinary nonfresh and 8 detector-estimated fresh hits. Not a representative convergence certificate.',
              'no_optimization':True,'device':'cpu/reference FP32','previous_diagnostic_sha256':E.R.file_hash(previous_path),
              'previous_cuda_parity':previous['numerical_parity'],'ids':ids.tolist(),
              'sources':{str(p):E.R.file_hash(p) for p in (Path(__file__),Path(Q.__file__),HERE/'e20_train.py',HERE/'tworld.py')},
              'rows':rows,'seconds':time.monotonic()-started}
    E.R.atomic_torch(out/'analysis_B_checkpoint_curve_cpu_rows.pt',{'ids':ids,'truth':truth,'predictions':raw})
    E.R.atomic_json(destination,report,immutable=True)
    print(json.dumps({'stage':'complete','seconds':report['seconds']}),flush=True)


if __name__=='__main__':main()
