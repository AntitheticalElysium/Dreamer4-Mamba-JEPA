"""Fail closed before future reboot relaunch: source/parent/data identity and decodable full working states."""
import datetime
import hashlib
import json
from pathlib import Path

import torch

HERE=Path(__file__).parent
W=Path('artifacts/eda/levers_tworlds_v1')
m=json.loads((HERE/'inputs.json').read_text())

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):
            h.update(b)
    return h.hexdigest()

# The working canvas state and lane log legitimately advance. Their historical copies/pins stay separate.
static={p:v for p,v in m['pins'].items() if p.endswith('.py') or
        ('recovery_inputs' in p) or ('levers_tworlds_v1/' in p and '/state/' not in p)}
for p,v in static.items():
    assert sha(p)==v['sha256'], f'Source/parent drift: {p}; do not silently reseal a running experiment'
for p,v in m['datasets'].items():
    print(json.dumps({'verifying_dataset':p}),flush=True)
    assert Path(p).stat().st_size==v['bytes'] and sha(p)==v['sha256'], f'Dataset drift: {p}'

checked=[]
for kind,target,frames in [('fcanvas',36000,6),('fmamba',6000,16)]:
    for seed in [7,8]:
        name=f'corrt_raw_teacher_s{seed}_fcanvas_u36000' if kind=='fcanvas' else f'corrt_rawlong_teacher_s{seed}_fmamba_L16b40_from36000'
        state=W/'state'/f'{name}.state.pt'
        final=W/f'{name}.pt'
        if final.exists():
            f=torch.load(final,map_location='cpu',weights_only=False)
            assert f['name']==name and f['args']['backbone']==kind and int(f['args']['seed'])==seed
            assert f['world']['time'].shape[0]==frames and f['history'][-1]['update']==target
            checked.append({'name':name,'final_update':target,'sha256':sha(final)})
        if state.exists():
            s=torch.load(state,map_location='cpu',weights_only=False)
            assert 0<int(s['update'])<=target and s['world']['time'].shape[0]==frames
            assert s['optimizer'] is not None and all(k in s for k in ['order','rng_cpu','rng_cuda','history'])
            assert len(s['rng_cuda'])==1 and all(torch.isfinite(v).all() for v in s['world'].values())
            checked.append({'name':name,'working_update':int(s['update']),'sha256':sha(state)})
out={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'static_pins_checked':len(static),
     'datasets_checked':len(m['datasets']),'states':checked,'status':'verified'}
tmp=HERE/'restart_checks.tmp'
tmp.write_text(json.dumps(out,indent=2)+'\n')
tmp.replace(HERE/'restart_checks.json')
with (HERE/'restart_checks.jsonl').open('a') as f:
    f.write(json.dumps(out)+'\n')
print(json.dumps(out),flush=True)
