"""Archive recovery state and pin inputs before research relaunch. Read-only on historical evidence."""
import datetime
import hashlib
import json
import shutil
from pathlib import Path

ROOT=Path.cwd()
HERE=Path(__file__).parent
W=Path('artifacts/eda/levers_tworlds_v1')

def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):
            h.update(b)
    return h.hexdigest()

assert not (HERE/'inputs.json').exists(), 'Manifest already exists; never reseal in place'
archive=HERE/'recovery_inputs'
archive.mkdir(exist_ok=True)
st=W/'state/corrt_raw_teacher_s7_fcanvas_u36000.state.pt'
saved=archive/'canvas_s7_u13000.state.pt'
shutil.copy2(st,saved)
paths=[st,saved]+list(Path('d4mj').rglob('*.py'))
for folder in ['artifacts/experiments/20260927_levers','artifacts/experiments/20260921_readout_ladder']:
    paths += list(Path(folder).glob('*.py'))
for s in [7,8]:
    paths += [W/f'corrt_raw_teacher_s{s}_u36000.pt',W/f'corrt_raw_teacher_s{s}_fmamba_u36000.pt',
              W/f'corrt_rawlong_teacher_s{s}_L16b40_from36000.pt']
paths += [Path('artifacts/eda/levers_logs/lanes.log'),
          Path('artifacts/eda/deepeval_v1/h16traj_corrt_raw_teacher_s8_fmamba_u36000_fit.f16')]
pins={str(p):{'sha256':sha(p),'bytes':p.stat().st_size} for p in paths if p.exists()}
datasets={}
for folder,manifest,files in [('artifacts/eda/spatial_pool_v1','pool.json',[('pool.pt','pool_sha256')]),
                             ('artifacts/eda/levers_mamba_long_pools_v1/raw','manifest.json',
                              [('labels.pt','labels_sha256'),('tokens.f16','tokens_sha256')])]:
    m=json.loads((Path(folder)/manifest).read_text())
    for file,key in files:
        p=Path(folder)/file
        print(json.dumps({'hashing':str(p),'bytes':p.stat().st_size}),flush=True)
        got=sha(p)
        assert got==m[key], (str(p),got,m[key])
        datasets[str(p)]={'sha256':got,'bytes':p.stat().st_size,'historical_manifest_match':True}
out={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'pins':pins,'datasets':datasets,
     'partial_legacy_cache':'Unjournalled, not reused. Preserved in place.',
     'checkpoint_interval':1000,'canvas_resume_update':13000,
     'm16_resume':'No full-run state; own 36k parents, not 500-update smoke'}
tmp=HERE/'inputs.tmp'
tmp.write_text(json.dumps(out,indent=2)+'\n')
tmp.replace(HERE/'inputs.json')
print(json.dumps({'status':'inputs_verified','files':len(pins),'datasets':len(datasets)}),flush=True)
