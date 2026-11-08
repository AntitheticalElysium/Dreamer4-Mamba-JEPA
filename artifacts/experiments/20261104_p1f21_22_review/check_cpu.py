"""Read-only CPU checks of the two in-preparation arms. Never runs their workers."""
import os
os.environ.setdefault('JAX_PLATFORMS','cpu')
os.environ.setdefault('OMP_NUM_THREADS','1')
import copy, hashlib, importlib, json
from pathlib import Path
import torch
from saccade import model as M, train as TR

HERE=Path(__file__).resolve().parent
A=importlib.import_module('artifacts.experiments.20261104_p1f21.train_p1f21')
B=importlib.import_module('artifacts.experiments.20261104_p1f22.train_p1f22')
BASE=importlib.import_module('artifacts.experiments.20261103_p1f2.train_p1f2')
torch.set_num_threads(1)
base=BASE.recipe(copy.deepcopy(TR.load_cfg()));a=A.recipe(copy.deepcopy(TR.load_cfg()));b=B.recipe(copy.deepcopy(TR.load_cfg()))
def diff(x,y,p=''):
 out={}
 for k in x.keys()|y.keys():
  q=p+'.'+k if p else k
  if isinstance(x.get(k),dict) and isinstance(y.get(k),dict):out.update(diff(x[k],y[k],q))
  elif x.get(k)!=y.get(k):out[q]=[x.get(k),y.get(k)]
 return out
result={'config_differences':{'diversity':diff(base,a),'acuity':diff(base,b)}}
assert set(result['config_differences']['diversity'])=={'data.corpus','p1.out_dir'}
assert set(result['config_differences']['acuity'])=={'perception.periphery.acuity'}
models=[]
for cfg in [base,a,b]:
 torch.manual_seed(1);m=M.M0(cfg);models.append((m.state_dict(),torch.get_rng_state().clone()))
ref=models[0][0]
for name,(state,rng) in zip(['diversity','acuity'],models[1:]):
 shared=[k for k in ref if k in state and ref[k].shape==state[k].shape]
 changed=[k for k in shared if not torch.equal(ref[k],state[k])]
 result[name+'_initialization']={'equal_shape_tensors':len(shared),'changed_tensors':len(changed),
  'equal_shape_elements':sum(ref[k].numel() for k in shared),'changed_elements':sum(ref[k].numel() for k in changed),
  'shape_changes':{k:[list(ref[k].shape),list(state[k].shape)] for k in ref if k in state and ref[k].shape!=state[k].shape},
  'post_initialization_rng_equal':torch.equal(models[0][1],rng),'first_changed_tensors':changed[:12]}
 if name=='diversity':assert not changed and torch.equal(models[0][1],rng)
# Check dependency closure without writing either arm's SOURCE_PINS.json.
start=set(json.loads((HERE.parent/'20261103_p1f2/SOURCE_PINS.json').read_text())['files'])
sealed=start|{str(p.relative_to(TR.ROOT)) for p in (HERE.parent/'20261104_p1f22').glob('*.py')}
required={str((HERE.parent/'20261104_p1f21'/f).relative_to(TR.ROOT)) for f in ['run.py','checks.py','pin_sources.py']}
result['acuity_shared_dependencies_not_in_seal']=sorted(required-sealed)
# Existing raw union manifest, without changing shared replay globals or reading images.
u=TR.ROOT/'artifacts/craftax_union_v1/manifest.json'
if u.exists():
 m=json.loads(u.read_text());offset=0;source_checks=[]
 for s in m['sources']:
  sm=json.loads((TR.ROOT/s['store']/'manifest.json').read_text())
  assert hashlib.sha256((TR.ROOT/s['store']/'manifest.json').read_bytes()).hexdigest()==s['manifest_sha256']
  block=m['shards'][offset:offset+s['shards']]
  assert len(block)==len(sm['shards'])
  for x,y in zip(block,sm['shards']):
   assert (u.parent/x['file']).resolve()==(TR.ROOT/s['store']/y['file']).resolve()
   assert x['sha256']==y['sha256']
  source_checks.append({'source':s['store'],'shards':len(block),'order_paths_and_declared_hashes_match':True});offset+=s['shards']
 assert offset==len(m['shards'])==1060
 result['union_manifest']={'episodes':m['episodes'],'transitions':m['transitions'],'sources':source_checks}
result['source_pins']={str(p.relative_to(TR.ROOT)):TR.file_sha256(p) for folder in ['20261104_p1f21','20261104_p1f22','20261104_f2_train_dev'] for p in (HERE.parent/folder).glob('*.py')}
(HERE/'CPU_CHECKS.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
