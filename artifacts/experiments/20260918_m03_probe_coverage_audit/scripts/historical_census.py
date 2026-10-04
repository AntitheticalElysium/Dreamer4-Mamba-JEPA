"""Read-only census of cached historical labels; use counts for triage, not as fresh gate labels."""
import torch,json,sqlite3,hashlib,io,collections,time
from pathlib import Path
from d4mj.m03.gate import STATIC_BINARY
from d4mj.m03.history import seed_split
torch.set_num_threads(1);base=Path('artifacts/lewm_gates_20260906/m03_bootstrap/evaluation_v2/historical/features');db=sqlite3.connect('file:artifacts/lewm_gates_20260906/cache.sqlite3?mode=ro',uri=True);counts=collections.defaultdict(set);addresses=collections.defaultdict(list);n=0
for p in sorted(base.glob('replay.*.pt')):
 wrapper=torch.load(p,weights_only=False,mmap=True);ref=wrapper['cache_ref'];checksum,data,path,selector=db.execute('SELECT sha256,payload,external,selector FROM nodes WHERE key=?',(ref['key'],)).fetchone()
 if path:value=torch.load(path,map_location='cpu',weights_only=False,mmap=True)
 else:
  assert hashlib.sha256(data).hexdigest()==checksum
  value=torch.load(io.BytesIO(data),map_location='cpu',weights_only=False)
 if selector:value=value[selector]
 if 'features' in value:value=value['features']
 for row,seed in enumerate(value['episode'].tolist()):
  split=seed_split(seed)
  if split not in ['fit','test']:continue
  for j,label in enumerate(STATIC_BINARY):
   if bool(value['root_binary'][row,j]):
    counts[(split,label)].add(seed)
    if len(addresses[(split,label)])<30:addresses[(split,label)].append({'seed':seed,'t':int(value['time'][row])})
 n+=len(value['episode'])
json.dump({'rows':n,'positive_seed_counts':{f'{sp}:{label}':len(v) for (sp,label),v in counts.items()},'example_addresses':{f'{sp}:{label}':v for (sp,label),v in addresses.items()}},open('/tmp/m03_coverage_historical.json','w'),indent=2);db.close()
