"""Model-free candidate indexing. Events are candidates, never certified prerequisites."""
import json,collections,time
from pathlib import Path
import torch
torch.set_num_threads(1);store=Path('artifacts/craftax_support_v2');m=json.loads((store/'manifest.json').read_text());counts=collections.Counter();examples=collections.defaultdict(list);do=collections.defaultdict(list);started=time.time()
for si,record in enumerate(m['shards']):
 payload=torch.load(store/record['file'],mmap=True,weights_only=False)
 for slot,e in enumerate(payload['episodes']):
  sp=e['split']
  if sp=='final':continue
  a=e['actions_taken'];n=len(a);positions=torch.arange(n);event=e['events']&(e['rewards']>0)
  for action in range(7,17):
   for eligible in [False,True]:
    idx=torch.where((a==action)&event&((positions>=63) if eligible else torch.ones(n,dtype=torch.bool)))[0]
    if len(idx):
     counts[(sp,action,'eligible' if eligible else 'all')]+=1
     if eligible and len(examples[(sp,action)])<30:examples[(sp,action)].append({'shard':si,'slot':slot,'t':int(idx[0]),'episode_id':e['episode_id']})
  idx=torch.where((a==5)&event&(positions>=63))[0]
  if len(idx) and len(do[sp])<700:do[sp].append({'shard':si,'slot':slot,'t':idx.tolist(),'episode_id':e['episode_id']})
q={'counts':[{'split':sp,'action':action,'scope':scope,'episodes':v} for (sp,action,scope),v in counts.items()],'examples':{f'{sp}:{a}':v for (sp,a),v in examples.items()}}
Path('/tmp/m03_coverage_event_candidates.json').write_text(json.dumps(q,indent=2));Path('/tmp/m03_coverage_do_candidates.json').write_text(json.dumps({f'{sp}:5':v[:120] for sp,v in do.items()},indent=2));Path('/tmp/m03_coverage_do_expanded.json').write_text(json.dumps(dict(do),indent=2));print(json.dumps({'seconds':time.time()-started,'candidate_episode_counts':q['counts']}))
