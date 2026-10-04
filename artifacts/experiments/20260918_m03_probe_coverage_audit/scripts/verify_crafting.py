"""Certify candidate crafting prerequisites with the current M03 label function."""
import json,time,os
os.environ['JAX_PLATFORMS']='cpu';os.environ['MPLCONFIGDIR']='/tmp/m03-coverage-mpl'
from artifacts.eda import replay
from d4mj.m03.gate import _state_binary_labels,STATIC_BINARY
q=json.load(open('/tmp/m03_coverage_event_candidates.json'));records=[];started=time.time()
for split in ['train','dev']:
 for action in [12,13]:
  positive=0
  for row in q['examples'][f'{split}:{action}'][:12]:
   state=replay.advance_to(row['shard'],row['slot'],row['t']);labels=_state_binary_labels(state);label='make_stone_pickaxe' if action==12 else 'make_iron_pickaxe';positive+=bool(labels[STATIC_BINARY.index(label)]);records.append({'split':split,'action':action,**row,'positive_labels':[n for n,v in zip(STATIC_BINARY,labels) if v]})
   if positive>=10:break
  print(json.dumps({'split':split,'action':action,'positive_episodes':positive,'seconds':time.time()-started}),flush=True)
json.dump(records,open('/tmp/m03_coverage_verified.json','w'),indent=2)
