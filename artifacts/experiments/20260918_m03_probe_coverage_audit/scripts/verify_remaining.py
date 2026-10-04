import json,time,os,collections
os.environ['JAX_PLATFORMS']='cpu';os.environ['MPLCONFIGDIR']='/tmp/m03-coverage-mpl'
import numpy as np
from artifacts.eda import replay
from d4mj.m03.gate import _state_binary_labels,STATIC_BINARY,_recorded_step_key
q=json.load(open('/tmp/m03_coverage_event_candidates.json'));do=json.load(open('/tmp/m03_coverage_do_candidates.json'));records=[];started=time.time();positives=collections.defaultdict(set)
_,_,_,step,frame=replay.env_and_render()
def check(split,row,reason):
 state=replay.advance_to(row['shard'],row['slot'],row['t']);b=_state_binary_labels(state)
 r={'split':split,**row,'candidate_reason':reason,'positive_labels':[n for n,v in zip(STATIC_BINARY,b) if v]}
 for n in r['positive_labels']:positives[(split,n)].add(row['episode_id'])
 records.append(r)
 return r,state
for split in ['train','dev']:
 for row in q['examples'][f'{split}:10'][:20]:
  check(split,row,'plant_achievement_action')
  if len(positives[(split,'place_plant')])>=10:break
 for episode in do[f'{split}:5']:
  if all(len(positives[(split,n)])>=10 for n in ['front_tree','front_water']):break
  for t in episode['t'][:6]:
   r,_=check(split,dict(episode,t=t),'do_achievement_action')
  if len(records)%20==0:print(json.dumps({'stage':'do_candidates','split':split,'checked':len(records),'tree':len(positives[(split,'front_tree')]),'water':len(positives[(split,'front_water')]),'seconds':time.time()-started}),flush=True)
 print(json.dumps({'stage':'split_complete','split':split,'checked':len(records),'positive_episode_counts':{n:len(positives[(split,n)]) for n in STATIC_BINARY},'seconds':time.time()-started}),flush=True)
json.dump(records,open('/tmp/m03_coverage_extra_verified.json','w'),indent=2)
all_records=json.load(open('/tmp/m03_coverage_verified.json'))+records
# Retain at most one useful root per episode; verifying pixels does not encode/train a model.
wanted={'front_tree','front_water','place_plant','make_wood_pickaxe','make_stone_pickaxe','make_iron_pickaxe','make_wood_sword','make_stone_sword','make_iron_sword','near_table'}
selected=[];seen=set()
for r in all_records:
 if not wanted.intersection(r['positive_labels']) or r['episode_id'] in seen:continue
 seen.add(r['episode_id']);selected.append(r)
for r in selected:
 state=replay.advance_to(r['shard'],r['slot'],r['t']);e=replay.episode_fields(r['shard'],r['slot']);rendered=np.asarray(frame(state));r['root_pixel_max_abs']=int(np.abs(rendered.astype(np.int16)-e['observations'][r['t']].numpy().astype(np.int16)).max())
 _,nxt,_,_,_=step(_recorded_step_key(replay,r),state,int(e['actions_taken'][r['t']]))
 r['factual_pixel_max_abs']=int(np.abs(np.asarray(frame(nxt)).astype(np.int16)-e['observations'][r['t']+1].numpy().astype(np.int16)).max())
 assert r['root_pixel_max_abs']==r['factual_pixel_max_abs']==0
json.dump(selected,open('/tmp/m03_coverage_replay_verified_selected.json','w'),indent=2)
print(json.dumps({'stage':'pixel_check_complete','roots':len(selected),'root_pixel_max_abs':max(r['root_pixel_max_abs'] for r in selected),'factual_pixel_max_abs':max(r['factual_pixel_max_abs'] for r in selected),'seconds':time.time()-started}),flush=True)
