import json,time,os,collections
os.environ['JAX_PLATFORMS']='cpu';os.environ['MPLCONFIGDIR']='/tmp/m03-coverage-mpl'
import numpy as np
from artifacts.eda import replay
from d4mj.m03.gate import _state_binary_labels,STATIC_BINARY,_recorded_step_key
records=json.load(open('/tmp/m03_coverage_verified.json'))+json.load(open('/tmp/m03_coverage_extra_verified.json'));q=json.load(open('/tmp/m03_coverage_do_expanded.json'));seen={(r['episode_id'],r['t']) for r in records};counts=collections.defaultdict(set);started=time.time();new=[]
for r in records:
 for n in r['positive_labels']:counts[(r['split'],n)].add(r['episode_id'])
for split in ['train','dev']:
 for episode in q[split]:
  if all(len(counts[(split,n)])>=10 for n in ['front_tree','front_ripe_plant']):break
  for t in episode['t'][:6]:
   if (episode['episode_id'],t) in seen:continue
   row=dict(episode,t=t);state=replay.advance_to(row['shard'],row['slot'],t);b=_state_binary_labels(state);r={'split':split,**row,'candidate_reason':'expanded_do_achievement_action','positive_labels':[n for n,v in zip(STATIC_BINARY,b) if v]};records.append(r);new.append(r)
   for n in r['positive_labels']:counts[(split,n)].add(r['episode_id'])
  if len(new)%100<6:print(json.dumps({'stage':'expanded','split':split,'new_roots':len(new),'tree':len(counts[(split,'front_tree')]),'ripe':len(counts[(split,'front_ripe_plant')]),'seconds':time.time()-started}),flush=True)
 print(json.dumps({'stage':'split_complete','split':split,'counts':{n:len(counts[(split,n)]) for n in STATIC_BINARY},'new_roots':len(new),'seconds':time.time()-started}),flush=True)
json.dump(records,open('/tmp/m03_coverage_all_verified.json','w'),indent=2)
# Greedily retain ten positive episodes per missing static label, then check exact pixels and successors.
targets=['make_iron_pickaxe','make_iron_sword','make_stone_pickaxe','make_stone_sword','make_wood_pickaxe','make_wood_sword','front_ripe_plant','front_water','front_tree','near_table','place_plant'];selected={};pos=collections.defaultdict(set)
for split in ['train','dev']:
 for name in targets:
  for r in records:
   if r['split']!=split or name not in r['positive_labels']:continue
   if len(pos[(split,name)])>=10:break
   selected[(r['episode_id'],r['t'])]=r
   for n in r['positive_labels']:pos[(split,n)].add(r['episode_id'])
_,_,_,step,frame=replay.env_and_render()
for r in selected.values():
 state=replay.advance_to(r['shard'],r['slot'],r['t']);e=replay.episode_fields(r['shard'],r['slot']);key=_recorded_step_key(replay,r);r['root_pixel_max_abs']=int(np.abs(np.asarray(frame(state)).astype(np.int16)-e['observations'][r['t']].numpy().astype(np.int16)).max());_,nxt,_,_,_=step(key,state,int(e['actions_taken'][r['t']]));r['factual_pixel_max_abs']=int(np.abs(np.asarray(frame(nxt)).astype(np.int16)-e['observations'][r['t']+1].numpy().astype(np.int16)).max());assert r['root_pixel_max_abs']==r['factual_pixel_max_abs']==0
 bs=[]
 for action in range(17):
  _,nxt,_,_,_=step(key,state,action);bs.append(_state_binary_labels(nxt))
 r['successor_positive_labels']=[n for n,v in zip(STATIC_BINARY,np.stack(bs).any(0)) if v]
json.dump(list(selected.values()),open('/tmp/m03_coverage_final_selected.json','w'),indent=2)
print(json.dumps({'stage':'integrity_and_successors_complete','roots':len(selected),'root_pixel_max_abs':0,'factual_pixel_max_abs':0,'seconds':time.time()-started}),flush=True)
