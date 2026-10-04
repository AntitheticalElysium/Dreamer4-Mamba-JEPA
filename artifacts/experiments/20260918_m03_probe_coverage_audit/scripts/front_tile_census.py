import os,json,time,collections,functools
os.environ['JAX_PLATFORMS']='cpu';os.environ['MPLCONFIGDIR']='/tmp/m03-coverage-mpl'
import numpy as np
import torch,jax,jax.numpy as jnp
from pathlib import Path
from artifacts.eda import replay
from craftax.craftax_classic.constants import DIRECTIONS,BlockType
from d4mj.m03.gate import _state_binary_labels,STATIC_BINARY,_recorded_step_key
started=time.time();torch.set_num_threads(1);env,params,reset,step,frame=replay.env_and_render()
@jax.jit
def scan(reset_key,keys,actions):
 _,state=env.reset(reset_key,params)
 def body(state,xs):
  key,action=xs;front=state.player_position+DIRECTIONS[state.player_direction];inside=jnp.all((front>=0)&(front<jnp.array(state.map.shape)));tile=jnp.where(inside,state.map[front[0],front[1]],BlockType.OUT_OF_BOUNDS.value)
  _,nxt,_,_,_=env.step(key,state,action,params)
  return nxt,tile
 return jax.lax.scan(body,state,(keys,actions))[1]
counts=collections.Counter();eligible=collections.Counter();candidates=collections.defaultdict(list);targets={'front_lava':int(BlockType.LAVA.value),'front_ripe_plant':int(BlockType.RIPE_PLANT.value),'front_water':int(BlockType.WATER.value),'front_tree':int(BlockType.TREE.value)};checked=collections.Counter()
for split in ['dev','train']:
 for si,record in enumerate(replay.manifest()['shards']):
  d=torch.load(replay.STORE/record['file'],mmap=True,weights_only=False)
  for slot,e in enumerate(d['episodes']):
   if e['split']!=split:continue
   n=len(e['actions_taken'])
   if n<64:continue
   checked[split]+=1;rk,keys=replay._slot_keys(si,slot);bucket=next(b for b in replay.BUCKETS if b>=n);actions=jnp.zeros(bucket,dtype=jnp.int32).at[:n].set(e['actions_taken'].numpy());tiles=np.asarray(scan(rk,keys[:bucket],actions))[63:n]
   for name,tile in targets.items():
    idx=np.flatnonzero(tiles==tile)
    if len(idx):
     counts[(split,name)]+=1
     if len(candidates[(split,name)])<30:candidates[(split,name)].append({'split':split,'shard':si,'slot':slot,'t':int(idx[0]+63),'episode_id':e['episode_id'],'candidate_reason':'all_logged_front_tile_census','target':name})
   if checked[split]%100==0:print(json.dumps({'stage':'census','split':split,'episodes':checked[split],'positives':{k:counts[(split,k)] for k in targets},'seconds':time.time()-started}),flush=True)
   if split=='train' and all(counts[(split,k)]>=10 for k in targets):break
  if split=='train' and all(counts[(split,k)]>=10 for k in targets):break
 print(json.dumps({'stage':'split_complete','split':split,'episodes':checked[split],'positives':{k:counts[(split,k)] for k in targets},'seconds':time.time()-started}),flush=True)
result={'scope':'DEV all eligible episodes, all logged pre-action times t>=63; TRAIN until ten positive episodes per front-tile target','minimum_context_frames':64,'checked_episodes':dict(checked),'positive_episode_counts':{f'{sp}:{k}':counts[(sp,k)] for sp in ['train','dev'] for k in targets},'candidates':{f'{sp}:{k}':v for (sp,k),v in candidates.items()}}
json.dump(result,open('/tmp/m03_front_census.json','w'),indent=2)
verified=[]
for (split,target),rows in candidates.items():
 if target not in ['front_lava','front_ripe_plant']:continue
 for row in rows[:10]:
  state=replay.advance_to(row['shard'],row['slot'],row['t']);b=_state_binary_labels(state);assert b[STATIC_BINARY.index(target)];e=replay.episode_fields(row['shard'],row['slot']);r=dict(row,positive_labels=[n for n,v in zip(STATIC_BINARY,b) if v]);r['root_pixel_max_abs']=int(np.abs(np.asarray(frame(state)).astype(np.int16)-e['observations'][row['t']].numpy().astype(np.int16)).max());key=_recorded_step_key(replay,row);_,nxt,_,_,_=step(key,state,int(e['actions_taken'][row['t']]));r['factual_pixel_max_abs']=int(np.abs(np.asarray(frame(nxt)).astype(np.int16)-e['observations'][row['t']+1].numpy().astype(np.int16)).max());assert r['root_pixel_max_abs']==r['factual_pixel_max_abs']==0
  bs=[]
  for action in range(17):
   _,nxt,_,_,_=step(key,state,action);bs.append(_state_binary_labels(nxt))
  r['successor_positive_labels']=[n for n,v in zip(STATIC_BINARY,np.stack(bs).any(0)) if v];verified.append(r)
json.dump(verified,open('/tmp/m03_front_census_verified.json','w'),indent=2)
print(json.dumps({'stage':'complete','verified_roots':len(verified),'seconds':time.time()-started}),flush=True)
