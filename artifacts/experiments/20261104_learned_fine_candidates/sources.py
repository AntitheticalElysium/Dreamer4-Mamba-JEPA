"""Exact simulator qualification, before native consequence inference."""
import os
os.environ.setdefault('JAX_PLATFORMS','cpu')
import json
from pathlib import Path
import numpy as np
import jax.numpy as jnp
import torch
from saccade import data as D
from craftax.craftax_classic.constants import BlockType
HERE=Path(__file__).resolve().parent

def cell_of(world,player):
    r,c=np.asarray(world)-np.asarray(player)+np.array([3,4]);return int(r*9+c) if 0<=r<7 and 0<=c<9 else -1
def occupants(s,world):
    return any(bool((np.asarray(getattr(s,k).mask)&(np.asarray(getattr(s,k).position)==world).all(-1)).any()) for k in ['zombies','cows','skeletons','arrows'])

def main():
    chosen=json.loads((HERE/'far_catalogue.json').read_text())['selected'];_,_,_,step,frame=D.RP.env_and_render()
    out=[];rejected=[]
    for i,r in enumerate(chosen):
        st=D.RP.advance_to(r['shard'],r['slot'],r['t']);raw=D.RP.episode_fields(r['shard'],r['slot'])['observations'];actual=np.asarray(frame(st))
        assert np.array_equal(actual,raw[r['t']].numpy())
        world=np.asarray(st.player_position)+np.array(divmod(r['cell'],9))-np.array([3,4])
        if r['family']=='mob':
            ent=getattr(st,r['kind']);ent=ent.replace(mask=ent.mask.at[r['j']].set(False),health=ent.health.at[r['j']].set(0))
            absent=st.replace(**{r['kind']:ent},mob_map=st.mob_map.at[world[0],world[1]].set(False))
        else:absent=st.replace(map=st.map.at[world[0],world[1]].set(BlockType.GRASS.value))
        key=D.RP._slot_keys(r['shard'],r['slot'])[1][r['t']];states=[];futures=[]
        for s in [st,absent]:
            _,ns,reward,term,_=step(key,s,jnp.asarray(r['action'],jnp.int32));states.append(ns);futures.append(np.asarray(frame(ns)))
        assert np.array_equal(futures[0],raw[r['t']+1].numpy())
        assert np.array_equal(np.asarray(states[0].player_position),np.asarray(states[1].player_position))
        if r['family']=='mob':
            region=[cell_of(world+d,states[0].player_position) for d in np.array([[0,0],[-1,0],[1,0],[0,-1],[0,1]])]
            assert min(region)>=0
            events=[]
            for s in states:
                ent=getattr(s,r['kind']);events.append(any(bool((np.asarray(ent.mask)&(np.asarray(ent.position)==world+d).all(-1)).any()) for d in np.array([[0,0],[-1,0],[1,0],[0,-1],[0,1]])))
        else:
            region=[cell_of(world,states[0].player_position)];assert min(region)>=0
            events=[int(s.map[world[0],world[1]])==r['ky'] and not occupants(s,world) for s in states]
        if events!=[True,False]:
            rejected.append(dict(meta=r,reason='selected visible semantic event does not differ',events=events));continue
        out.append(dict(meta=dict(**r,target_cells=region,world=world.tolist()),current=np.stack([actual,np.asarray(frame(absent))]),future=np.stack(futures)))
        print('qualified',i+1,'/',len(chosen),r['family'],r['kind'],'missed',r['missed'],flush=True)
    torch.save(dict(rows=out,rejected=rejected),HERE/'sources.pt')
    (HERE/'source_qualification.json').write_text(json.dumps(dict(accepted=[r['meta'] for r in out],rejected=rejected),indent=2)+'\n')

if __name__=='__main__':main()
