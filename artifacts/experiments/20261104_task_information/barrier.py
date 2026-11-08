"""Source-driven affordance check for the strongest confirming observer contrast.

Episode307,t161: DOWN is stone, LEFT is unmineable iron with current tools.
The queried zombie is down-left. DOWN,DO opens the mineable barrier. The fixed
plan is declared from source geometry before these consequence outputs.
"""
import os
os.environ.setdefault('JAX_PLATFORMS','cpu')
import json
from pathlib import Path
import numpy as np
import torch
import jax,jax.numpy as jnp
from saccade import data as D
HERE=Path(__file__).resolve().parent

def main():
    r=next(r for r in json.loads((HERE/'paired_readers_strong.json').read_text()) if r['meta']['episode']==307);meta=r['meta']
    assert r['qualified'];env,params,_,_,frame=D.RP.env_and_render();st=D.RP.advance_to(meta['shard'],meta['slot'],meta['time'])
    assert np.array_equal(np.asarray(frame(st)),D.RP.episode_fields(meta['shard'],meta['slot'])['observations'][meta['time']].numpy())
    below=st.player_position+jnp.array([1,0]);assert int(st.map[below[0],below[1]])==4 and int(st.inventory.wood_pickaxe)>0 and int(st.inventory.stone_pickaxe)==0
    j=meta['j'];ent=st.zombies.replace(mask=st.zombies.mask.at[j].set(False),health=st.zombies.health.at[j].set(0));world=st.zombies.position[j]
    other=st.replace(zombies=ent,mob_map=st.mob_map.at[world[0],world[1]].set(False))
    acts=jnp.array([4,5,4,0,0,0,0,0,0],jnp.int32)
    def one(state,keys):
        def body(carry,ka):
            s,dead=carry
            def alive(s):
                _,n,r,d,_=env.step(ka[0],s,ka[1],params);return n,r,d
            n,r,d=jax.lax.cond(dead,lambda s:(s,jnp.asarray(0.,jnp.float32),jnp.asarray(True)),alive,s)
            obs=jnp.array([s.player_health-n.player_health,r,d,n.player_health,n.player_position[0],n.player_position[1],
                n.inventory.stone,n.map[below[0],below[1]],n.zombies.position[j,0],n.zombies.position[j,1]],jnp.float32)
            return (n,dead|d),obs
        return jax.lax.scan(body,(state,jnp.asarray(False)),(keys,acts))[1]
    run=jax.jit(jax.vmap(one,in_axes=(None,0)));keys=jax.random.split(jax.random.PRNGKey(619307),32*len(acts)).reshape(32,len(acts),2)
    v=np.stack([np.asarray(run(s,keys)) for s in [st,other]])
    # This verifies the actual affordance the plan was meant to test.
    assert (v[:,:,0,7]==4).all() and (v[:,:,1,7]==7).all()
    out=dict(meta=meta,actions=np.asarray(acts),values=v,columns=['health_decrement','environment_reward','terminated','health','player_row','player_col','stone_inventory','barrier_tile','queried_zombie_row','queried_zombie_col'])
    torch.save(out,HERE/'barrier.pt')
    summary=dict(episode=307,time=161,keys=32,actions=np.asarray(acts).tolist(),barrier_opened_both_worlds=True,
        health_decrements=v[:,:,:,0].clip(min=0).sum(-1).mean(-1).tolist(),reward=v[:,:,:,1].sum(-1).mean(-1).tolist(),
        death=v[:,:,:,2].max(-1).mean(-1).tolist())
    (HERE/'barrier.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
