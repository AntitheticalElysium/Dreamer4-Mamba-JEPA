"""Matched environment branches establish whether the observed fact matters.

The environment is used as a causal label generator, not as a learned observer
or as an actor. Both worlds receive identical action plans and random keys.
"""
import os
os.environ.setdefault('JAX_PLATFORMS','cpu')
import json,sys
from pathlib import Path
import numpy as np
import torch
import jax,jax.numpy as jnp
from saccade import data as D
HERE=Path(__file__).resolve().parent
KINDS={1:'zombies',2:'cows',3:'skeletons',4:'arrows'}

def visible_kind(st,world):
    ky=0
    for k,name in KINDS.items():
        ent=getattr(st,name)
        if bool((np.asarray(ent.mask)&(np.asarray(ent.position)==world).all(-1)).any()):ky=k
    return ky

def main():
    focused='--focus' in sys.argv
    balanced='--balanced-coarse' in sys.argv
    assert not balanced or focused
    source='focus_balanced_results.json' if balanced else ('focus_local_results.json' if focused else 'selection.json')
    selected=json.loads((HERE/source).read_text())['selected'];env,params,_,_,frame=D.RP.env_and_render()
    output='balanced_consequences.pt' if balanced else ('focus_consequences.pt' if focused else 'consequences.pt')
    def rollout(state,keys,action):
        def run(carry,key):
            st,dead,damage,reward=carry
            def alive(st):
                _,ns,r,d,_=env.step(key,st,action,params)
                return ns,r,d
            ns,r,d=jax.lax.cond(dead,lambda s:(s,jnp.asarray(0.,jnp.float32),jnp.asarray(True)),alive,st)
            injury=jnp.maximum(0,st.player_health-ns.player_health)
            return (ns,dead|d,damage+injury,reward+r),None
        (ns,dead,damage,reward),_=jax.lax.scan(run,(state,jnp.asarray(False),jnp.asarray(0,jnp.int32),jnp.asarray(0.,jnp.float32)),keys)
        return jnp.array([damage,reward,dead,ns.player_health,ns.player_position[0],ns.player_position[1]],jnp.float32)
    run=jax.jit(jax.vmap(jax.vmap(rollout,in_axes=(None,0,None)),in_axes=(None,None,0)))
    plans=jnp.arange(6,dtype=jnp.int32);out=[];reject=[]
    for i,r in enumerate(selected):
        st=D.RP.advance_to(r['shard'],r['slot'],r['time']);actual=np.asarray(frame(st));raw=D.RP.episode_fields(r['shard'],r['slot'])['observations'][r['time']].numpy()
        assert np.array_equal(actual,raw)
        world=np.asarray(st.player_position)+np.array(divmod(r['cell'],9))-np.array([3,4]);assert visible_kind(st,world)==r['kind']
        if r['task']=='kind':
            name=KINDS[r['kind']];ent=getattr(st,name);assert np.array_equal(np.asarray(ent.position[r['j']]),world)
            ent=ent.replace(mask=ent.mask.at[r['j']].set(False),health=ent.health.at[r['j']].set(0));other=st.replace(**{name:ent})
            if r['kind']!=4:
                occupied=any(bool((np.asarray(getattr(other,k).mask)&(np.asarray(getattr(other,k).position)==world).all(-1)).any()) for k in ['zombies','cows','skeletons'])
                other=other.replace(mob_map=other.mob_map.at[world[0],world[1]].set(occupied))
            alternate_label=visible_kind(other,world)
        else:
            other=st.replace(arrow_directions=st.arrow_directions.at[r['j']].set(-st.arrow_directions[r['j']]))
            alternate_label=[1,0,3,2][r['heading']]
        if alternate_label==(r['kind'] if r['task']=='kind' else r['heading']):
            reject.append(dict(meta=r,reason='removed entity not the sole visible explanation'));continue
        keys=jax.random.split(jax.random.PRNGKey(618000+i),32*6).reshape(32,6,2)
        values=np.stack([np.asarray(run(s,keys,plans)) for s in [st,other]])
        assert np.isfinite(values).all()
        out.append(dict(meta=r,alternate_label=alternate_label,current=np.stack([actual,np.asarray(frame(other))]),values=values))
        print('causal',len(out),'/',len(selected),'kind',r['kind'],'episode',r['episode'],'max paired injury effect',float((values[0,:,:,0]-values[1,:,:,0]).mean(-1).max()),flush=True)
        torch.save(dict(rows=out,rejected=reject,plans=['NOOP','LEFT','RIGHT','UP','DOWN','DO'],keys=32,horizon=6,columns=['cumulative_health_decrements','reward_sum','terminated','final_health','player_row','player_col']),HERE/output)
    rejected='balanced_consequence_rejections.json' if balanced else ('focus_consequence_rejections.json' if focused else 'consequence_rejections.json')
    (HERE/rejected).write_text(json.dumps(reject,indent=2)+'\n')

if __name__=='__main__':main()
