"""CPU integrity check of actual uniform tails and exact replay after the switch."""
import os
os.environ.setdefault('JAX_PLATFORMS','cpu')
import importlib
import json
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
import torch

R=importlib.import_module('artifacts.experiments.20261104_rare_collection.risk_collect')
C=R.C;HERE=Path(__file__).resolve().parent;STORE=C.ROOT/'artifacts/craftax_risk_v1'


@jax.jit
def flags_and_actions(seed,r,eps):
    key,_=jax.random.split(jax.random.PRNGKey(seed+r));keys=jax.random.split(key,len(eps))
    def one(k,e):
        _,kn,ku,_=jax.random.split(k,4)
        return jax.random.uniform(ku,(24,))<e,jax.random.randint(kn,(24,),0,17)
    return jax.vmap(one)(keys,eps)


def main():
    m=json.loads((STORE/'manifest.json').read_text());assert m['contract']['num_envs']==24
    checked=0;cases=[]
    for sh in m['shards']:
        with np.load(STORE/sh['audit_file']) as z:
            eps,policy=R.schedule(int(z['ep_round'][0]));rnd,acts=jax.device_get(flags_and_actions(R.SEED,int(z['ep_round'][0]),jnp.asarray(eps)))
            off=np.r_[0,np.cumsum(z['ep_T'])]
            for col,T in enumerate(z['ep_T']):
                sl=slice(off[col],off[col+1]);flag=z['t_exogenous'][sl]
                assert np.array_equal(flag,rnd[:T,col]),('random flag',sh['file'],col)
                assert np.array_equal(z['t_action'][sl][flag],acts[:T,col][flag]),('uniform action',sh['file'],col)
                checked+=int(T)
                if len(cases)<2 and policy['prefix'] is not None and T>=policy['prefix']+16:
                    cases.append(dict(shard=sh['file'],round=int(z['ep_round'][col]),slot=col,T=int(T),prefix=policy['prefix']))
    assert len(cases)==2,'Need actual retained histories extending beyond the uniform switch'
    env=C.make_craftax_env_from_name('Craftax-Classic-Symbolic-v1',auto_reset=False);ep=env.default_params
    renderer=C.make_craftax_pixel_renderer(int(C.BLOCK_PIXEL_SIZE_AGENT))
    render=lambda s:jnp.clip(jnp.round(renderer(s)),0,255).astype(jnp.uint8)
    @jax.jit
    def replay(rk,keys,actions):
        _,s=env.reset(rk,ep);initial=render(s)
        def body(s,x):
            key,action=x;_,s,r,d,_=env.step(key,s,action,ep)
            return s,(render(s),r,d)
        _,out=jax.lax.scan(body,s,(keys,actions));return initial,out
    for case in cases:
        e=torch.load(STORE/case['shard'],weights_only=False,mmap=True)['episodes'][case['slot']]
        actions=np.zeros(2500,np.int32);actions[:case['T']]=e['actions_taken'].numpy()
        key,rk=jax.random.split(jax.random.PRNGKey(R.SEED+case['round']))
        keys=jax.vmap(lambda k:jax.random.split(jax.random.split(k,4)[3],24)[case['slot']])(jax.random.split(key,2500))
        initial,(f,r,d)=jax.device_get(replay(jax.random.split(rk,24)[case['slot']],keys,jnp.asarray(actions)))
        T=case['T'];assert np.array_equal(initial,e['observations'][0].numpy())
        actual=e['observations'][1:].numpy();delta=f[:T].astype(np.int16)-actual.astype(np.int16)
        mismatch=(delta!=0).reshape(T,-1).any(-1)
        case.update(every_retained_frame_exact=bool(not mismatch.any()),
            maximum_pixel_difference=int(np.abs(delta).max()),different_pixel_channels=int((delta!=0).sum()),
            total_pixel_channels=int(delta.size),different_frames=int(mismatch.sum()),
            first_different_frame=int(np.flatnonzero(mismatch)[0]+1) if mismatch.any() else None,
            every_reward_exact=bool(np.array_equal(r[:T],e['rewards'].numpy())),
            every_termination_exact=bool(np.array_equal(d[:T],e['terminated'].numpy())))
    out=dict(snapshot_rounds=m['rounds'],random_flags_checked_transitions=checked,
        random_actions_exact=True,actual_post_switch_replays=cases,backend=jax.default_backend(),
        original_all_frames_exact_gate_passed=all(c['every_retained_frame_exact'] and c['every_reward_exact'] and c['every_termination_exact'] for c in cases),
        interpretation='Original CPU cross-backend pixel-exact assertion failed; this rerun quantifies the failure without relaxing that gate. Native GPU parity and within-GPU replay are separate checks.')
    (HERE/'RANDOM_TAIL_CHECK.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out),flush=True)


if __name__=='__main__':main()
