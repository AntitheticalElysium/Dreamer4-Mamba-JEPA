"""Additional natural risk/death episodes; frozen, already-checked mechanics."""
from __future__ import annotations
import argparse
import importlib
import json
from pathlib import Path
import time

import jax
import jax.numpy as jnp
import numpy as np
import torch

C=importlib.import_module('artifacts.experiments.20261104_rare_collection.collect')
SEED=202611500
FIRST=C.ROOT/'artifacts/craftax_coverage_v1/manifest.json'


def schedule(r):
    if r%9!=8:
        eps=(.5,1.,.5,1.,.5,1.,.25,1.)[r%9]
        return np.full(1024,eps,np.float32),dict(component='reset_exploration',base_epsilon=eps,prefix=None)
    prefix=(512,1024)[(r//9)%2]
    return np.where(np.arange(2500)<prefix,.02,1.).astype(np.float32),dict(
        component='late_failure',base_epsilon=.02,prefix=prefix)


def fresh_keys(args):
    old=C.check_fresh_keys(args)
    first=json.loads(FIRST.read_text());cfg=first['contract'];existing=set()
    assert first['complete'],'Do not collide with the first collector'
    for r in range(first['rounds']):
        existing.update(map(tuple,C.reset_keys(cfg['seed'],r,cfg['num_envs']).tolist()))
    for r in range(args.rounds):
        assert not (set(map(tuple,C.reset_keys(args.seed,r,args.num_envs).tolist()))&existing)
    return {**old,'first_collection_reset_keys':len(existing),'first_collection_overlap':0}


def replay_check(episodes,audit,keys,env,ep,frame,*,seed,limit):
    selected=np.array([0,len(episodes)-1]);actions=np.zeros((limit,2),np.int32)
    for col,i in enumerate(selected):actions[:len(episodes[i]),col]=episodes[i].actions_taken.numpy()
    r=int(audit['ep_round'][0]);key,_=jax.random.split(jax.random.PRNGKey(seed+r))
    env_keys=jax.vmap(lambda k:jax.random.split(jax.random.split(k,4)[3],len(episodes))[selected])(
        jax.random.split(key,limit))
    reset=jax.vmap(env.reset,in_axes=(0,None));step=jax.vmap(env.step,in_axes=(0,0,0,None));render=jax.vmap(frame)
    @jax.jit
    def replay(rk,ek,acts):
        _,s=reset(rk,ep);initial=render(s)
        def body(s,x):
            k,a=x;_,s,r,d,_=step(k,s,a,ep)
            return s,(render(s),r,d)
        _,out=jax.lax.scan(body,s,(ek,acts));return initial,out
    initial,(f,r,d)=jax.device_get(replay(jnp.asarray(keys[selected]),env_keys,jnp.asarray(actions)))
    result=[]
    for col,i in enumerate(selected):
        e=episodes[i];T=len(e)
        assert np.array_equal(initial[col],e.observations[0].numpy())
        assert np.array_equal(f[:T,col],e.observations[1:].numpy()),('replay frames',int(i))
        assert np.array_equal(r[:T,col],e.rewards.numpy()),('replay rewards',int(i))
        assert np.array_equal(d[:T,col],e.terminated.numpy()),('replay done',int(i))
        result.append(dict(slot=int(i),steps=T,all_frames_exact=True,all_rewards_exact=True,all_terminations_exact=True))
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True)
    p.add_argument('--rounds',type=int,default=512);p.add_argument('--num-envs',type=int,default=24)
    p.add_argument('--seed',type=int,default=SEED);p.add_argument('--max-seconds',type=int,default=7200)
    args=p.parse_args();assert args.seed==SEED and args.num_envs>=2 and args.rounds>=9
    torch.set_num_threads(2);assert jax.default_backend()=='gpu'
    assert json.loads(FIRST.read_text())['complete'],'First collection must have finished'
    pins=C.source_pins();pins[str(Path(__file__))]=C.digest(Path(__file__))
    pins[str(FIRST)]=C.digest(FIRST)
    contract=dict(version='saccade-risk-v1',seed=args.seed,rounds=args.rounds,num_envs=args.num_envs,
        max_seconds=args.max_seconds,limits=[1024,2500],source_pins=pins,
        policy='8 reset-exploration batches [.5,1,.5,1,.5,1,.25,1], then 1 late-failure batch: .02 prefix 512/1024 then uniform',
        split='fresh TRAIN sources; fixed5% monitor episode-id hash',retention='all outcomes through first done/cap')
    args.out.mkdir(parents=True,exist_ok=True);mp=args.out/'manifest.json'
    if mp.exists():
        m=json.loads(mp.read_text());assert m['contract']==contract,'Resume contract changed'
        if m['complete']:print('already complete',flush=True);return
    else:
        m=dict(format=C.STORE_FORMAT,kind='saccade-risk-v1',complete=False,contract=contract,shards=[],rounds=0,
            episodes=0,transitions=0,terminal_episodes=0,started_unix=time.time(),coverage={},components={})
        C.atomic_manifest(mp,m)
    env,ep,net,params,render_v,frame=C.setup()
    if not m.get('preflight'):
        m['preflight']={**fresh_keys(args),**C.parity(env,ep,net,params,render_v)}
        C.atomic_manifest(mp,m);print('PREFLIGHT',json.dumps(m['preflight']),flush=True)
    runs={};checks=m.setdefault('first_component_replays',{})
    while m['rounds']<args.rounds and time.time()-m['started_unix']<args.max_seconds:
        ix=m['rounds'];eps,policy=schedule(ix);limit=len(eps);started=time.time()
        if limit not in runs:runs[limit]=C.make_rollout(net,params,env,ep,render_v,args.num_envs,limit)
        f0,s0,out,rk=jax.device_get(runs[limit](jax.random.PRNGKey(args.seed+ix),jnp.asarray(eps)))
        # The constant-epsilon Episode field is accurate for reset exploration;
        # scheduled late-failure episodes have epsilon=None and exact sidecars.
        pack_policy=dict(component='competent' if policy['component']=='reset_exploration' else 'scheduled',
                         base_epsilon=policy['base_epsilon'])
        episodes,a=C.pack(f0,s0,out,eps,round_index=ix,seed=args.seed,policy=pack_policy)
        a['ep_limit']=np.full(len(episodes),limit,np.int32)
        if policy['component'] not in checks:
            checks[policy['component']]=replay_check(episodes,a,rk,env,ep,frame,seed=args.seed,limit=limit)
            print('COMPONENT_REPLAY',policy['component'],json.dumps(checks[policy['component']]),flush=True)
        shard=args.out/f'shard-{ix:06d}.pt';side=args.out/f'audit-{ix:06d}.npz'
        assert not shard.exists() and not side.exists(),'Unregistered output: inspect before resume'
        record=C.save_episode_shard(shard,episodes)
        temp=side.with_suffix('.tmp.npz');np.savez_compressed(temp,**a);temp.replace(side)
        saved=torch.load(shard,weights_only=False,mmap=True)
        assert len(saved['episodes'])==len(episodes)
        for row,e in zip(saved['episodes'],episodes):
            assert torch.equal(row['observations'],e.observations) and torch.equal(row['actions_taken'],e.actions_taken)
        q=C.summarize(episodes,a)
        for name,part in q.items():
            selected=[e for e in episodes if name=='all' or
                (int(C.hashlib.sha256(e.episode_id.encode()).hexdigest(),16)%10000<500)==(name=='monitor')]
            part['death_causes']={cause:sum(e.terminal_cause==cause for e in selected) for cause in ('health','lava')}
            assert sum(part['death_causes'].values())==part['deaths']
        record.update(policy=policy,limit=limit,audit_file=side.name,audit_sha256=C.digest(side),
                      seconds=time.time()-started,coverage=q)
        m['shards'].append(record);m['rounds']+=1;m['episodes']+=len(episodes)
        m['transitions']+=sum(map(len,episodes));m['terminal_episodes']+=sum(bool(e.terminated.any()) for e in episodes)
        C.add_counts(m['coverage'],q);C.add_counts(m['components'].setdefault(policy['component'],{}),q)
        m['seconds']=time.time()-m['started_unix'];C.atomic_manifest(mp,m)
        print('BATCH',json.dumps(dict(round=m['rounds'],policy=policy,seconds=record['seconds'],episodes=m['episodes'],
            transitions=m['transitions'],coverage={k:m['coverage']['train'][k] for k in
                ['episodes','transitions','deaths','damage','risk','risk_deaths','random_actions']})),flush=True)
        del f0,s0,out,episodes,a,saved
    m.update(complete=True,seconds=time.time()-m['started_unix'],completed_unix=time.time(),
             stop_reason='fixed_round_budget' if m['rounds']==args.rounds else 'wall_budget')
    C.atomic_manifest(mp,m)
    print('COMPLETE',json.dumps({k:v for k,v in m.items() if k not in ['contract','shards']}),flush=True)


if __name__=='__main__':main()
