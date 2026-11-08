"""Fresh complete episodes with competent play and fixed exploratory bursts.

Run as a module from the repository root. Training/evaluation corpora are not
modified. The GPU preflight checks exact parity with the existing collector.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import sys
import time

os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE','false')
import jax
import jax.numpy as jnp
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from artifacts import collect_support as OLD
from craftax.craftax_classic.constants import Achievement, BlockType, BLOCK_PIXEL_SIZE_AGENT
from craftax.craftax_classic.renderer import make_craftax_pixel_renderer
from craftax.craftax_env import make_craftax_env_from_name
from d4mj.data import Episode, STORE_FORMAT, atomic_manifest, save_episode_shard

HERE=Path(__file__).resolve().parent
OLD_STORE=ROOT/'artifacts/craftax_support_v2'
SEED=202611040
VERSION='saccade-coverage-v1'


def digest(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()


def schedule(round_index,limit):
    t=np.arange(limit);group=round_index//2
    if round_index%2==0:
        base=(0.,.02,.05,.10)[group%4]
        return np.full(limit,base,np.float32),dict(component='competent',base_epsilon=base,prefix=None)
    prefix=(512,1024)[group%2]
    burst=(t>=prefix)&((t-prefix)%128<16)
    return np.where(burst,1.,.02).astype(np.float32),dict(component='bursts',base_epsilon=.02,prefix=prefix)


def facts(s):
    inv=jnp.stack([getattr(s.inventory,f) for f in s.inventory.__dataclass_fields__])
    view=jax.lax.dynamic_slice(jnp.pad(s.map,4,constant_values=BlockType.OUT_OF_BOUNDS.value),
                               (s.player_position[0]+1,s.player_position[1]),(7,9)).astype(jnp.int8)
    return dict(pos=s.player_position,dir=s.player_direction,light=s.light_level,sleep=s.is_sleeping,
        body=jnp.stack([s.player_health,s.player_food,s.player_drink,s.player_energy]),inv=inv,
        ach=s.achievements,view=view,timestep=s.timestep,
        **{f'{kind}_{attr}':getattr(getattr(s,kind),field) for kind in ('zombies','cows','skeletons','arrows')
           for attr,field in [('p','position'),('m','mask'),('hp','health'),('cd','attack_cooldown')]})


def make_rollout(network,params,env,ep,render_v,num_envs,limit):
    reset=jax.vmap(env.reset,in_axes=(0,None));step=jax.vmap(env.step,in_axes=(0,0,0,None))
    rec=jax.vmap(facts)
    def dead(s):
        return (s.map[s.player_position[0],s.player_position[1]]==BlockType.LAVA.value)|(s.player_health<=0)
    dead_v=jax.vmap(dead);n_actions=int(env.action_space(ep).n)
    @jax.jit
    def run(key,epsilons):
        key,rk=jax.random.split(key);reset_keys=jax.random.split(rk,num_envs)
        obs,state=reset(reset_keys,ep);initial=rec(state);initial_frames=render_v(state)
        hidden=OLD.ScannedRNN.initialize_carry(num_envs,OLD.LAYER)
        def body(carry,xs):
            hidden,state,obs,previous_done=carry;k,epsilon=xs
            ka,kn,ku,sk=jax.random.split(k,4)
            hidden,logits,_=network.apply(params,hidden,(obs[None],previous_done[None]))
            policy_action=jax.random.categorical(ka,logits[0])
            random_action=jax.random.randint(kn,(num_envs,),0,n_actions)
            take=jax.random.uniform(ku,(num_envs,))<epsilon
            action=jnp.where(take,random_action,policy_action)
            expert_p=jnp.take_along_axis(jax.nn.softmax(logits[0]),action[:,None],axis=-1)[:,0]
            propensity=(1-epsilon)*expert_p+epsilon/n_actions
            obs2,state2,reward,done,_=step(jax.random.split(sk,num_envs),state,action,ep)
            return (hidden,state2,obs2,done),(render_v(state2),action,reward,done,dead_v(state2),take,propensity,rec(state2))
        _,out=jax.lax.scan(body,(hidden,state,obs,jnp.zeros(num_envs,dtype=bool)),
                           (jax.random.split(key,limit),epsilons))
        return initial_frames,initial,out,reset_keys
    return run


def pack(initial_frames,initial,out,epsilons,*,round_index,seed,policy):
    frames,actions,rewards,dones,deads,exogenous,probability,states=out
    episodes=[];arrays={};lengths=[]
    for slot in range(frames.shape[1]):
        done=dones[:,slot].astype(bool);T=int(np.argmax(done))+1 if done.any() else len(done)
        terminal=bool(deads[T-1,slot]) if done.any() else False
        assert not done[:T-1].any()
        if done.any():assert terminal,'Unexpected time-limit termination below the 2500 collector cap'
        terminated=np.zeros(T,bool);terminated[-1]=terminal
        truncated=np.zeros(T,bool);truncated[-1]=not terminal
        f={k:np.concatenate([initial[k][slot][None],v[:T,slot]],0) for k,v in states.items()}
        observations=np.concatenate([initial_frames[slot][None],frames[:T,slot]],0)
        cause=None
        if terminal:cause='health' if f['body'][-1,0]<=0 else 'lava'
        episodes.append(Episode(observations=torch.from_numpy(np.array(observations,copy=True)),
            actions_taken=torch.from_numpy(np.array(actions[:T,slot],dtype=np.int64)),
            rewards=torch.from_numpy(np.array(rewards[:T,slot],dtype=np.float32)),
            terminated=torch.from_numpy(terminated),truncated=torch.from_numpy(truncated),
            events=torch.from_numpy((f['ach'][1:].sum(-1)>f['ach'][:-1].sum(-1))),
            epsilon=policy['base_epsilon'] if policy['component']=='competent' else None,
            uniform_eligible=True,bc_eligible=False,split='train',
            episode_id=f'coverage-v1:{seed}:{round_index}:{slot}',terminal_cause=cause))
        for k,v in f.items():arrays.setdefault('s_'+k,[]).append(v)
        for k,v in [('action',actions[:T,slot]),('reward',rewards[:T,slot]),
                    ('terminated',terminated),('truncated',truncated),('exogenous',exogenous[:T,slot]),
                    ('epsilon',epsilons[:T]),('behavior_probability',probability[:T,slot])]:
            arrays.setdefault('t_'+k,[]).append(v)
        lengths.append(T)
    audit={k:np.concatenate(v) for k,v in arrays.items()}
    audit.update(ep_T=np.array(lengths,np.int32),ep_slot=np.arange(len(lengths),dtype=np.int32),
                 ep_round=np.full(len(lengths),round_index,np.int32))
    validate_pack(episodes,audit)
    return episodes,audit


def validate_pack(episodes,a):
    so=np.r_[0,np.cumsum(a['ep_T']+1)];ao=np.r_[0,np.cumsum(a['ep_T'])]
    for i,e in enumerate(episodes):
        sl=slice(ao[i],ao[i+1]);ss=slice(so[i],so[i+1]);T=len(e)
        assert len(e.observations)==T+1 and e.observations.dtype==torch.uint8
        assert e.observations.shape[1:]==(63,63,3)
        assert np.array_equal(e.actions_taken.numpy(),a['t_action'][sl])
        assert np.array_equal(e.rewards.numpy(),a['t_reward'][sl])
        assert np.array_equal(e.terminated.numpy(),a['t_terminated'][sl])
        assert not (e.terminated&e.truncated).any()
        assert int(e.terminated.sum()+e.truncated.sum())==1
        assert bool(e.terminated[-1] or e.truncated[-1])
        assert np.array_equal(a['s_timestep'][ss],np.arange(T+1))
        assert np.all((a['t_behavior_probability'][sl]>0)&(a['t_behavior_probability'][sl]<=1))
        eps=a['t_epsilon'][sl];ex=a['t_exogenous'][sl]
        assert ex[eps==1].all() and not ex[eps==0].any()
        assert np.allclose(a['t_behavior_probability'][sl][eps==1],1/17)
        assert not e.bc_eligible and e.split=='train'


def summarize(episodes,a):
    out={}
    for part in ('all','train','monitor'):
        result=dict(episodes=0,transitions=0,deaths=0,damage=0,risk=0,risk_deaths=0,
                    random_actions=0,unlocks=Counter(),inventory_increases=Counter(),
                    inventory_before_value={},body_increases=Counter(),body_decreases=Counter(),
                    entity_visible_frames=Counter(),day_unlocks=Counter(),night_unlocks=Counter())
        so=np.r_[0,np.cumsum(a['ep_T']+1)];ao=np.r_[0,np.cumsum(a['ep_T'])]
        for i,e in enumerate(episodes):
            monitor=int(hashlib.sha256(e.episode_id.encode()).hexdigest(),16)%10000<500
            if part!='all' and monitor!=(part=='monitor'):continue
            g=np.arange(so[i],so[i+1]-1);tr=slice(ao[i],ao[i+1]);h=a['s_body'][g,0];hn=a['s_body'][g+1,0]
            result['episodes']+=1;result['transitions']+=len(e);result['deaths']+=int(e.terminated.sum())
            result['damage']+=int((hn<h).sum());result['random_actions']+=int(a['t_exogenous'][tr].sum())
            adj=((np.abs(a['s_zombies_p'][g]-a['s_pos'][g,None]).sum(-1)==1)&a['s_zombies_m'][g]).any(-1)
            risk=(h<=2)&adj;result['risk']+=int(risk.sum());result['risk_deaths']+=int((risk&e.terminated.numpy()).sum())
            unlocked=a['s_ach'][g+1].astype(bool)&~a['s_ach'][g].astype(bool);night=a['s_light'][g]<.5
            for ac in Achievement:
                hit=unlocked[:,ac.value];result['unlocks'][ac.name]+=int(hit.sum())
                result['day_unlocks'][ac.name]+=int((hit&~night).sum());result['night_unlocks'][ac.name]+=int((hit&night).sum())
            for col in range(a['s_inv'].shape[1]):
                before=a['s_inv'][g,col];hit=a['s_inv'][g+1,col]>before;name=str(col)
                result['inventory_increases'][name]+=int(hit.sum())
                bands=result['inventory_before_value'].setdefault(name,Counter())
                for value,n in zip(*np.unique(before[hit],return_counts=True)):bands[str(int(value))]+=int(n)
            for col,name in enumerate(('health','food','drink','energy')):
                delta=a['s_body'][g+1,col]-a['s_body'][g,col]
                result['body_increases'][name]+=int((delta>0).sum());result['body_decreases'][name]+=int((delta<0).sum())
            for kind in ('zombies','cows','skeletons','arrows'):
                rel=a[f's_{kind}_p'][g]-a['s_pos'][g,None]
                visible=a[f's_{kind}_m'][g]&(np.abs(rel[:,:,0])<=3)&(np.abs(rel[:,:,1])<=4)
                result['entity_visible_frames'][kind]+=int(visible.sum())
        out[part]=result
    return out


def add_counts(a,b):
    for key,value in b.items():
        if isinstance(value,dict):add_counts(a.setdefault(key,{}),value)
        else:a[key]=a.get(key,0)+value


def reset_keys(seed,round_index,n):
    _,rk=jax.random.split(jax.random.PRNGKey(seed+round_index))
    return np.asarray(jax.random.split(rk,n))


def check_fresh_keys(args):
    old=json.loads((OLD_STORE/'manifest.json').read_text());existing=set()
    for r in range(old['rounds']):existing.update(map(tuple,reset_keys(old['seed'],r,old['num_envs']).tolist()))
    new=set()
    for r in range(args.rounds):
        keys=set(map(tuple,reset_keys(args.seed,r,args.num_envs).tolist()))
        assert len(keys)==args.num_envs and not(keys&existing) and not(keys&new)
        new.update(keys)
    return dict(old_reset_keys=len(existing),fresh_reset_keys=len(new),overlap=0)


def setup():
    env=make_craftax_env_from_name('Craftax-Classic-Symbolic-v1',auto_reset=False);ep=env.default_params
    network,params,_=OLD.load_expert(OLD.PARAMS,obs_dim=int(env.observation_space(ep).shape[0]),
                                    action_dim=int(env.action_space(ep).n),layer_size=OLD.LAYER)
    render=make_craftax_pixel_renderer(int(BLOCK_PIXEL_SIZE_AGENT))
    frame=lambda s:jnp.clip(jnp.round(render(s)),0,255).astype(jnp.uint8)
    return env,ep,network,params,jax.vmap(frame),jax.jit(frame)


def parity(env,ep,network,params,render_v):
    n,limit,eps=4,32,.1;key=jax.random.PRNGKey(202611039)
    run=make_rollout(network,params,env,ep,render_v,n,limit)
    f0,s0,new,rk=jax.device_get(run(key,jnp.full(limit,eps)))
    old=jax.device_get(OLD.rollout_batch(network,params,env,ep,render_v,key,n,eps,limit))
    for name,x,y in [('initial frame',f0,old[0]),('initial achievements',s0['ach'],old[1]),
        ('frames',new[0],old[2][0]),('actions',new[1],old[2][1]),('rewards',new[2],old[2][2]),
        ('termination',new[3],old[2][3]),('achievements',new[7]['ach'],old[2][4]),
        ('deaths',new[4],old[2][5]),('reset keys',rk,old[3])]:
        assert np.array_equal(x,y),('legacy parity',name)
    return dict(exact_legacy_parity=True,environments=n,steps=limit,epsilon=eps)


def replay_check(episodes,audit,keys,env,ep,frame):
    """Independent replay of every retained frame/action/outcome, first/last env."""
    selected=np.array([0,len(episodes)-1]);actions=np.zeros((2500,2),np.int32)
    for col,i in enumerate(selected):actions[:len(episodes[i]),col]=episodes[i].actions_taken.numpy()
    r=int(audit['ep_round'][0]);key,_=jax.random.split(jax.random.PRNGKey(SEED+r))
    step_keys=jax.random.split(key,2500)
    env_keys=jax.vmap(lambda k:jax.random.split(jax.random.split(k,4)[3],len(episodes))[selected])(step_keys)
    reset=jax.vmap(env.reset,in_axes=(0,None));step=jax.vmap(env.step,in_axes=(0,0,0,None))
    render=jax.vmap(frame)
    @jax.jit
    def replay(reset_keys,env_keys,actions):
        _,state=reset(reset_keys,ep);initial=render(state)
        def body(state,xs):
            k,action=xs;_,state,reward,done,_=step(k,state,action,ep)
            return state,(render(state),reward,done)
        _,out=jax.lax.scan(body,state,(env_keys,actions))
        return initial,out
    initial,(frames,rewards,dones)=jax.device_get(replay(jnp.asarray(keys[selected]),env_keys,jnp.asarray(actions)))
    checked=[]
    for col,i in enumerate(selected):
        e=episodes[i];T=len(e)
        assert np.array_equal(initial[col],e.observations[0].numpy())
        assert np.array_equal(frames[:T,col],e.observations[1:].numpy()),('replay frames',int(i))
        assert np.array_equal(rewards[:T,col],e.rewards.numpy()),('replay rewards',int(i))
        assert np.array_equal(dones[:T,col],e.terminated.numpy()),('replay done',int(i))
        checked.append(dict(slot=int(i),steps=T,all_frames_exact=True,all_rewards_exact=True,all_terminations_exact=True))
    return checked


def source_pins():
    import craftax.craftax_classic as classic
    paths=[Path(__file__),ROOT/'artifacts/collect_support.py',ROOT/'d4mj/data.py',
           ROOT/'d4_mamba_jepa/expert/ppo_expert.py',ROOT/OLD.PARAMS,OLD_STORE/'manifest.json']
    paths+=sorted(Path(classic.__file__).parent.rglob('*.py'))
    return {str(p):digest(p) for p in paths}


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True)
    p.add_argument('--rounds',type=int,default=128);p.add_argument('--num-envs',type=int,default=24)
    p.add_argument('--seed',type=int,default=SEED);p.add_argument('--max-seconds',type=int,default=7200)
    args=p.parse_args();assert args.seed==SEED and args.rounds>0 and args.num_envs>0
    torch.set_num_threads(2)
    assert jax.default_backend()=='gpu','This launch is budgeted for the verified GPU path'
    args.out.mkdir(parents=True,exist_ok=True);mp=args.out/'manifest.json'
    contract=dict(version=VERSION,seed=args.seed,rounds=args.rounds,num_envs=args.num_envs,limit=2500,
        max_seconds=args.max_seconds,jax_version=jax.__version__,source_pins=source_pins(),
        schedule='even: eps cycle 0,.02,.05,.10; odd: eps .02, 16/128 fully random after alternating 512/1024 prefix',
        split='all fresh train sources; fixed monitor hash sha256(episode_id)%10000 < 500',
        retention='all complete first-done/cap episodes, no outcome selection')
    if mp.exists():
        manifest=json.loads(mp.read_text());assert manifest['contract']==contract,'Resume contract changed'
        if manifest['complete']:print('already complete',flush=True);return
    else:
        manifest=dict(format=STORE_FORMAT,kind=VERSION,complete=False,contract=contract,shards=[],rounds=0,
            episodes=0,transitions=0,terminal_episodes=0,started_unix=time.time(),coverage={},components={})
        atomic_manifest(mp,manifest)
    env,ep,network,params,render_v,frame=setup()
    if not manifest.get('preflight'):
        manifest['preflight']={**check_fresh_keys(args),**parity(env,ep,network,params,render_v)}
        atomic_manifest(mp,manifest);print('PREFLIGHT',json.dumps(manifest['preflight']),flush=True)
    run=make_rollout(network,params,env,ep,render_v,args.num_envs,2500)
    while manifest['rounds']<args.rounds:
        if time.time()-manifest['started_unix']>=args.max_seconds:break
        r=manifest['rounds'];epsilon,policy=schedule(r,2500);started=time.time()
        f0,s0,out,rk=jax.device_get(run(jax.random.PRNGKey(args.seed+r),jnp.asarray(epsilon)))
        episodes,audit=pack(f0,s0,out,epsilon,round_index=r,seed=args.seed,policy=policy)
        if r==0:
            manifest['first_batch_replay']=replay_check(episodes,audit,rk,env,ep,frame)
            print('FIRST_BATCH_REPLAY',json.dumps(manifest['first_batch_replay']),flush=True)
        shard=args.out/f'shard-{r:06d}.pt';side=args.out/f'audit-{r:06d}.npz'
        assert not shard.exists() and not side.exists(),'Unregistered output; inspect before resuming'
        record=save_episode_shard(shard,episodes)
        temp=side.with_suffix('.tmp.npz');np.savez_compressed(temp,**audit);temp.replace(side)
        saved=torch.load(shard,weights_only=False,mmap=True)
        assert len(saved['episodes'])==len(episodes)
        for x,e in zip(saved['episodes'],episodes):
            assert torch.equal(x['observations'],e.observations) and torch.equal(x['actions_taken'],e.actions_taken)
        counts=summarize(episodes,audit);record.update(policy=policy,audit_file=side.name,audit_sha256=digest(side),
            seconds=time.time()-started,coverage=counts)
        manifest['shards'].append(record);manifest['rounds']+=1;manifest['episodes']+=len(episodes)
        manifest['transitions']+=sum(map(len,episodes));manifest['terminal_episodes']+=sum(bool(e.terminated.any()) for e in episodes)
        add_counts(manifest['coverage'],counts);add_counts(manifest['components'].setdefault(policy['component'],{}),counts)
        manifest['seconds']=time.time()-manifest['started_unix'];atomic_manifest(mp,manifest)
        print('BATCH',json.dumps(dict(round=manifest['rounds'],policy=policy,seconds=record['seconds'],
            episodes=manifest['episodes'],transitions=manifest['transitions'],terminal_episodes=manifest['terminal_episodes'],
            new_unlocks=counts['all']['unlocks'],train_unlocks=manifest['coverage']['train']['unlocks'])),flush=True)
        del f0,s0,out,episodes,audit,saved
    manifest.update(complete=True,completed_unix=time.time(),seconds=time.time()-manifest['started_unix'],
        stop_reason='fixed_round_budget' if manifest['rounds']==args.rounds else 'wall_budget')
    atomic_manifest(mp,manifest)
    print('COMPLETE',json.dumps({k:v for k,v in manifest.items() if k not in ('contract','shards')}),flush=True)


if __name__=='__main__':main()
