"""Small inverse-channel optimization control; not a candidate world or gate.

Four FIT day roots, distinct observed futures: same history/action, varying target.
No-channel, exact-hard posterior, soft posterior, and assigned hard-code decoder
control. Assigned codes are dataset identities, not learned innovations. The soft
arm is a deliberately continuous future bypass, used only to diagnose quantization.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np
import torch
import torch.nn.functional as F
sys.path.insert(0,str(Path(__file__).parent))
import innovation_model as I
import innovation_train as T
import loss_intervention as L
from d4mj.config import config_from_dict
from d4mj.train import autocast_context,phase_optimizer,optimizer_step,_phase_lr

ARMS=('none','hard','soft','assigned')
DIRECTORY=T.OUT/'optimization_control'


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--updates',type=int,default=2000)
    p.add_argument('--batch',type=int,default=16)
    p.add_argument('--stop-after',type=int)
    a=p.parse_args();torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.manual_seed(7);torch.cuda.manual_seed_all(7)
    panel=torch.load(L.OUT/'panel.pt',map_location='cpu',weights_only=False)
    fan=torch.load(T.OUT/'real_futures/fan.pt',map_location='cpu',weights_only=False)
    visible=torch.load(L.OUT/'snap_visible.pt',map_location='cpu',weights_only=False)['visible']
    split=json.loads((L.OUT/'generated_prefix/readout_refit/contract.json').read_text())['split']
    fit=set(split['FIT']);selected=[]
    for i,root in enumerate(panel['roots']):
        # Renderer night static is exactly disabled at daylight >=0.5;
        # timestep-based light is identical across RNG keys at this transition.
        if root['episode_id'] in fit and bool((visible[i,15:17,1521]>=.5).all()):
            variants=fan['layouts'][i].flatten(1).unique(dim=0)
            if len(variants)>1:selected.append(i)
        if len(selected)==4:break
    assert len(selected)==4
    frames,actions,targets,root_index=[],[],[],[]
    for j,root in enumerate(selected):
        unique=fan['tokens'][root].flatten(1).unique(dim=0).view(-1,81,192)
        for target in unique:
            frames.append(panel['tokens'][root,1:16]);actions.append(panel['actions'][root,1:16])
            targets.append(target);root_index.append(j)
    frames=torch.stack(frames).float();actions=torch.stack(actions);targets=torch.stack(targets).float()
    root_index=torch.tensor(root_index);n=len(frames);assert n<=128
    # Uniform over distinct observed futures. This is a representability check,
    # not the environment's frequencies or a deployment training distribution.
    root_means=torch.stack([targets[root_index==j].mean(0)for j in range(4)])
    conditional_floor=float((targets-root_means[root_index]).square().mean())
    assert conditional_floor>0
    ids=torch.arange(n);assigned=torch.stack([(ids//(8**k))%8 for k in range(6)],-1)
    config=config_from_dict(torch.load(L.TW.S.CHECKPOINT,map_location='cpu',weights_only=False)['config'])
    contract={'format':'inverse_channel_tiny_support_v1','sources':{str(Path(f).relative_to(L.ROOT)):L.sha(f)for f in (__file__,I.__file__,T.__file__,L.__file__)},
              'parent_sha256':L.sha(T.INIT),'panel_sha256':L.sha(L.OUT/'panel.pt'),'fan_sha256':L.sha(T.OUT/'real_futures/fan.pt'),
              'selected_FIT_roots':[panel['roots'][i]for i in selected],'frames_sha256':hashlib.sha256(frames.numpy().tobytes()).hexdigest(),
              'targets_sha256':hashlib.sha256(targets.numpy().tobytes()).hexdigest(),'cases':n,'updates':a.updates,'batch':a.batch,
              'arms':ARMS,'loss':'full D on the current next-state only; no prior or rate term',
              'scope':'deliberate overfit on four reused FIT roots; no generalization or factual-only training claim; no TEST roots trained',
              'controls':'same full world, channel init, seeded case batches, phase AdamW; only current code type changes',
              'zero_initial_adapter':True,'assigned_code_bits':int(np.ceil(np.log2(n))),
              'conditional_mean_squared_error_floor':conditional_floor}
    contract=json.loads(json.dumps(contract))
    cid=hashlib.sha256(json.dumps(contract,sort_keys=True).encode()).hexdigest()
    DIRECTORY.mkdir(exist_ok=True)
    path=DIRECTORY/'contract.json'
    if path.exists():assert json.loads(path.read_text())==contract
    else:L.atomic(contract,path,True)
    parent=torch.load(T.INIT,map_location='cpu',weights_only=False)
    template=I.Innovation();worlds,channels,optimizers={},{},{}
    for arm in ARMS:
        w=L.TW.TWorld('corrt',None,'fmamba','all',0,False,16);w.load_state_dict(parent['world'])
        worlds[arm]=w.cuda().train();channels[arm]=copy.deepcopy(template).cuda().train()
        opt=phase_optimizer([worlds[arm]],config)
        if arm!='none':opt.add_param_group({'params':channels[arm].completion_parameters(),
                                           'lr':opt.param_groups[0]['lr'],'weight_decay':opt.param_groups[0]['weight_decay']})
        optimizers[arm]=opt
    sampler=torch.Generator().manual_seed(919)
    state_path=DIRECTORY/'state.pt';start=0;history=[]
    if state_path.exists():
        st=torch.load(state_path,map_location='cpu',weights_only=False);assert st['contract_id']==cid
        for arm in ARMS:
            worlds[arm].load_state_dict(st['worlds'][arm]);channels[arm].load_state_dict(st['channels'][arm]);optimizers[arm].load_state_dict(st['optimizers'][arm])
            assert L.restored_equal(L.cpu_state(worlds[arm]),st['worlds'][arm])
            assert L.restored_equal(L.cpu_state(channels[arm]),st['channels'][arm])
            assert L.restored_equal(optimizers[arm].state_dict(),st['optimizers'][arm])
        sampler.set_state(st['sampler']);torch.set_rng_state(st['torch_rng']);torch.cuda.set_rng_state_all(st['cuda_rng'])
        start,history=st['update'],st['history']
    if a.stop_after and start>=a.stop_after:return
    def prediction(arm,indices):
        fr,ac=frames[indices].cuda(),actions[indices].cuda()
        world,channel=worlds[arm],channels[arm]
        weights=None
        if arm=='none':delta=None;code=None
        else:
            belief=I.native_belief(world,fr,ac,config)
            if arm=='assigned':
                code=assigned[indices].cuda();weights=F.one_hot(code,8).float()
            else:
                logits=channel.posterior(belief[:,-1],fr[:,-1],targets[indices].cuda())
                weights,code=(channel.hard(logits) if arm=='hard'else (logits.softmax(-1),logits.argmax(-1)))
            delta=F.pad(channel.condition(belief[:,-1],weights)[:,None],(0,0,0,0,14,0))
        with I.conditioned(world,delta),autocast_context(config):out=world(fr,ac)[0][:,-1].float()
        return out,code
    def evaluate(update):
        row={'update':update,'models':{}}
        for arm in ARMS:
            outputs,codes=[],[]
            with torch.no_grad():
                for first in range(0,n,8):
                    out,code=prediction(arm,torch.arange(first,min(n,first+8)))
                    outputs.append(out.cpu())
                    if code is not None:codes.append(code.cpu())
            out=torch.cat(outputs);error=(out-targets).square()
            row['models'][arm]={'MSE':float(error.mean()),'MSE_over_conditional_floor':float(error.mean()/conditional_floor),
                                'D':float(L.objective(L.components(out[:,None],targets[:,None]),'D')),
                                'joint_codes':int(torch.cat(codes).unique(dim=0).shape[0]) if codes else None}
            L.atomic({'contract_id':cid,'update':update,'outputs':out,'codes':torch.cat(codes)if codes else None},DIRECTORY/f'{arm}_u{update:04d}.pt')
        history.append(row);print(json.dumps(row),flush=True)
    def save(update):
        L.atomic({'contract_id':cid,'update':update,'worlds':{k:L.cpu_state(v)for k,v in worlds.items()},
                  'channels':{k:L.cpu_state(v)for k,v in channels.items()},'optimizers':{k:copy.deepcopy(v.state_dict())for k,v in optimizers.items()},
                  'sampler':sampler.get_state(),'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all(),'history':history},state_path)
    if start==0:evaluate(0);save(0)
    for update in range(start,a.updates):
        indices=torch.randint(n,(a.batch,),generator=sampler)
        for arm in ARMS:
            out,_=prediction(arm,indices)
            loss=L.objective(L.components(out[:,None],targets[indices].cuda()[:,None]),'D')
            parameters=[p for group in optimizers[arm].param_groups for p in group['params']]
            optimizer_step(optimizers[arm],loss,parameters,learning_rate=_phase_lr(config,update),
                           grad_clip=config.agent.grad_clip,strict=True,zero_grad=True)
        if (update+1)%100==0 or update+1==a.updates:evaluate(update+1);save(update+1)
        elif (update+1)%25==0:save(update+1)
        if a.stop_after and update+1>=a.stop_after:
            save(update+1);return
    L.atomic({'contract_id':cid,'history':history,'scope':contract['scope']},DIRECTORY/'result.json',True)


if __name__=='__main__':main()
