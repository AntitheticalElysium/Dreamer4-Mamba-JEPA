"""Matched observation-only innovation feasibility; atomic resume and source pins."""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import sys
import time

os.environ.setdefault('TRITON_F32_DEFAULT','ieee')
assert os.environ['TRITON_F32_DEFAULT']=='ieee'

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0,str(Path(__file__).parent))
import loss_intervention as L
import innovation_model as I
from d4mj.config import config_from_dict
from d4mj.train import autocast_context,optimizer_step,phase_optimizer,_phase_lr

OUT=L.OUT/'learned_innovation'
INIT=L.OUT/'worlds/control_deltairis_from_corrt_rawlong_teacher_s7_fmamba_di_L16b40_from36000_u3000_b16.pt'
ARMS=('control','capacity','rate')


def log(**row):
    print(json.dumps(row),flush=True)


def set_lr(opt, lr):
    for group in opt.param_groups:
        group['lr']=lr


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--updates',type=int,default=3000)
    parser.add_argument('--batch',type=int,default=16)
    parser.add_argument('--save-every',type=int,default=100)
    parser.add_argument('--stop-after',type=int)
    parser.add_argument('--directory',type=Path,default=OUT)
    args=parser.parse_args()
    assert args.updates>0 and args.batch>0 and args.save_every>0
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.manual_seed(7);torch.cuda.manual_seed_all(7)
    config=config_from_dict(torch.load(L.TW.S.CHECKPOINT,map_location='cpu',weights_only=False)['config'])
    pooldir=L.TW.POOLS['rawlong']
    manifest=json.loads((pooldir/'manifest.json').read_text())
    source=[Path(__file__),Path(I.__file__),Path(L.__file__),Path(L.TW.__file__),Path(L.TW.S.__file__),
            L.ROOT/'d4mj/train.py',L.ROOT/'d4mj/mamba_recurrence.py',Path(L.TW.S.CHECKPOINT),INIT,
            pooldir/'manifest.json',pooldir/'labels.pt']
    contract={'format':'learned_innovation_feasibility_v1','sources_inputs':{str(p.relative_to(L.ROOT)):L.sha(p) for p in source},
              'tokens_sha256':manifest['tokens_sha256'],'updates':args.updates,'batch':args.batch,
              'arms':ARMS,'init':'unconditioned D control; zero unused oracle parameters verified',
              'features':'frozen Raw 81x192 tokens','context':16,'targets':'all15next states',
              'channel':{'slots':I.SLOTS,'choices':I.CHOICES,'maximum_bits':18,'future_bypass':False},
              'rate_penalty':{'control':0,'capacity':0,'rate':.001},
              'completion_loss':'D = MSE+0.1L1+0.01mean frame maximum squared error',
              'prior':'native causal spatial/Mamba features, autoregressive factor likelihood; detached targets and features',
              'posterior':'observed current/next token pair + native causal features; hard argmax ST',
              'sampler_seed':11,'initialization_seed':7,'precision':'BF16/IEEE, TF32 disabled',
              'training_labels':'observed actions and terminal window metadata only; no layouts, events, classes or keys',
              'scope':'continuation feasibility; one seed; no claim of entity identification or four-way separation'}
    contract=json.loads(json.dumps(contract))
    cid=hashlib.sha256(json.dumps(contract,sort_keys=True).encode()).hexdigest()
    directory=args.directory;directory.mkdir(parents=True,exist_ok=True)
    cp=directory/'contract.json'
    if cp.exists():assert json.loads(cp.read_text())==contract,'source/input contract drift'
    else:L.atomic(contract,cp,True)
    pool=torch.load(pooldir/'labels.pt',map_location='cpu',weights_only=False,mmap=True)
    assert L.sha(pooldir/'labels.pt')==manifest['labels_sha256']
    tokenpath=pooldir/'tokens.f16';stat=tokenpath.stat()
    fingerprint={'sha256':manifest['tokens_sha256'],'size':stat.st_size,'mtime_ns':stat.st_mtime_ns}
    verified=L.OUT/'generated_prefix/token_verification.json'
    assert json.loads(verified.read_text())==fingerprint,'verified training tokens changed'
    L.atomic(fingerprint,directory/'token_verification.json',True)
    tokens=np.memmap(tokenpath,mode='r',dtype=np.float16,shape=(len(pool['terminal']),64,81,192))
    main_rows=torch.where(~pool['terminal'])[0];term_rows=torch.where(pool['terminal'])[0]
    parent=torch.load(INIT,map_location='cpu',weights_only=False)
    assert parent['args']['arm']=='control'
    assert all(torch.count_nonzero(v)==0 for v in parent['oracle'].values()),'parent is privileged-conditioned'
    channel_template=I.Innovation()
    worlds,channels,optimizers,prior_optimizers={},{},{},{}
    for arm in ARMS:
        world=L.TW.TWorld('corrt',None,'fmamba','all',0,False,16)
        world.load_state_dict(parent['world'])
        worlds[arm]=world.to('cuda').train()
        channels[arm]=copy.deepcopy(channel_template).to('cuda').train()
        parameters=list(world.parameters())+(channels[arm].completion_parameters() if arm!='control' else [])
        optimizers[arm]=phase_optimizer([world],config)
        if arm!='control':
            optimizers[arm].add_param_group({'params':channels[arm].completion_parameters(),
                                           'lr':optimizers[arm].param_groups[0]['lr'],
                                           'weight_decay':optimizers[arm].param_groups[0]['weight_decay']})
        prior_optimizers[arm]=torch.optim.AdamW(channels[arm].prior.parameters(),lr=1e-4,weight_decay=.01)
    assert all(L.restored_equal(L.cpu_state(worlds['control']),L.cpu_state(worlds[arm]))for arm in ARMS)
    order=torch.Generator().manual_seed(11)
    history=[];chain='initial';start=0
    summary={arm:{'steps':0,'compute_seconds':0.,'clipped':0,'targets':0}for arm in ARMS}
    state=directory/'state.pt'
    if state.exists():
        saved=torch.load(state,map_location='cpu',weights_only=False)
        assert saved['contract_id']==cid
        for arm in ARMS:
            worlds[arm].load_state_dict(saved['worlds'][arm]);channels[arm].load_state_dict(saved['channels'][arm])
            optimizers[arm].load_state_dict(saved['optimizers'][arm]);prior_optimizers[arm].load_state_dict(saved['prior_optimizers'][arm])
            assert L.restored_equal(L.cpu_state(worlds[arm]),saved['worlds'][arm])
            assert L.restored_equal(L.cpu_state(channels[arm]),saved['channels'][arm])
            assert L.restored_equal(optimizers[arm].state_dict(),saved['optimizers'][arm])
            assert L.restored_equal(prior_optimizers[arm].state_dict(),saved['prior_optimizers'][arm])
        order.set_state(saved['order']);torch.set_rng_state(saved['torch_rng']);torch.cuda.set_rng_state_all(saved['cuda_rng'])
        start,history,summary,chain=saved['update'],saved['history'],saved['summary'],saved['ledger_chain']
        log(stage='resume',update=start,all_serialized_state_exact=True)

    def save(update):
        L.atomic({'contract_id':cid,'update':update,'worlds':{a:L.cpu_state(w)for a,w in worlds.items()},
                  'channels':{a:L.cpu_state(c)for a,c in channels.items()},
                  'optimizers':{a:copy.deepcopy(o.state_dict())for a,o in optimizers.items()},
                  'prior_optimizers':{a:copy.deepcopy(o.state_dict())for a,o in prior_optimizers.items()},
                  'order':order.get_state(),'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all(),
                  'history':history,'summary':summary,'ledger_chain':chain},state)

    def snapshots(update):
        for arm in ARMS:
            L.atomic({'contract_id':cid,'contract':contract,'arm':arm,'update':update,
                      'world':L.cpu_state(worlds[arm]),'channel':L.cpu_state(channels[arm]),
                      'ledger_chain':chain},directory/f'{arm}_u{update:04d}.pt')

    if start==0:snapshots(0)
    begin=time.time()
    for update in range(start,args.updates):
        terminal=torch.rand(args.batch,generator=order)<L.TW.TERMINAL_SHARE
        rows=torch.where(terminal,term_rows[torch.randint(len(term_rows),(args.batch,),generator=order)],
                         main_rows[torch.randint(len(main_rows),(args.batch,),generator=order)])
        starts=torch.where(terminal,48,torch.randint(0,49,(args.batch,),generator=order))
        raw=rows.numpy().tobytes()+starts.numpy().tobytes()
        chain=hashlib.sha256((b'' if chain=='initial'else bytes.fromhex(chain))+raw).hexdigest()
        frames=torch.stack([torch.from_numpy(np.array(tokens[int(r),int(s):int(s)+16]))for r,s in zip(rows,starts)]).to('cuda').float()
        actions=torch.stack([pool['actions'][int(r),int(s):int(s)+15]for r,s in zip(rows,starts)]).to('cuda')
        padded=F.pad(actions,(0,1));records={}
        for arm in ARMS:
            when=time.perf_counter();world,channel=worlds[arm],channels[arm]
            beta=.001 if arm=='rate'else 0.
            rate=prior_loss=None;code_stats={}
            if arm!='control':
                belief=I.native_belief(world,frames,padded,config)
                q=channel.posterior(belief[:,:15],frames[:,:15],frames[:,1:])
                weights,codes=channel.hard(q)
                logits,_=channel.prior(belief[:,:15],codes.detach())
                rate=I.frame_rate(logits.detach(),weights)
                prior_loss=F.cross_entropy(logits.reshape(-1,I.CHOICES),codes.detach().reshape(-1))
                delta=F.pad(channel.condition(belief[:,:15],weights),(0,0,0,0,0,1))
                if (update+1)%100==0 or update==start:
                    probabilities=q.float().softmax(-1)
                    code_stats={'assignment_nll_bits':float(rate.detach()/np.log(2)),
                                'soft_entropy_bits':float(-(probabilities*probabilities.clamp_min(1e-9).log()).sum(-1).sum(-1).mean()/np.log(2)),
                                'used_per_factor':[int(codes[...,i].unique().numel())for i in range(I.SLOTS)]}
            else:delta=None
            with I.conditioned(world,delta),autocast_context(config):
                prediction=world(frames,padded)[0][:,:15]
                c=L.components(prediction,frames[:,1:])
                completion=L.objective(c,'D')
                loss=completion+(beta*rate if rate is not None else 0.)
            parameters=[p for group in optimizers[arm].param_groups for p in group['params']]
            norm=float(optimizer_step(optimizers[arm],loss,parameters,learning_rate=_phase_lr(config,update),
                                      grad_clip=config.agent.grad_clip,strict=True,zero_grad=True))
            if prior_loss is not None:
                prior_optimizers[arm].zero_grad(set_to_none=True);prior_loss.backward()
                torch.nn.utils.clip_grad_norm_(channel.prior.parameters(),1.)
                prior_optimizers[arm].step();prior_optimizers[arm].zero_grad(set_to_none=True)
            elapsed=time.perf_counter()-when
            summary[arm]['steps']+=1;summary[arm]['compute_seconds']+=elapsed
            summary[arm]['clipped']+=int(norm>config.agent.grad_clip);summary[arm]['targets']+=args.batch*15
            records[arm]={'D':float(completion.detach()),'components':{k:float(v.detach())for k,v in c.items()},
                          'gradient_norm':norm,'seconds':elapsed,**code_stats}
        if (update+1)%100==0 or update==start:
            row={'update':update+1,'arms':records,'ledger_chain':chain,'active_seconds':time.time()-begin,
                 'gpu_peak_gb':torch.cuda.max_memory_allocated()/2**30}
            history.append(row);log(stage='train',**row)
        if (update+1)%args.save_every==0 or update+1==args.updates:save(update+1)
        if update+1 in (500,1500,args.updates):snapshots(update+1)
        if args.stop_after and update+1>=args.stop_after:
            save(update+1);log(stage='interruption_check',update=update+1);return
    L.atomic({'contract_id':cid,'update':args.updates,'history':history,'summary':summary,'ledger_chain':chain},directory/'training.json',True)
    log(stage='done',update=args.updates,summary=summary)


if __name__=='__main__':main()
