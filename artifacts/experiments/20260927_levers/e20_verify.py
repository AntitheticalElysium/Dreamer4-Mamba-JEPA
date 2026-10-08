"""E20 source-only health reader and actual CUDA mechanics/restart verification."""
import argparse
import copy
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import e20_train as T
D,R=T.D,T.R


def reader():
    torch.set_num_threads(3)
    labels,mm=T.load_pool()
    spec={'version':'e20-factual-health63-ridge-v1','source':T.source_pins(),
          'pool':R.file_hash(D.OUT/'pool/labels.pt'),'lambda_grid':[.0001,.001,.01,.1,1],
          'feature':'health token63 only; normalized health0..1',
          'split':'sha256 episode ID modulo5; remainder0 holdout; not unseen from parent pretraining'}
    store=R.Store(D.OUT/'reader_fit',spec)
    with store.lock():
        if store.load('result') is not None:
            return
        x=torch.from_numpy(np.array(mm[:,[14,15],63,:])).float().reshape(-1,192)
        y=labels['health'][:,[14,15]].float().reshape(-1)/9
        val=torch.tensor([int(hashlib.sha256(eid.encode()).hexdigest(),16)%5==0
                          for eid,_ in labels['ids']]).repeat_interleave(2)
        gen=torch.Generator().manual_seed(20261007)
        tr=torch.where(~val)[0];va=torch.where(val)[0]
        tr=tr[torch.randperm(len(tr),generator=gen)[:16384]]
        va=va[torch.randperm(len(va),generator=gen)[:8192]]
        mean=x[tr].mean(0);std=x[tr].std(0).clamp_min(1e-4)
        xx=torch.cat([((x[tr]-mean)/std).double(),torch.ones(len(tr),1,dtype=torch.float64)],1)
        yy=y[tr].double()
        vv=torch.cat([((x[va]-mean)/std).double(),torch.ones(len(va),1,dtype=torch.float64)],1)
        gram=xx.T@xx;rhs=xx.T@yy
        best=None;selection=[]
        for lam in spec['lambda_grid']:
            coef=torch.linalg.solve(gram+lam*len(tr)*torch.eye(193,dtype=torch.float64),rhs)
            mae=float(((vv@coef)-y[va]).abs().mean()*9)
            selection.append({'lambda':lam,'heldout_mae_health':mae})
            if best is None or mae<best[0]:best=(mae,lam,coef)
        coef=best[2].float()
        rr={'mean':mean,'std':std,'weight':coef[:-1],'bias':coef[-1]}
        predicted=T.read_health(x,rr).reshape(-1,2)*9
        difference=predicted[:,1]-predicted[:,0]
        epval=val.reshape(-1,2)[:,0]
        damage=epval&(labels['classes']==1)
        unchanged=epval&(labels['classes']==2)
        catch=float((difference[damage]<-1.5).float().mean())
        false=float((difference[unchanged]<-1.5).float().mean())
        report={'contract':spec,'selection':selection,'lambda':best[1],
                'heldout_mae_health':best[0],'heldout_damage':int(damage.sum()),
                'heldout_damage_catch':catch,'heldout_unchanged':int(unchanged.sum()),
                'heldout_false_drops':false,'fit_frames':len(tr),'select_frames':len(va),
                'passed':best[0]<=.15 and catch>=.98 and false<=.01,
                'scope':'factual TRAIN source labels; no simulator/fork labels; inner episode holdout'}
        R.atomic_torch(D.OUT/'health_reader.pt',rr)
        report['reader_sha256']=R.file_hash(D.OUT/'health_reader.pt')
        store.save('result',report,1)
        R.atomic_json(D.OUT/'health_reader.json',report,immutable=True)
        print(json.dumps({k:v for k,v in report.items()if k not in ('contract','selection')}),flush=True)


def diff(a,b):
    if torch.is_tensor(a):
        return float((a.cpu()-b.cpu()).abs().max()) if a.numel() else 0.
    if isinstance(a,dict):
        assert a.keys()==b.keys()
        return max([diff(a[k],b[k])for k in a]+[0.])
    if isinstance(a,(list,tuple)):
        assert len(a)==len(b)
        return max([diff(x,y)for x,y in zip(a,b)]+[0.])
    assert a==b
    return 0.


def mechanics():
    torch.set_num_threads(4)
    labels,mm=T.load_pool()
    plans={a:T.schedule(labels,a)for a in 'ABC'}
    assert torch.equal(plans['B'][0],plans['C'][0])
    assert all(torch.equal(plans['A'][1],plans[a][1])for a in 'BC')
    actualA=labels['classes'][plans['A'][0]];actualB=labels['classes'][plans['B'][0]]
    assert torch.equal(actualA,actualB)
    for c in (0,2,3):
        assert torch.equal(plans['A'][0][actualA==c],plans['B'][0][actualB==c])
    device=torch.device('cuda')
    cfg=T.E.config()
    from d4mj.train import phase_optimizer,optimizer_step,autocast_context,_phase_lr
    rr=torch.load(D.OUT/'health_reader.pt',map_location=device,weights_only=False)
    torch.manual_seed(7);torch.cuda.manual_seed_all(7)
    model,parent=T.make_world(7,device)
    values=T.batch(mm,labels,plans['B'][0][0],int(plans['B'][1][0]),device)
    before=time.monotonic()
    with autocast_context(cfg):
        plain,pp=T.objective(model,values)
        augmented,ap=T.objective(model,values,rr)
    loss_identity=abs(float(augmented-plain)-ap['health'])
    assert loss_identity<=1e-6 and pp['teacher']==ap['teacher']
    augmented.backward()
    assert all(torch.isfinite(p.grad).all()for p in model.parameters()if p.grad is not None)
    model.zero_grad(set_to_none=True)
    src,target,actions,cls,dh=values
    altered_target=target+9
    with torch.no_grad(),autocast_context(cfg):
        p0=model(src,actions)[0][:,-1]
        # Future target is absent from the forward arguments.
        p1=model(src,actions)[0][:,-1]
        changed=src.clone();changed[:,3:]+=3
        achanged=actions.clone();achanged[:,3:]=(achanged[:,3:]+7)%17
        early0=model(src,actions)[0][:,:3]
        early1=model(changed,achanged)[0][:,:3]
    future_difference=float((early0-early1).abs().max())
    assert torch.equal(p0,p1) and future_difference<=1e-5
    assert altered_target.shape==target.shape
    del model,plain,augmented,p0,p1,early0,early1
    torch.cuda.empty_cache()
    long_model,_=T.make_world(7,device)
    with autocast_context(cfg):
        long_loss,_=T.objective(long_model,T.batch(mm,labels,plans['B'][0][0],15,device),rr)
    long_loss.backward()
    assert torch.isfinite(long_loss) and all(torch.isfinite(p.grad).all() for p in long_model.parameters() if p.grad is not None)
    del long_model,long_loss
    torch.cuda.empty_cache()
    parity={}
    # Actual full-batch CUDA update/serialization proof for all three arms.
    for arm in 'ABC':
        def run(end,state=None):
            torch.manual_seed(7);torch.cuda.manual_seed_all(7)
            world,_=T.make_world(7,device)
            opt=phase_optimizer([world],cfg)
            params=[p for g in opt.param_groups for p in g['params']]
            start=0
            if state is not None:
                world.load_state_dict(state['world']);opt.load_state_dict(state['optimizer'])
                R.restore_rng(state['rng'],device);start=state['update']
            ledger,lengths=plans[arm]
            timings=[]
            for u in range(start,end):
                tick=time.monotonic()
                v=T.batch(mm,labels,ledger[u],int(lengths[u]),device)
                with autocast_context(cfg):loss,_=T.objective(world,v,rr if arm=='C'else None)
                optimizer_step(opt,loss,params,learning_rate=_phase_lr(cfg,u),grad_clip=cfg.agent.grad_clip,
                               strict=True,zero_grad=True)
                torch.cuda.synchronize();timings.append(time.monotonic()-tick)
            path=D.OUT/f'verify_{arm}_{end}.pt'
            R.atomic_torch(path,{'world':world.state_dict(),'optimizer':opt.state_dict(),
                                'rng':R.rng_state(device),'update':end})
            saved=torch.load(path,map_location='cpu',weights_only=False)
            del world,opt;torch.cuda.empty_cache()
            return saved,timings
        uninterrupted,timing=run(4)
        part,_=run(2)
        resumed,_=run(4,part)
        parity[arm]={'world_max_abs':diff(uninterrupted['world'],resumed['world']),
                     'optimizer_max_abs':diff(uninterrupted['optimizer'],resumed['optimizer']),
                     'rng_max_abs':diff(uninterrupted['rng'],resumed['rng']),
                     'seconds_per_update':sum(timing[1:])/len(timing[1:])}
        assert max(parity[arm][k]for k in ('world_max_abs','optimizer_max_abs','rng_max_abs'))<=1e-6
        print(json.dumps({'stage':'resume_proof','arm':arm,**parity[arm]}),flush=True)
    report={'passed':True,'sources':T.source_pins(),'B_C_ledger_identical':True,
            'A_B_nonordinary_targets_identical':True,'class_row_balance_exact':True,
            'full_batch_long15_loss_and_gradients_finite':True,
            'loss_identity_abs':loss_identity,'future_perturbation_max_abs':future_difference,
            'serialized_resume':parity,'peak_gb':torch.cuda.max_memory_allocated()/1e9,
            'seconds':time.monotonic()-before}
    R.atomic_json(D.OUT/'mechanics.json',report,immutable=True)
    print(json.dumps(report),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['reader','mechanics']);a=p.parse_args()
    reader()if a.stage=='reader'else mechanics()
