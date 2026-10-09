"""Exploratory CPU check: does E22M prior's frame input retain zombie location?

Compare raw tile192, prior tile8 and global frame512 using grouped ridge readouts.
All models frozen, same five true-frame depths, roots split by episode/seed.
These are finite linear readouts on opened diagnostics, not information ceilings.
No world training, posterior encoding or CUDA use; outputs into the existing health_chain_v1 EDA store.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['OMP_NUM_THREADS'] = '2'
os.environ['MKL_NUM_THREADS'] = '2'
import sys, json, hashlib, time
from pathlib import Path
import torch
torch.set_num_threads(2)
torch.set_num_interop_threads(1)
ROOT = Path('/home/antithetical/EPITA/PERSO/DynamicHorizons-Mamba-JEPA')
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT/'artifacts/experiments/20260927_levers'))
import dworld as DW

def fitted(x,y,tr,va,te):
    mu,sd=x[tr].mean(0),x[tr].std(0).clamp_min(1e-6)
    z=(x-mu)/sd
    z=torch.cat([z,torch.ones(len(z),1)],1).double()
    if y.ndim==1:y=y[:,None]
    yy=y.double()
    A=z[tr].T@z[tr]; B=z[tr].T@yy[tr]
    eye=torch.eye(z.shape[1],dtype=torch.double)
    best=None
    for lam in (.001,.01,.1,1.,10.):
        w=torch.linalg.solve(A+lam*tr.sum()*eye,B)
        loss=float(((z[va]@w-yy[va])**2).mean())
        if best is None or loss<best[0]:best=(loss,lam,w)
    return (z[te]@best[2]).float(),best[1]

def metrics(y,p):
    y,p=y.flatten(),p.flatten()
    positives=int(y.sum()); negatives=len(y)-positives
    _,order=p.sort(descending=True)
    scores,ys=p[order],y[order]
    _,counts=torch.unique_consecutive(scores,return_counts=True)
    ends=counts.cumsum(0)-1
    tp=ys.cumsum(0)[ends]; fp=ends+1-tp
    tpr=torch.cat([torch.zeros(1),tp/max(1,positives)])
    fpr=torch.cat([torch.zeros(1),fp/max(1,negatives)])
    auc=float(torch.trapezoid(tpr,fpr))
    ap=float(((tp-torch.cat([torch.zeros(1),tp[:-1]]))/max(1,positives)*tp/(ends+1)).sum())
    return {'auc':auc, 'ap':ap,'brier_raw':float(((p-y)**2).mean()),'positives':positives,'n':len(y)}

start=time.time()
meta_path=ROOT/'artifacts/eda/diagnosis_rollouts_v1/meta.pt'
cache_path=ROOT/'artifacts/eda/levers_futures_tokens_raw.pt'
meta=torch.load(meta_path,map_location='cpu',weights_only=False,mmap=True)
cache=torch.load(cache_path,map_location='cpu',weights_only=False,mmap=True)
depths=[0,1,4,8,16]
frames=torch.cat([cache['ctx'][:,-1,None],cache['fut']],1)[:,depths].float()
vis=torch.cat([meta['root_visible'][:,None],meta['future_visible'][:,0]],1)[:,depths]
y=(vis[...,1071:1512].reshape(len(vis),len(depths),63,7)[...,0]>0).float()
alive=torch.cat([torch.ones(len(vis),1,dtype=torch.bool),~meta['future_dead'][:,0].cumsum(1).bool()],1)[:,depths]
groups=meta['seed'][:,None].expand(-1,len(depths))[alive]
steps=torch.tensor(depths)[None].expand(len(vis),-1)[alive]
frames=frames[alive]; y=y[alive]
u=groups.unique(); u=u[torch.randperm(len(u),generator=torch.Generator().manual_seed(0))]
train=u[:int(.7*len(u))]; val=train[-max(1,len(train)//5):]; fit=train[:-max(1,len(train)//5)]
tr,va,te=torch.isin(groups,fit),torch.isin(groups,val),~torch.isin(groups,train)
trc,vac,tec=[m[:,None].expand(-1,63).flatten() for m in (tr,va,te)]
report={'protocol':__doc__,'depths':depths,'frames':len(frames),'seed_counts':{'fit':len(fit),'val':len(val),'test':len(u)-len(train)},
        'split_seed':0,'readouts':{},'checkpoints':{},'cuda_available':torch.cuda.is_available()}
pred,lam=fitted(frames[:,:63].flatten(0,1),y.flatten(),trc,vac,tec)
report['readouts']['raw_tile192']={**metrics(y[te],pred),'lambda':lam}
print(json.dumps({'stage':'raw','result':report['readouts']['raw_tile192']}),flush=True)
near=[22,30,32,40]
for updates in (20000,60000):
    path=ROOT/f'artifacts/eda/levers_tworlds_v1/dworld_a_s7_from2000_revalways_mwauto_L16_di_b16_u2000_prior_u{updates}.pt'
    if not path.exists():continue
    payload=torch.load(path,map_location='cpu',weights_only=False,mmap=True)
    prior=DW.Prior().eval();prior.load_state_dict(payload['prior'])
    with torch.no_grad():
        local=prior.tile(frames)
        glob=prior.frame(local.flatten(-2))
    pp,lam=fitted(local[:,:63].flatten(0,1),y.flatten(),trc,vac,tec)
    report['readouts'][f'tile8_{updates}']={**metrics(y[te],pp),'lambda':lam}
    pp,lam=fitted(glob,y,tr,va,te)
    item={**metrics(y[te],pp),'lambda':lam,'near':metrics(y[te][:,near],pp[:,near]),'by_depth':{}}
    for d in depths:
        m=steps[te]==d
        item['by_depth'][str(d)]=metrics(y[te][m],pp[m])
    report['readouts'][f'frame512_{updates}']=item
    history=payload.get('history',[])
    report['checkpoints'][str(updates)]={'path':str(path),'bytes':path.stat().st_size,'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
        'history_last':history[-1] if history else None,'stage_a':payload.get('stage_a')}
    print(json.dumps({'stage':updates,'tile':report['readouts'][f'tile8_{updates}'],'frame':item}),flush=True)
report['seconds']=time.time()-start
Path('/home/antithetical/EPITA/PERSO/DynamicHorizons-Mamba-JEPA/artifacts/eda/health_chain_v1/d4mj_prior_input_probe_20261009.json').write_text(json.dumps(report,indent=2))
print(json.dumps({'complete':True,'seconds':report['seconds']}),flush=True)
