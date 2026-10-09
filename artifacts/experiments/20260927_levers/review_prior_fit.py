"""Exploratory CPU FP32 prior fit: 20k vs60k, same frozen posterior labels.
Retains per-root/per-future code scores; contextual deterministic information
in each code means these are code-fit strata, not decoded move probabilities.
CUDA disabled, two threads. Outputs into the existing health_chain_v1 EDA store.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['OMP_NUM_THREADS']='2';os.environ['MKL_NUM_THREADS']='2'
import sys,json,time,hashlib,argparse
from pathlib import Path
import torch
import torch.nn.functional as F
torch.set_num_threads(2);torch.set_num_interop_threads(1)
ROOT=Path('/home/antithetical/EPITA/PERSO/DynamicHorizons-Mamba-JEPA')
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'artifacts/experiments/20260927_levers'))
import dworld as DW
import scroll
parser=argparse.ArgumentParser();parser.add_argument('--reuse-rows',action='store_true');args=parser.parse_args()
start=time.time();home=ROOT/'artifacts/eda/levers_tworlds_v1'
base='dworld_a_s7_from2000_revalways_mwauto_L16_di_b16_u2000'
payloads={u:torch.load(home/f'{base}_prior_u{u}.pt',map_location='cpu',mmap=True,weights_only=False) for u in (20000,60000)}
stage=ROOT/payloads[20000]['stage_a']
assert payloads[20000]['stage_a']==payloads[60000]['stage_a']
sta=torch.load(stage,map_location='cpu',mmap=True,weights_only=False)
post=DW.Posterior().eval();post.load_state_dict(sta['post'])
prior={u:DW.Prior().eval() for u in payloads}
for u in prior:prior[u].load_state_dict(payloads[u]['prior'])
cache=torch.load(ROOT/'artifacts/eda/levers_futures_tokens_raw.pt',map_location='cpu',mmap=True,weights_only=False)
meta=torch.load(ROOT/'artifacts/eda/diagnosis_rollouts_v1/meta.pt',map_location='cpu',mmap=True,weights_only=False)
R=len(meta['seed']);frames=torch.cat([cache['ctx'],cache['fut']],1).float();acts=torch.cat([cache['ctx_a'],cache['fut_a']],1)
ys=torch.empty(R,19,4,dtype=torch.long)
ce={u:torch.empty(R,19,4) for u in prior};correct={u:torch.empty(R,19,4,dtype=torch.bool) for u in prior}
conf={u:torch.empty(R,19,4) for u in prior};entropy={u:torch.empty(R,19,4) for u in prior}
if args.reuse_rows:
    previous=torch.load('/home/antithetical/EPITA/PERSO/DynamicHorizons-Mamba-JEPA/artifacts/eda/health_chain_v1/d4mj_prior_conditional_rows_20261009.pt',map_location='cpu',weights_only=False)
    ys,ce,correct,conf,entropy=[previous[k] for k in ('targets','ce','correct','confidence','entropy')]
else:
    with torch.no_grad():
        for i in range(0,R,8):
            s=frames[i:i+8];a=acts[i:i+8]
            labels=post.quantizer(post.encode(s[:,:-1].flatten(0,1),a.flatten(),s[:,1:].flatten(0,1)))[1].view(len(s),19,4)
            ys[i:i+len(s)]=labels
            for u,p in prior.items():
                logits=p(s,F.pad(a,(0,1)),F.pad(labels,(0,0,0,1)))[0][:,:-1].float()
                prob=logits.softmax(-1)
                ce[u][i:i+len(s)]=F.cross_entropy(logits.reshape(-1,DW.CODES),labels.flatten(),reduction='none').view_as(labels)
                correct[u][i:i+len(s)]=logits.argmax(-1)==labels
                conf[u][i:i+len(s)]=prob.max(-1).values
                entropy[u][i:i+len(s)]=-(prob*prob.clamp_min(1e-30).log()).sum(-1)
            if i%128==0:print(json.dumps({'roots':i+len(s),'seconds':round(time.time()-start,1)}),flush=True)
vis=torch.cat([meta['root_visible'][:,None],meta['future_visible'][:,0]],1).float()
z=(vis[...,1071:1512].reshape(R,17,7,9,7)[...,0]>0)
n=z.flatten(2).sum(-1);alive=~meta['future_dead'][:,0].cumsum(1).bool()
froot=torch.cat([cache['ctx'][:,-1,None],cache['fut']],1).float()
sc=torch.cat([scroll.estimate(froot[i:i+16,:-1].flatten(0,1),froot[i:i+16,1:].flatten(0,1)).view(-1,16) for i in range(0,R,16)])
ix0=z[:,:-1].flatten(2).float().argmax(-1);ix1=z[:,1:].flatten(2).float().argmax(-1)
v0=torch.stack([ix0//9,ix0%9],-1);v1=torch.stack([ix1//9,ix1%9],-1)
move=v1-v0;rel=v0-torch.tensor([3,4]);dist=rel.abs().sum(-1)
eligible=(n[:,:-1]==1)&(n[:,1:]==1)&alive&(sc==0)&(dist>=1)&(dist<=6)&(move.abs().sum(-1)<=1)
stay=(move==0).all(-1)
amax=rel.abs().max(-1,keepdim=True).values
toward=(move.abs().sum(-1)==1)&((move*-rel.sign())==1).any(-1)&((move!=0)&(rel.abs()==amax)).any(-1)
kind={'all_no_scroll_single_zombie':eligible,'stay':eligible&stay,'toward':eligible&toward,'other_move':eligible&~stay&~toward}
reg=(v1[...,0]//5)*2+(v1[...,1]//5)
future={u:{'ce':ce[u][:,3:],'correct':correct[u][:,3:],'conf':conf[u][:,3:],'entropy':entropy[u][:,3:]} for u in prior}
out={'protocol':__doc__,'roots':R,'seed_clusters':len(meta['seed'].unique()),'code_count':ys.numel(),'cpu_dtype':'float32',
     'all':{},'strata':{},'stage_a':str(stage),'cuda_available':torch.cuda.is_available()}
def summary(cs,ok,cf,en):
    return {'n':cs.numel(),'ce':float(cs.mean()),'accuracy':float(ok.float().mean()),'mean_confidence':float(cf.mean()),'entropy':float(en.mean())}
for u in prior:out['all'][str(u)]=summary(ce[u],correct[u],conf[u],entropy[u])
for name,mask in kind.items():
    ri,ki=torch.where(mask);rg=reg[ri,ki]
    out['strata'][name]={'transitions':len(ri),'priors':{}}
    for u in prior:
        vals=future[u]
        out['strata'][name]['priors'][str(u)]=summary(*[vals[key][ri,ki,rg] for key in ('ce','correct','conf','entropy')])
out['seconds']=time.time()-start
torch.save({'seed':meta['seed'],'targets':ys,'ce':ce,'correct':correct,'confidence':conf,'entropy':entropy,'masks':kind,'region':reg},'/home/antithetical/EPITA/PERSO/DynamicHorizons-Mamba-JEPA/artifacts/eda/health_chain_v1/d4mj_prior_conditional_rows_20261009.pt')
Path('/home/antithetical/EPITA/PERSO/DynamicHorizons-Mamba-JEPA/artifacts/eda/health_chain_v1/d4mj_prior_conditional_fit_20261009.json').write_text(json.dumps(out,indent=2))
print(json.dumps(out,indent=1),flush=True)
