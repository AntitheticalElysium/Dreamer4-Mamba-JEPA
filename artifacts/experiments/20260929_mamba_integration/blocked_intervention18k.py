"""Diagnostic do-intervention: remove direction-specific copied source on visible-rule-blocked held-out moves.

Uses visible-rule blocked labels only to localize loss, never for a trained model or a deployable score.
"""
import json,sys,hashlib
from pathlib import Path
import torch
from torch.nn import functional as F
ROOT=Path('/home/antithetical/EPITA/PERSO/DynamicHorizons-Mamba-JEPA')
sys.path[:0]=[str(ROOT),str(ROOT/'artifacts/experiments/20260927_levers'),str(ROOT/'artifacts/experiments/20260926_diagnosis'),str(ROOT/'artifacts/experiments/20260921_readout_ladder')]
import teval as T
import tworld as W
from onestep import classify
from d4mj.config import config_from_dict
from d4mj.train import autocast_context
import spatial as S
SCROLL=torch.tensor([0,3,4,1,2]); N=17
@torch.no_grad()
def run(pool):
 device=torch.device('cuda');tag=f'int_corrg_{pool}_suffix_s7_fmamba_u18000';path=ROOT/'artifacts/eda/levers_mamba_integration_v1'/f'{tag}.pt'
 world,_=T.load_world(path,device);meta,tr,_=T.split();cl,_=classify(meta);cache=T.build_cache(pool,device)
 config=config_from_dict(torch.load(S.CHECKPOINT,map_location='cpu',weights_only=False)['config'])
 rows=[(i,a) for i in torch.where(~tr)[0].tolist() for a in range(1,5) if int(cl[i,a])==1]
 actual=[];cut=[];copy=[];maxdiff=0.
 with autocast_context(config):
  for start in range(0,len(rows),8):
   rr=rows[start:start+8];ii=torch.tensor([i for i,a in rr]);aa=torch.tensor([a for i,a in rr]);b=len(rr)
   s=cache['ctx'][ii].float().to(device);act=torch.cat([cache['ctx_a'][ii],aa[:,None]],1).to(device)
   p=world(s,act)[0][:,-1].float();w=world.last_weights[:,-1].float().clone();gen=world.last_generated[:,-1].float()
   grid=s[:,-1].float().view(b,9,9,192);pad=F.pad(grid,(0,0,1,1,1,1));cands=[grid]+[pad[:,1+dr:10+dr,1+dc:10+dc] for dr,dc in W.NEIGHBOURS]
   cands=torch.stack([x.reshape(b,81,192) for x in cands]+[gen],-2)
   reconstruct=F.layer_norm((w[...,None]*cands).sum(-2),(192,));maxdiff=max(maxdiff,float((reconstruct-p).abs().max()))
   ix=SCROLL[aa].to(device);w[torch.arange(b,device=device)[:,None],torch.arange(81,device=device)[None,:],ix[:,None]]=0
   w=w/w.sum(-1,keepdim=True).clamp_min(1e-10)
   alt=F.layer_norm((w[...,None]*cands).sum(-2),(192,))
   true=cache['one'][ii,aa].float().to(device);root=s[:,-1]
   actual.extend(((p-true)**2).sum((-1,-2)).cpu().tolist());cut.extend(((alt-true)**2).sum((-1,-2)).cpu().tolist());copy.extend(((root-true)**2).sum((-1,-2)).cpu().tolist())
 return {'checkpoint_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'n_test_blocked':len(rows),'reconstruction_max_abs':maxdiff,'actual_over_copy':sum(actual)/sum(copy),'cut_direction_scroll_over_copy':sum(cut)/sum(copy),'relative_error_change':sum(cut)/sum(actual)-1}
if __name__=='__main__':
 out={pool:run(pool) for pool in ('raw','ldad1','ldad10')};p=ROOT/'artifacts/experiments/20260929_mamba_integration/blocked_intervention18k.json';p.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
