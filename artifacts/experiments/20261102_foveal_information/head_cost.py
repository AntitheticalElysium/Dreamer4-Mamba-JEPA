"""Untrained cheap-head compute benchmark ONLY. Not a selection or P1 benefit test."""
import os
os.environ.setdefault('JAX_PLATFORMS','cpu');os.environ.setdefault('MAMBA_DETERMINISTIC','1')
import importlib,json,time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from saccade import model as M,data as D,train as TR
from saccade.eval import s1 as S
C=importlib.import_module('artifacts.experiments.20261102_foveal_information.conditional')
HERE=Path(__file__).resolve().parent;DEV=torch.device('cuda')
class Head(nn.Module):
 def __init__(self):
  super().__init__();self.net=nn.Sequential(nn.Linear(9*16+8+4*16+17+2,128),nn.GELU(),nn.Linear(128,128),nn.GELU(),nn.Linear(128,2))
  self.register_buffer('cells',M.window(torch.tensor([(r,c) for r in range(1,8) for c in range(1,8)])))
 def forward(self,p,v,h,a):
  win=p[:,self.cells].flatten(-2);ctx=torch.cat([v,h,a],-1)[:,None].expand(-1,49,-1)
  coords=torch.tensor([(r/8,c/8) for r in range(1,8) for c in range(1,8)],device=p.device)[None].expand(len(p),-1,-1)
  return self.net(torch.cat([win,ctx,coords],-1))
@torch.no_grad()
def main():
 torch.set_num_threads(4);torch.manual_seed(611620);m,cfg=C.U.load();head=Head().to(DEV).eval()
 L=S.first_episodes(dict(np.load(S.LABELS/'s0_dev_labels.npz')),32);eps=[e for e in S.episodes(L) if e.T>=64][:16]
 assert len(eps)==16
 obs=torch.cat([D.RP.episode_fields(e.shard,e.slot)['observations'][:64] for e in eps]).to(DEV)
 fix=torch.full((1024,2),4,device=DEV,dtype=torch.long)
 with TR.autocast(DEV,cfg):P=m.perception(obs,fix)
 # Timing-only context: zero history/actions. Arithmetic and shapes do not depend on labels.
 hist=torch.zeros(1024,64,device=DEV);act=torch.zeros(1024,17,device=DEV);act[:,0]=1
 def cost(which):
  def one():
   with TR.autocast(DEV,cfg):
    p=m.perception(obs,fix) if which=='extra_native_perception_and_head' else P
    head(p.p,p.v,hist,act)
  for _ in range(5):one()
  times=[]
  for _ in range(30):
   torch.cuda.synchronize();st=time.perf_counter();one();torch.cuda.synchronize();times.append(time.perf_counter()-st)
  return {'median_seconds_per_1024':float(np.median(times)),'min':min(times),'max':max(times),'repetitions':len(times)}
 ans={'head_parameters':sum(p.numel() for p in head.parameters()),'head_only':cost('head_only'),'extra_native_perception_and_head':cost('extra_native_perception_and_head'),
      'scope':'Untrained shape/compute benchmark; no selector accuracy, sparse teacher cost, head training or production collection ETA. Native full perception includes an unused dummy glimpse, giving one concrete extra-pass scenario.',
      'script_sha256':TR.file_sha256(__file__),'checkpoint_sha256':TR.file_sha256(C.U.CK)}
 (HERE/'head_cost.json').write_text(json.dumps(ans,indent=1));print(json.dumps(ans,indent=1))
if __name__=='__main__':main()
