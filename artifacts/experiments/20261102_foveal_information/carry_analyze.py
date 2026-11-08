"""Matched readout metrics and episode-clustered intervals; no fitting."""
import json
from pathlib import Path
import numpy as np
import torch
HERE=Path(__file__).resolve().parent

def main():
 torch.set_num_threads(4)
 q=torch.load(HERE/'carry_pool.pt',weights_only=False);z=torch.load(HERE/'train_400.pt',weights_only=False)
 ix=q['ids'][q['validation']];y=z['truth'].flatten()[ix].numpy();ep=z['episode'][ix//63].numpy();ep=np.unique(ep,return_inverse=True)[1]
 inside=(z['localg'].flatten()[ix]>=0).numpy();night=z['night'][ix//63].numpy()
 logs={k:v.float().numpy() for k,v in torch.load(HERE/'carry_logp.pt',weights_only=False).items()};ans={}
 def metric(v,mask):
  E=ep.max()+1
  sums=np.bincount((ep*5+y)[mask],weights=v[mask],minlength=E*5).reshape(E,5)
  counts=np.bincount((ep*5+y)[mask],minlength=E*5).reshape(E,5)
  i=np.random.default_rng(611610).integers(E,size=(2000,E));n=counts[i].sum(1);ok=(n>0).all(1)
  boot=(sums[i].sum(1)/n.clip(min=1)).mean(1)
  return {'macro':float((sums.sum(0)/counts.sum(0)).mean()),'ci95':np.quantile(boot[ok],[.025,.975]).tolist(),'kind_counts':counts.sum(0).tolist()}
 for seed in (0,1):
  base=logs[f'{seed}_base'];bc=-np.take_along_axis(base,y[:,None],-1)[:,0];ba=(base.argmax(-1)==y).astype(float)
  for mode in ('base','fine','predicted','carry'):
   key=f'{seed}_{mode}';lp=logs[key];ce=-np.take_along_axis(lp,y[:,None],-1)[:,0];acc=(lp.argmax(-1)==y).astype(float)
   ans[key]={}
   for name,mask in [('inside',inside),('inside_night',inside&night),('outside',~inside)]:
    ans[key][name]={'ce':metric(ce,mask),'accuracy':metric(acc,mask),'ce_delta':metric(ce-bc,mask),'accuracy_delta':metric(acc-ba,mask)}
 (HERE/'carry_summary.json').write_text(json.dumps(ans,indent=1))
 for k,v in ans.items():print(k,v['inside']['accuracy'],v['inside']['accuracy_delta'])

if __name__=='__main__':main()
