"""Matched near/far terrain edit to test whether CLS spatial blindness is zombie-specific."""
import json,sys
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[3]
sys.path[:0]=[str(ROOT),str(ROOT/'artifacts/experiments/20260926_diagnosis')]
from twins import base_states,render_all,natural_frames,encoder_at,encode,inverse_sqrt_cov
from d4mj.env import _env
BLOCK={'lava':14,'water':3,'stone':4}
def main():
 states=base_states(); frames=render_all(states); natural=natural_frames(); env,_=_env(); obs=jax.jit(env.get_obs)
 device=torch.device('cuda'); e=encoder_at(10000,device); nat=encode(e,natural,device)
 report={'states':len(states),'edits':{}}
 for edit,block in BLOCK.items():
  far=[]
  for s,right,pos in states:
   t=s.replace(map=s.map.at[pos[0],pos[1]].set(block),light_level=jnp.asarray(1.,jnp.float32))
   far.append(np.asarray(obs(t)))
  far=torch.from_numpy(np.stack(far)*255).round().to(torch.uint8)
  b=encode(e,frames['day','base'],device); n=encode(e,frames['day',edit],device); f=encode(e,far,device)
  row={}
  for name,i in (('z',0),('cls',1),('right_patch',2)):
   bb,nn,ff,v=(x[i] for x in (b,n,f,nat))
   if name=='right_patch':bb,nn,ff,v=(x[:,3*9+5] for x in (bb,nn,ff,v))
   dn=(nn-bb).double();df=(ff-bb).double(); wh=inverse_sqrt_cov(v)
   row[name]={'near_far_mean_cosine':float(torch.nn.functional.cosine_similarity(dn.mean(0)[None],df.mean(0)[None]).item()),
              'near_far_fisher_dprime':float((wh@(dn-df).mean(0)).norm()),
              'near_fisher_dprime':float((wh@dn.mean(0)).norm())}
  report['edits'][edit]=row
  print(edit,json.dumps(row),flush=True)
 Path(__file__).with_suffix('.json').write_text(json.dumps(report,indent=2)+'\n')
if __name__=='__main__':main()
