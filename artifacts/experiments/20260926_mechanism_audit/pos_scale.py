"""Intervene only on trained ViT patch-position embedding strength, same paired scenes."""
import json,sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[3]
sys.path[:0]=[str(ROOT),str(ROOT/'artifacts/experiments/20260926_diagnosis')]
from twins import base_states,render_all,natural_frames,encoder_at,encode,inverse_sqrt_cov

def main():
 states=base_states(); frames=render_all(states); natural=natural_frames(); device=torch.device('cuda')
 encoder=encoder_at(10000,device)
 pos=encoder.backbone.embeddings.position_embeddings
 original=pos.data.clone()
 out={'states':len(states),'scales':{}}
 for scale in (0,1,2,4,8):
  pos.data.copy_(original)
  pos.data[:,1:].mul_(scale)
  nat=encode(encoder,natural,device)
  result={}
  for light in ('day','night'):
   base=encode(encoder,frames[light,'base'],device)
   near=encode(encoder,frames[light,'zombie'],device)
   far=encode(encoder,frames[light,'zombie_far'],device)
   vals={}
   for name,i in (('cls',1),('z',0)):
    b,n,f,natv=(v[i] for v in (base,near,far,nat))
    dn=(n-b).double(); df=(f-b).double(); contrast=dn-df
    wh=inverse_sqrt_cov(natv)
    vals[name]={'near_far_fisher_dprime':float((wh@contrast.mean(0)).norm()),
                'near_far_pair_cosine':float(torch.nn.functional.cosine_similarity(dn,df).mean()),
                'contrast_over_near':float(contrast.norm(dim=1).mean()/dn.norm(dim=1).mean())}
   result[light]=vals
  out['scales'][str(scale)]=result
  print(scale,json.dumps(result),flush=True)
 pos.data.copy_(original)
 Path(__file__).with_suffix('.json').write_text(json.dumps(out,indent=2)+'\n')
if __name__=='__main__':main()
