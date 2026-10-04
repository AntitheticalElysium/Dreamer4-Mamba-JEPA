"""Controlled paired near-vs-far zombie direction in frozen encoder features."""
import json
import sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[3]
sys.path[:0]=[str(ROOT),str(ROOT/'artifacts/experiments/20260926_diagnosis')]
from twins import base_states,render_all,natural_frames,encoder_at,encode,inverse_sqrt_cov

def main():
    states=base_states()
    frames=render_all(states)
    natural=natural_frames()
    device=torch.device('cuda')
    report={'states':len(states),'steps':{}}
    for step in (0,10000):
        encoder=encoder_at(step,device)
        base=encode(encoder,frames['day','base'],device)
        near=encode(encoder,frames['day','zombie'],device)
        far=encode(encoder,frames['day','zombie_far'],device)
        nat=encode(encoder,natural,device)
        result={}
        for name,i in (('z',0),('cls',1),('right_patch',2)):
            b,n,f,natv=(v[i] for v in (base,near,far,nat))
            if name=='right_patch':
                b,n,f,natv=(v[:,3*9+5] for v in (b,n,f,natv))
            dn=(n-b).double(); df=(f-b).double(); delta=dn-df
            wh=inverse_sqrt_cov(natv)
            result[name]={
              'near_far_mean_cosine':float(torch.nn.functional.cosine_similarity(dn.mean(0)[None],df.mean(0)[None]).item()),
              'near_far_pair_cosine_mean':float(torch.nn.functional.cosine_similarity(dn,df).mean()),
              'near_far_fisher_dprime':float((wh@delta.mean(0)).norm()),
              'near_far_euclid_over_natural_radius':float(delta.norm(dim=1).mean()/(natv-natv.mean(0)).norm(dim=1).median()),
              'near_change_norm':float(dn.norm(dim=1).mean()),'far_change_norm':float(df.norm(dim=1).mean())}
        report['steps'][str(step)]=result
        print(step,json.dumps(result),flush=True)
        del encoder
        torch.cuda.empty_cache()
    Path(__file__).with_suffix('.json').write_text(json.dumps(report,indent=2)+'\n')
if __name__=='__main__': main()
