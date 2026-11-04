"""Fresh, genuine paired observations: reject a fine reader that just keeps calling 'threat'."""
import os
os.environ.setdefault('JAX_PLATFORMS','cpu');os.environ.setdefault('MAMBA_DETERMINISTIC','1')
import importlib,json,sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from . import fit as A
N=importlib.import_module('artifacts.experiments.20261104_fine_causal_design.test_native')
HERE=Path(__file__).resolve().parent

@torch.no_grad()
def main():
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=True
    strong='--strong' in sys.argv
    m,cfg=N.load();feat=A.Features(A.load('dev'));saved=torch.load(HERE/'readers.pt',map_location='cpu',weights_only=False);nets={}
    control=json.loads((HERE/'coarse_control.json').read_text()) if strong else None
    if strong:saved.update(torch.load(HERE/'coarse_control_readers.pt',map_location='cpu',weights_only=False))
    for key,r in saved.items():
        if r['mode'] not in ['coarse','fine','coarse_tsp']:continue
        net=A.Reader(feat.dim,r['family']).to(A.DEV);net.load_state_dict(r['state']);nets[key]=net.eval()
    rows=torch.load(HERE/'consequences.pt',map_location='cpu',weights_only=False)['rows'];out=[]
    for i,r in enumerate(rows):
        meta=r['meta'];q=torch.tensor([meta['query']],device=A.DEV);f=meta['frame'];fx=feat.z['fix'][f:f+1].expand(2,-1);frames=torch.tensor(r['current'],device=A.DEV)
        with N.TR.autocast(A.DEV,cfg):
            P=m.perception(frames,fx);mean,_=m.pixels(m.tsp.predict(feat.z['prior'][f].float().expand(2,-1),P.p,P.v,N.M.window(fx)),P.v,N.M.window(fx))
        raw=F.adaptive_avg_pool2d(frames.reshape(-1,9,7,9,7,3).permute(0,1,3,5,2,4).reshape(-1,3,7,7).float(),3).round().byte().reshape(2,81,27)
        fields=dict(p=P.p.half(),g=P.g.half(),v=P.v.half(),raw=raw,tsp=mean.half(),fine_pixels=N.M.cell_pixels(N.M.crop(frames,fx)))
        old={k:feat.z[k][f].clone() for k in fields};parity={k:float((old[k].float()-v[0].float()).abs().max()) for k,v in fields.items()}
        probabilities={}
        for state in [0,1]:
            for k,v in fields.items():feat.z[k][f]=v[state]
            for key,net in nets.items():
                mode=saved[key]['mode']
                with torch.autocast('cuda',dtype=torch.bfloat16):x=net(feat.make(q,mode))[0,:5].float()
                cal=saved[key]['calibration']['kind'];p=(x/np.exp(cal['log_temperature'])+torch.tensor(cal['bias'],device=A.DEV)).softmax(-1)
                probabilities.setdefault(key,[]).append(p.cpu().tolist())
        for k,v in old.items():feat.z[k][f]=v
        families=[f'{family}_{seed}' for family in ['flat','split'] for seed in [0,1]]
        chosen={k:control[k+'_coarse_tsp']['chosen_by_task']['kind'] if strong else k+'_coarse' for k in families}
        factual_gain=[probabilities[k+'_fine'][0][meta['kind']]-probabilities[chosen[k]][0][meta['kind']] for k in families]
        factual_correct=[int(np.argmax(probabilities[k+'_fine'][0]))==meta['kind'] for k in families]
        alternate_correct=[int(np.argmax(probabilities[k+'_fine'][1]))==r['alternate_label'] for k in families]
        out.append(dict(meta=meta,alternate_label=r['alternate_label'],probabilities=probabilities,chosen_coarse=chosen,fresh_cache_maxdiff=parity,
            factual_gain=factual_gain,factual_correct=factual_correct,alternate_correct=alternate_correct,
            qualified=bool(min(factual_gain)>0 and all(factual_correct) and all(alternate_correct))))
        print('paired readers',i+1,'/',len(rows),'qualified',out[-1]['qualified'],flush=True)
    (HERE/('paired_readers_strong.json' if strong else 'paired_readers.json')).write_text(json.dumps(out,indent=2)+'\n')

if __name__=='__main__':main()
