"""Frozen 18k candidate-supply versus routing diagnosis on visible-rule-blocked TEST moves.

The oracle selects the lowest-error single source per token using the true successor.
Mixtures can outperform a single source, so this is a diagnostic, not a lower bound or deployable model.
"""
import hashlib
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT=Path('/home/antithetical/EPITA/PERSO/DynamicHorizons-Mamba-JEPA')
sys.path[:0]=[str(ROOT),str(ROOT/'artifacts/experiments/20260927_levers'),
              str(ROOT/'artifacts/experiments/20260926_diagnosis'),
              str(ROOT/'artifacts/experiments/20260921_readout_ladder')]
import teval as T
import tworld as W
from onestep import classify
from d4mj.config import config_from_dict
from d4mj.train import autocast_context
import spatial as S


@torch.no_grad()
def run(pool):
    device=torch.device('cuda')
    path=ROOT/'artifacts/eda/levers_mamba_integration_v1'/f'int_corrg_{pool}_suffix_s7_fmamba_u18000.pt'
    world,_=T.load_world(path,device)
    meta,tr,_=T.split()
    cl,_=classify(meta)
    cache=T.build_cache(pool,device)
    config=config_from_dict(torch.load(S.CHECKPOINT,map_location='cpu',weights_only=False)['config'])
    rows=[(i,a) for i in torch.where(~tr)[0].tolist() for a in range(1,5) if int(cl[i,a])==1]
    names=('map','hud','all')
    numer={region:{key:0. for key in ('actual','self','gen','oracle_raw','oracle_normalized','hard')} for region in names}
    source_count=torch.zeros(6,dtype=torch.long)
    weight_sum=torch.zeros(6,dtype=torch.float64)
    samples=0
    with autocast_context(config):
        for start in range(0,len(rows),8):
            rr=rows[start:start+8]
            ii=torch.tensor([i for i,a in rr]); aa=torch.tensor([a for i,a in rr]); b=len(rr)
            s=cache['ctx'][ii].to(device)
            act=torch.cat([cache['ctx_a'][ii],aa[:,None]],1).to(device)
            p=world(s,act)[0][:,-1].float()
            w=world.last_weights[:,-1].float().clone()
            gen=world.last_generated[:,-1].float()
            root=s[:,-1].float()
            target=cache['one'][ii,aa].float().to(device)
            grid=root.view(b,9,9,192);pad=F.pad(grid,(0,0,1,1,1,1))
            cands=[grid]+[pad[:,1+dr:10+dr,1+dc:10+dc] for dr,dc in W.NEIGHBOURS]
            cands=torch.stack([x.reshape(b,81,192) for x in cands]+[gen],-2)
            assert torch.equal(F.layer_norm((w[...,None]*cands).sum(-2),(192,)),p)
            e=((cands-target[...,None,:])**2).sum(-1)
            ix=e.argmin(-1)
            oracle_raw=cands.gather(-2,ix[...,None,None].expand(-1,-1,1,192))[...,0,:]
            norm_cands=F.layer_norm(cands,(192,))
            en=((norm_cands-target[...,None,:])**2).sum(-1)
            ixn=en.argmin(-1)
            oracle_normalized=norm_cands.gather(-2,ixn[...,None,None].expand(-1,-1,1,192))[...,0,:]
            hidx=w.argmax(-1)
            hard=cands.gather(-2,hidx[...,None,None].expand(-1,-1,1,192))[...,0,:]
            hard=F.layer_norm(hard,(192,))
            outputs={'actual':p,'self':root,'gen':F.layer_norm(gen,(192,)),
                     'oracle_raw':F.layer_norm(oracle_raw,(192,)),
                     'oracle_normalized':oracle_normalized,'hard':hard}
            for region,sl in (('map',slice(0,63)),('hud',slice(63,81)),('all',slice(None))):
                for key,value in outputs.items():
                    numer[region][key]+=float(((value[:,sl]-target[:,sl])**2).sum())
            source_count+=torch.bincount(hidx.cpu().reshape(-1),minlength=6)
            weight_sum+=w.double().sum((0,1)).cpu()
            samples+=b
    out={r:{'error_over_self':{k:v['self'] and v[k]/v['self'] for k in v},
            'error_share_actual':v['actual']/numer['all']['actual'],
            'error_share_self':v['self']/numer['all']['self']} for r,v in numer.items()}
    return {'checkpoint_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
            'n_blocked':samples,'regions':out,
            'dominant_source_fraction':(source_count/source_count.sum()).tolist(),
            'mean_source_weight':(weight_sum/weight_sum.sum()).tolist()}


if __name__=='__main__':
    out={pool:run(pool) for pool in ('raw','ldad1','ldad10')}
    path=ROOT/'artifacts/experiments/20260929_mamba_integration/blocked_candidates18k.json'
    path.write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps(out,indent=2))
