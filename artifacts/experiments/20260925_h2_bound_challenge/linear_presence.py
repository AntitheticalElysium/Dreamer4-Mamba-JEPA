"""Regularized linear control for the visible zombie/lava adjacency probes.

Fixed before run. Same frozen Raw H2 encoder, FIT-train seeds and inner-FIT seed
split, and preinspected 54k judge roots as hazard_presence.py. Standardize on
FIT-train only. Fit ridge least-squares binary discriminants on z1/z4 with
lambda/n in (0.01, 0.1, 1, 10, 100), choose lambda by inner-FIT AUC, then
report 54k AUC. A linear success would falsify a broad 'z cannot decode mobs'
reading of the MLP result. This does not test action consequences.
"""
import json
import os
import sys
import time
from pathlib import Path
import torch

ROOT=Path(__file__).resolve().parents[3]
HERE=Path(__file__).parent
OLD=ROOT/'artifacts/experiments/20260921_readout_ladder'
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(HERE));sys.path.insert(0,str(OLD))
from boundary import judge_store
from confirm import seeds_for
from frozen_ladder import strata
from hazard_presence import encode,auc
from observability import load
from d4mj.data import _sha256
from d4mj.experiments import _load_bridge_parent
from d4mj.lewm_diagnostics import FORK_STORE

CHECKPOINT=ROOT/'artifacts/lewm_m4_canonical/raw/bridge/step-002000.pt'
STORE=ROOT/'artifacts/eda/observe_fresh_v5'
LAMBDA=(.01,.1,1.,10.,100.)


def main():
    start=time.time();os.chdir(ROOT)
    assert _sha256(CHECKPOINT)==json.loads((OLD/'evidence/confirm.json').read_text())['checkpoint_sha256']
    partition=json.loads((OLD/'evidence/root_partition.json').read_text())
    fit_seeds,_=seeds_for(partition,FORK_STORE)
    fit_data=load(fit_seeds)['fit']
    judge,manifest,files=judge_store(STORE)
    bundle,_,_=_load_bridge_parent(CHECKPOINT)
    bundle.encoder.freeze()
    fit_x=encode(bundle.encoder,fit_data['frames'],bundle.device)
    judge_x=encode(bundle.encoder,judge['frames'],bundle.device)
    del bundle
    seeds=fit_data['seed'];unique=seeds.unique().sort().values
    perm=torch.randperm(len(unique),generator=torch.Generator().manual_seed(20261009))
    hold=set(unique[perm[:round(.15*len(unique))]].tolist())
    hold_mask=torch.tensor([int(s) in hold for s in seeds.tolist()])
    tr=torch.where(~hold_mask)[0];va=torch.where(hold_mask)[0]
    result={'status':'exploratory_preinspected_54k','script_sha256':_sha256(Path(__file__)),
            'checkpoint_sha256':_sha256(CHECKPOINT),'judge_manifest':manifest,
            'rows':{'fit':len(tr),'inner':len(va),'judge':len(judge['seed'])},'labels':{}}
    labels={'zombie_adjacent':(strata(fit_data['visible'])['zombie_adjacent'].float(),
                                strata(judge['visible'])['zombie_adjacent'].float()),
            'lava_adjacent':(strata(fit_data['visible'])['lava_adjacent'].float(),
                              strata(judge['visible'])['lava_adjacent'].float())}
    device=torch.device('cuda')
    for name,(y,yj) in labels.items():
        result['labels'][name]={}
        for arm in ('z1','z4'):
            x=fit_x[arm].float();xj=judge_x[arm].float()
            mean=x[tr].mean(0);std=x[tr].std(0).clamp_min(1e-5)
            x=(x-mean)/std;xj=(xj-mean)/std
            xt=x[tr].to(device); xv=x[va].to(device); xj=xj.to(device)
            yt=y[tr].to(device);cy=yt.mean()
            lhs=xt.T@xt;rhs=xt.T@(yt-cy)
            eye=torch.eye(x.shape[-1],device=device)
            best=(-1,None,None)
            candidates=[]
            for scale in LAMBDA:
                w=torch.linalg.solve(lhs+scale*len(tr)*eye,rhs)
                score=(xv@w+cy).cpu()
                value=auc(score,y[va])
                candidates.append({'lambda_per_n':scale,'inner_auc':value})
                if value>best[0]:best=(value,scale,w)
            score=(xj@best[2]+cy).cpu()
            row={'selected_lambda_per_n':best[1],'inner_auc':best[0],
                 'judge_auc':auc(score,yj),'candidates':candidates}
            result['labels'][name][arm]=row
            print(json.dumps({'stage':'arm','label':name,'arm':arm,**{k:v for k,v in row.items() if k!='candidates'},
                              'seconds':round(time.time()-start,1)}),flush=True)
    result['seconds']=round(time.time()-start,1)
    (HERE/'linear_presence.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'stage':'done','seconds':result['seconds']}),flush=True)


if __name__=='__main__':main()
