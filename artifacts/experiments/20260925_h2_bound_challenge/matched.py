"""Exploratory matched-readout challenge to h2_bound, fixed before this run.

The previous input->generated comparison used a 17-output vector root head but a
shared scalar-per-branch generated head. Here both readouts use the *same*
Head('branch') family, all-action soft ranking, FIT-train/FIT-dev partition,
three seeds, 3000 steps, and 54k judgment roots. Root branch features are the
four z vectors, three recorded actions, and one candidate action. Generated
branch features are the H2 generated agent readout for that candidate. Refit
z4_actions with its prior 17-output head as a reproduction control.

Primary: within-root expected safe choice on zombie-adjacent opportunity roots;
paired episode-seed-clustered difference root_branch - generated_features.
Report disjoint zombie-only and lava-only strata to check the 'trade' wording.
This is exploratory: the 54k block was previously inspected and no finite head
can certify an information-theoretic upper bound.
"""
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).parent
OLD = ROOT / 'artifacts/experiments/20260921_readout_ladder'
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(OLD))
from boundary import judge_store
from confirm import seeds_for
from frozen_heads import dev_rows
from frozen_ladder import materialize, scores, standardize, strata, train
from ladder import paired
from observability import expected_safe, load
from d4mj.data import _sha256
from d4mj.experiments import _load_bridge_parent
from d4mj.lewm_diagnostics import FORK_STORE

N, STEPS, SEEDS = 17, 3000, 3
CHECKPOINT = ROOT / 'artifacts/lewm_m4_canonical/raw/bridge/step-002000.pt'
STORE = ROOT / 'artifacts/eda/observe_fresh_v5'


def log(**kw):
    print(json.dumps(kw),flush=True)


def root_branch(data):
    root=torch.cat((data['z4'],data['actions'][:,-3:].flatten(1)),1)
    cand=F.one_hot(torch.arange(N),N).float()[None].expand(len(root),-1,-1)
    return torch.cat((root[:,None].expand(-1,N,-1),cand),-1)


def main():
    start=time.time()
    os.chdir(ROOT)
    partition=json.loads((OLD/'evidence/root_partition.json').read_text())
    fit_seeds, unallocated=seeds_for(partition,FORK_STORE)
    dev_seeds=sorted(partition['fit_dev']['seeds'])
    assert not ((set(partition['reserved_for_gate']['seeds'])|set(unallocated)) & (set(fit_seeds)|set(dev_seeds)))
    assert _sha256(CHECKPOINT)==json.loads((OLD/'evidence/confirm.json').read_text())['checkpoint_sha256']
    fit=load(fit_seeds)['fit']
    dev=dev_rows(dev_seeds)
    judge,manifest,files=judge_store(STORE)
    assert set(judge['seed'].tolist()).isdisjoint(set(fit_seeds)|set(dev_seeds))
    bundle,_,_=_load_bridge_parent(CHECKPOINT)
    bundle.encoder.freeze();bundle.world.eval()
    for module in bundle.world.modules():
        if isinstance(module,nn.BatchNorm1d):module.eval()
    for data in (fit,dev,judge):
        m=materialize(bundle,data['frames'],data['actions'])
        data.update({k:m[k] for k in ('z4','generated_features')})
        data['z4_actions']=torch.cat((data['z4'],data['actions'][:,-3:].flatten(1)),1)
        data['root_branch']=root_branch(data)
        del data['frames']
    del bundle
    torch.cuda.empty_cache()
    log(stage='features',fit=len(fit['seed']),dev=len(dev['seed']),judge=len(judge['seed']),seconds=round(time.time()-start,1))
    p=torch.cat((fit['p_death1'],dev['p_death1']))
    pj=judge['p_death1']
    tr=torch.arange(len(fit['seed']))
    hold=torch.arange(len(fit['seed']),len(p))
    prior=int(fit['p_death1'][fit['p_death1'].amax(1)>fit['p_death1'].amin(1)].mean(0).argmin())
    safe={'prior':expected_safe(-F.one_hot(torch.full((len(pj),),prior),N).float(),pj)[0]}
    arms={'root_branch':('branch','root_branch'),
          'generated_features':('branch','generated_features'),
          'z4_actions_vector':('vector','z4_actions')}
    details={}
    device=torch.device('cuda')
    for name,(kind,key) in arms.items():
        x=torch.cat((fit[key],dev[key])).float()
        xj=judge[key].float()
        mean,std=standardize(x,tr)
        x=(x-mean)/std;xj=(xj-mean)/std
        runs=[];selected=[]
        for seed in range(SEEDS):
            model,meta=train(kind,x.shape[1:],x,p,tr,hold,seed=seed,device=device,steps=STEPS)
            pred=scores(model,xj,torch.arange(len(pj)),device)
            runs.append(expected_safe(pred,pj)[0])
            selected.append({'seed':seed,'fit_step':meta['selected_step'],'inner_safe':meta['inner_safe'],
                             'params':meta['parameters']})
            del model
        safe[name]=torch.stack(runs).mean(0)
        details[name]=selected
        log(stage='arm',arm=name,overall=round(float(safe[name][pj.amax(1)>pj.amin(1)].mean()),4),
            selected=selected,seconds=round(time.time()-start,1))
    opp=pj.amax(1)>pj.amin(1)
    st=strata(judge['visible'])
    masks={'overall':opp,'zombie':opp&st['zombie_adjacent'],
           'zombie_only':opp&st['zombie_adjacent']&~st['lava_adjacent'],
           'lava':opp&st['lava_adjacent'],
           'lava_only':opp&st['lava_adjacent']&~st['zombie_adjacent']}
    test=lambda a,b,m:paired(safe[a][m],safe[b][m],judge['seed'][m],draws=1000,seed=20261008)
    result={'status':'exploratory_preinspected_54k','script_sha256':_sha256(Path(__file__)),
            'checkpoint_sha256':_sha256(CHECKPOINT),'judge_manifest':manifest,
            'judge_files':files,'roots':{'fit':len(fit['seed']),'dev':len(dev['seed']),'judge':len(pj)},
            'prior_action':prior,'fit':details,'strata':{}}
    for name,mask in masks.items():
        result['strata'][name]={'count':int(mask.sum()),
            'expected_safe':{a:float(v[mask].mean()) for a,v in safe.items()},
            'contrasts':{f'{a}_minus_{b}':test(a,b,mask) for a,b in
                         (('root_branch','generated_features'),('root_branch','prior'),
                          ('generated_features','prior'),('root_branch','z4_actions_vector'))}}
    result['seconds']=round(time.time()-start,1)
    (HERE/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    log(stage='done',seconds=result['seconds'],summary={s:{k:round(v,4) for k,v in d['expected_safe'].items()}
         for s,d in result['strata'].items()})


if __name__=='__main__':main()
