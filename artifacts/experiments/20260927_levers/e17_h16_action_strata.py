"""Post-hoc E17 CPU decision decomposition, declared before execution Oct7.

No fitting, GPU, world forward or new roots. Reproduce the official decisions;
restrict the same scores to movements1..4, and measure full-minus-movement.
Strata fixed from root visible state: adjacent lava/zombie (exclusive four-way)
and health<=2 versus>2. Also partition by true H16 best stationary versus best
movement action; this outcome-defined partition diagnoses selected mistakes,
not a deployable classification. Histograms count3 independent heads, not an
ensemble. Paired4000 episode-cluster intervals; reused DEV-B exploratory.
Input-bound atomic result and raw rows; logs EDA, no notebook writes.
"""
import json
from pathlib import Path
import numpy as np
import torch
import h16_resume as R
import e17_h16_diagnose as D

HERE=Path(__file__).resolve().parent


def main():
    torch.set_num_threads(2)
    fp=HERE/'evals/resume/e17_h16_final'
    fc=json.loads((fp/'contract.json').read_text());fs=R.Store(fp,fc)
    final=fs.load('result');rows=fs.load('rows');assert final is not None
    mp=D.CACHE/'dev_meta.pt';meta=torch.load(mp,mmap=True,weights_only=False)
    ib=torch.where(meta['seed']%2==1)[0]
    vis=meta['visible'][ib]
    worlds={};inputs={str(mp):R.file_hash(mp)}
    for bb in ('attention','mamba'):
        for seed in (7,8):
            name=f'corrt_rawlong_teacher_s{seed}'+('_fmamba' if bb=='mamba' else '')+'_L16b40_from36000'
            ds=list(D.RESUME.glob(name+'__w15__det__*'));assert len(ds)==1
            p=ds[0];c=json.loads((p/'contract.json').read_text());ws=R.Store(p,c)
            v=ws.load('result');assert v is not None
            rec=json.loads((p/'result.json').read_text());inputs[str(p/rec['file'])]=R.file_hash(p/rec['file'])
            worlds[bb+str(seed)]=v
    base=worlds['mamba7'];P,opp=base['P_devB'][...,-1],base['opportunity_devB']
    assert torch.equal(P,meta['p16'][ib]);seeds=base['seed_devB'][opp].numpy()
    tiles=vis[:,:1071].reshape(-1,7,9,17).argmax(-1)
    mobs=vis[:,1071:1512].reshape(-1,7,9,7)[...,0]
    positions=((2,4),(4,4),(3,3),(3,5))
    lava=torch.stack([tiles[:,r,c]==14 for r,c in positions]).any(0)[opp].numpy()
    zombie=torch.stack([mobs[:,r,c]>0 for r,c in positions]).any(0)[opp].numpy()
    hp=(vis[:,1512]*9)[opp].numpy();low=hp<=2+1e-6
    moves=torch.tensor([1,2,3,4]);stays=torch.tensor([0,*range(5,17)])
    best_move=P[opp][:,moves].amin(1);best_stay=P[opp][:,stays].amin(1)
    groups={'all':np.ones(len(seeds),bool),'zombie_only':zombie&~lava,'lava_only':lava&~zombie,
        'both':zombie&lava,'neither':~zombie&~lava,'health_at_most2':low,'health_over2':~low,
        'move_strictly_better':(best_move<best_stay).numpy(),
        'stationary_strictly_better':(best_stay<best_move).numpy(),'best_classes_tied':(best_move==best_stay).numpy()}
    contract={'scope':__doc__,'sources':{p:R.file_hash(p) for p in (__file__,R.__file__,D.__file__)},
              'inputs':inputs,'final_contract':fs.contract,'runtime':str(torch.__version__)}
    store=R.Store(HERE/'evals/resume/e17_h16_action_strata',contract)
    with store.lock():
        done=store.load('result')
        if done is not None:print(json.dumps(done),flush=True);return
        raw={};out={}
        for name,v in worlds.items():
            assert torch.equal(P,v['P_devB'][...,-1]) and torch.equal(opp,v['opportunity_devB'])
            scores=torch.stack([v['raw_scores'][f'trajectory_{s}'] for s in range(3)])
            picks=scores.argmin(-1);restricted=moves[scores[:,:,moves].argmin(-1)]
            truth=P[opp].unsqueeze(0).expand(3,-1,-1)
            safe=1-truth.gather(-1,picks[:,opp,None]).squeeze(-1)
            movement=1-truth.gather(-1,restricted[:,opp,None]).squeeze(-1)
            sr=safe.mean(0).numpy();mr=movement.mean(0).numpy()
            assert np.max(np.abs(sr-rows['safe'][name]['full16']))<=1e-6
            result={}
            for g,use in groups.items():
                if not use.any():result[g]={'roots':0};continue
                actions=picks[:,opp][:,use]
                result[g]={'roots':int(use.sum()),'score':float(sr[use].mean()),'movement_only':float(mr[use].mean()),
                    'prior':float(rows['prior'][use].mean()),'full_minus_prior':D.paired(sr[use],rows['prior'][use],seeds[use]),
                    'full_minus_movement':D.paired(sr[use],mr[use],seeds[use]),
                    'histogram':torch.bincount(actions.flatten(),minlength=17).tolist(),
                    'stationary_choice_fraction':float((~torch.isin(actions,moves)).float().mean())}
            raw[name]={'full':sr,'movement':mr,'choices':picks[:,opp]};out[name]=result
        store.save('rows',{'worlds':raw,'groups':groups,'seeds':seeds,'health':hp},1)
        result={'scope':__doc__,'contract':contract,'worlds':out,'group_counts':{k:int(v.sum()) for k,v in groups.items()}}
        store.save('result',result,1);R.atomic_json(HERE/'evals/e17_h16_action_strata.json',result)
        print(json.dumps({'group_counts':result['group_counts'],'worlds':out}),flush=True)


if __name__=='__main__':main()
