"""E17 fixed-clock visual-history intervention; no new world training.

On 200 diagnosis roots drawn before scoring (seed20261005), sample0, use the
existing recall cell ledger. Compare true up-to15-frame history with the same
length/actions/time rows and latest5 frames, replacing older frames by the oldest
of those5. This removes the older sighting while preserving the evaluation clock.
It is a deliberately inconsistent visual-history ablation, not an in-distribution
information ceiling. Per-root records, source/input/checkpoint-bound resume, paired
seed-cluster intervals. Report same-slot/moved-slot ages6..15 separately.
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import check_recall as C
import h16_resume as R
T=C.T


def stats(a):
    # row: count, original error, ablated error, sighting error, neighbour error
    v=a.sum(0);den=v[4]-v[3]
    return {'n':int(v[0]),'original_capture':float((v[4]-v[1])/den),
            'ablated_capture':float((v[4]-v[2])/den),
            'history_gain':float((v[2]-v[1])/den)}


def run(path,seqs,acts,alive,roots,seeds,device,cfg):
    world,st=T.load_world(path,device)
    store=R.frozen_eval_store(path,st['name']+'__e17_fixed_clock',
        {'roots':200,'sample':0,'seed':20261005,'window':15,'retained_recent':5},
        {'frames':seqs,'actions':acts,'alive':alive,'root_indices':roots})
    with store.lock():
        result=store.load('result')
        if result is not None:return result
        rows=[]
        with torch.no_grad():
            for ri,r in enumerate(roots.tolist()):
                record=store.load('root_'+str(ri))
                if record is None:
                    s=seqs[r].float();a=acts[r];tt,ce,cls,sf,sc,nb=C.cells(s)
                    use=(cls==1)&alive[r,tt]&(tt>=4)&((tt-sf)>=6)&((tt-sf)<=15)
                    out={g:torch.zeros(5,dtype=torch.float64)for g in ('same','moved')}
                    for t in tt[use].unique().tolist():
                        mask=use&(tt==t);x=s[max(0,t-15):t];aa=a[max(0,t-15):t]
                        ab=x.clone();ab[:-5]=x[-5]
                        pred=T.step(world,torch.stack([x,ab]),torch.stack([aa,aa]),device,cfg)
                        target=s[t,ce[mask]]
                        e1=(pred[0,ce[mask]]-target).square().sum(-1)
                        e2=(pred[1,ce[mask]]-target).square().sum(-1)
                        es=(s[sf[mask],sc[mask]]-target).square().sum(-1)
                        en=(s[t,nb[mask]]-target).square().sum(-1)
                        same=sc[mask]==ce[mask]
                        for g,m in [('same',same),('moved',~same)]:
                            out[g]+=torch.tensor([int(m.sum()),float(e1[m].sum()),float(e2[m].sum()),
                                                  float(es[m].sum()),float(en[m].sum())],dtype=torch.float64)
                    record=out;store.save('root_'+str(ri),record,1)
                rows.append(record)
                if ri%50==0:print(json.dumps({'name':st['name'],'roots':ri}),flush=True)
        raw={g:torch.stack([x[g]for x in rows]).numpy()for g in ('same','moved')}
        groups=[np.flatnonzero(seeds[roots].numpy()==s)for s in seeds[roots].unique().tolist()]
        rng=np.random.default_rng(20261005);result={'name':st['name'],'scope':__doc__,'groups':{}}
        for g,v in raw.items():
            reading=stats(v);draws=[]
            for _ in range(2000):
                idx=np.concatenate([groups[i]for i in rng.integers(len(groups),size=len(groups))])
                if v[idx].sum(0)[4]>v[idx].sum(0)[3]:draws.append(stats(v[idx])['history_gain'])
            reading['history_gain_interval95']=np.quantile(draws,[.025,.975]).tolist()
            result['groups'][g]=reading
        result['resume_contract']=store.contract
        R.atomic_torch(HERE/'evals'/(st['name']+'__e17_fixed_clock_rows.pt'),{'raw':raw,'roots':roots,'seed':seeds[roots]})
        R.atomic_json(HERE/'evals'/(st['name']+'__e17_fixed_clock.json'),result)
        store.save('result',result,1)
        return result


if __name__=='__main__':
    torch.set_num_threads(3);device=torch.device('cuda');torch.cuda.set_per_process_memory_fraction(.16)
    from d4mj.config import config_from_dict
    import spatial as S
    cfg=config_from_dict(torch.load(S.CHECKPOINT,map_location='cpu',weights_only=False)['config'])
    meta,_,_=T.split();cache=T.build_cache('raw',device)
    import stochdiag as SD
    fut,_=SD.token_cache(device)
    seqs=torch.cat([cache['ctx'],fut[:,0]],1)
    acts=torch.cat([cache['ctx_a'],cache['fut_a']],1)
    alive=torch.cat([torch.ones(len(seqs),4,dtype=torch.bool),~meta['future_dead'][:,0].cumsum(1).bool()],1)
    roots=torch.randperm(len(seqs),generator=torch.Generator().manual_seed(20261005))[:200]
    for seed in (7,8):
        for suffix in ('_fmamba',''):
            path=T.ROOT/'artifacts/eda/levers_tworlds_v1'/f'corrt_rawlong_teacher_s{seed}{suffix}_L16b40_from36000.pt'
            print(json.dumps(run(path,seqs,acts,alive,roots,meta['seed'],device,cfg),indent=2),flush=True)
            torch.cuda.empty_cache()
