"""Extend the completed CPU recurrence lesion to the full original200-root cohort.

The64-root result showed differing convolution dependence at seeds7/8. Reuse its
64 per-root records only after exact prefix input, weights and source checks;
compute the remaining136 roots using identical lesions/metrics. Same original
200-root fixed-clock panel, not new sealed evidence or retraining. CPU FP32,
source/input-bound per-root journals, paired seed-cluster bootstrap via the same
statistics function. No model/trainer source edits or GPU allocation.
"""
import json
from pathlib import Path
import numpy as np
import torch
import e17_recurrence as E

R,T,C,SD,S=E.R,E.T,E.C,E.SD,E.S
HERE=Path(__file__).resolve().parent


def measure(world,s,a,alive):
    tt,ce,cls,sf,sc,nb=C.cells(s)
    use=(cls==1)&alive[tt]&(tt>=4)&((tt-sf)>=6)&((tt-sf)<=15)
    record={g:torch.zeros(len(E.MODES),5,dtype=torch.float64)for g in ('same','moved')}
    with torch.no_grad():
        for t in tt[use].unique().tolist():
            mask=use&(tt==t);x=s[max(0,t-15):t];aa=a[max(0,t-15):t]
            ab=x.clone();ab[:-5]=x[-5]
            frames=torch.stack([x,ab]);actions=torch.stack([aa,aa]);target=s[t,ce[mask]]
            es=(s[sf[mask],sc[mask]]-target).square().sum(-1)
            en=(s[t,nb[mask]]-target).square().sum(-1);same=sc[mask]==ce[mask]
            for mi,mode in enumerate(E.MODES):
                with E.lesion(world,mode):pred=world(frames,actions)[0][:,-1]
                e1=(pred[0,ce[mask]]-target).square().sum(-1)
                e2=(pred[1,ce[mask]]-target).square().sum(-1)
                for g,m in [('same',same),('moved',~same)]:
                    record[g][mi]+=torch.tensor([int(m.sum()),float(e1[m].sum()),float(e2[m].sum()),
                                                float(es[m].sum()),float(en[m].sum())],dtype=torch.float64)
    for v in record.values():assert torch.equal(v[:,[0,3,4]],v[:1,[0,3,4]].expand(len(E.MODES),-1))
    return record


def main():
    torch.set_num_threads(3)
    meta,_,_=T.split();p=Path(str(T.CACHE).format('raw'));assert p.exists() and (SD.CACHE/'DONE').exists()
    cache=torch.load(p,map_location='cpu',mmap=True,weights_only=False)
    roots=torch.randperm(len(meta['seed']),generator=torch.Generator().manual_seed(20261005))[:200]
    n=int((SD.CACHE/'DONE').read_text());assert n==len(meta['seed'])
    future=torch.from_numpy(SD.memmap(SD.CACHE/'fut5.f16',(n,5,16,81,192)))[roots,0]
    seqs=torch.cat([cache['ctx'][roots],future],1)
    acts=torch.cat([cache['ctx_a'][roots],cache['fut_a'][roots]],1)
    alive=torch.cat([torch.ones(len(roots),4,dtype=torch.bool),~meta['future_dead'][roots,0].cumsum(1).bool()],1)
    seeds=meta['seed'][roots]
    tensors={'frames':seqs,'actions':acts,'alive':alive,'roots':roots,'seeds':seeds}
    for seed in (7,8):
        name=f'corrt_rawlong_teacher_s{seed}_fmamba_L16b40_from36000'
        path=T.ROOT/'artifacts/eda/levers_tworlds_v1'/(name+'.pt')
        priorroot=HERE/'evals/resume'/(name+'__e17_recurrence_cpu')
        prior_spec=json.loads((priorroot/'contract.json').read_text())
        old=R.Store(priorroot,prior_spec);assert old.load('result')is not None
        assert R.file_hash(path)==prior_spec['checkpoint']
        for q,h in prior_spec['sources'].items():assert R.file_hash(q)==h,q
        for k,v in tensors.items():assert R.tensor_hash(v[:64])==prior_spec['inputs'][k],k
        spec={'scope':__doc__,'base_contract':old.contract,'checkpoint':R.file_hash(path),
              'sources':{**prior_spec['sources'],str(Path(__file__).resolve()):R.file_hash(__file__)},
              'inputs':{k:R.tensor_hash(v)for k,v in tensors.items()},'precision':'CPU FP32','threads':3,
              'constructor_config':R.file_hash(S.CHECKPOINT),'roots':200,'reused':64,'draws':2000,'bootstrap_seed':20261006}
        assert spec['constructor_config']==prior_spec['canonical_constructor_config']
        store=R.Store(HERE/'evals/resume'/(name+'__e17_recurrence_full_cpu'),spec)
        with store.lock():
            result=store.load('result')
            if result is not None:print(json.dumps(result,indent=2),flush=True);continue
            world,_=T.load_world(path,torch.device('cpu'))
            control=store.load('reuse_control')
            if control is None:
                # Recompute one nonempty old root before accepting any cached records.
                ri=next(i for i in range(64)if sum(float(v[0,0])for v in old.load('root_'+str(i)).values())>0)
                ref=old.load('root_'+str(ri));new=measure(world,seqs[ri].float(),acts[ri],alive[ri])
                delta=max(float((new[g]-ref[g]).abs().max())for g in ref)
                assert delta<=1e-6,delta
                control={'root':ri,'max_absolute_record_difference':delta,'prefix_inputs_exact':True,
                         'sources_exact':True,'checkpoint_exact':True}
                store.save('reuse_control',control,1);print(json.dumps({'reuse_control':control}),flush=True)
            rows=[]
            for ri in range(len(roots)):
                record=store.load('root_'+str(ri))
                if record is None:
                    record=old.load('root_'+str(ri))if ri<64 else measure(world,seqs[ri].float(),acts[ri],alive[ri])
                    store.save('root_'+str(ri),record,1)
                rows.append(record)
                if ri%16==0:print(json.dumps({'name':name,'roots_done':ri+1}),flush=True)
            raw={g:torch.stack([x[g]for x in rows]).numpy()for g in ('same','moved')}
            result={'scope':__doc__,'name':name,'contract':spec,'roots':len(roots),'seed_clusters':len(seeds.unique()),
                    'groups':E.statistics(raw,seeds.numpy()),'reuse_control':control}
            store.save('rows',{'raw':raw,'roots':roots,'seeds':seeds},1);store.save('result',result,1)
            R.atomic_json(HERE/'evals'/(name+'__e17_recurrence_full_cpu.json'),result)
            print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
