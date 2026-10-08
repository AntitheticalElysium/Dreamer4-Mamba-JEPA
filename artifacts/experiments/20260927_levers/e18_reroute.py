"""Frozen final-block canvas memory rerouting; oracle addresses, past features only.

Same16 preselected held windows as e18_access. Correct remembered residual,
zero residual and cyclically permuted remembered residual versus unchanged.
No model/head training. CPU FP32, exact non-treated-output control, paired
window bootstrap. This does not test a learned deployable read or an information
ceiling. Targets score predictions; future coordinates only select addresses.
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import e18_access as A
T,C,R=A.T,A.C,A.R


def main():
    torch.set_num_threads(3)
    root=HERE/'evals/resume/e18_access_s7_at12000'
    old=R.Store(root,json.loads((root/'contract.json').read_text()))
    records=old.load('addresses');prior=old.load('result')
    selected=prior['CPU_reference_windows']
    path=T.OUT/'corrt_raw_teacher_s7_fcanvas_u36000_at12000.pt';pp=T.POOLS['raw']/'pool.pt'
    spec={'scope':__doc__,'inputs':{str(p):R.file_hash(p)for p in (path,pp,Path(A.__file__).resolve(),Path(__file__).resolve())},
          'access_contract':old.contract,'windows':selected,'precision':'CPU FP32','threads':3,
          'sources':prior['contract']['sources'],'draws':2000,'bootstrap_seed':20261006}
    assert spec['inputs'][str(pp)]==prior['contract']['pool'] and spec['inputs'][str(path)]==prior['contract']['checkpoint']
    store=R.Store(HERE/'evals/resume/e18_reroute_s7_at12000',spec)
    with store.lock():
        result=store.load('result')
        if result is not None:print(json.dumps(result,indent=2));return
        pool=torch.load(pp,map_location='cpu',mmap=True,weights_only=False)
        w=T.TWorld('corrt',backbone='fcanvas').eval();w.load_state_dict(torch.load(path,map_location='cpu',weights_only=False)['world'])
        pieces=[]
        for start in range(0,len(selected),4):
            part=store.load(f'batch_{start}')
            if part is None:
                ids=selected[start:start+4];rec=[r for r in records if r['row']in ids]
                s=pool['tokens'][ids].float();a=F.pad(pool['actions'][ids],(0,1))
                b,t=s.shape[:2];pad=t-1;cols=9+2*pad
                sh=F.pad(A.estimate(s[:,:-1],s[:,1:]),(1,0));off=torch.tensor(A.SHIFTS)[sh].cumsum(1)
                rr=torch.arange(7)[:,None]+off[...,0,None,None]+pad
                cc=torch.arange(9)[None]+off[...,1,None,None]+pad
                raw=(rr*cols+cc).flatten(2);present=torch.zeros(b,(7+2*pad)*cols,dtype=torch.bool).scatter(1,raw.flatten(1),True)
                packed=(present.long().cumsum(1)-1).gather(1,raw.flatten(1)).view(b,t,63)
                streams=int(present.sum(1).max())+19;queries=[[]for _ in range(t)];dest=[[]for _ in range(t)]
                index=torch.tensor([[ids.index(r['row']),r['target']-1,r['cell']]for r in rec])
                treated=torch.zeros(b,t,81,dtype=torch.bool);treated[index[:,0],index[:,1],index[:,2]]=True
                for r in rec:
                    bi=ids.index(r['row']);at=r['target']-1
                    queries[at].append(bi*streams+int(packed[bi,r['sighting'],r['oldcell']]))
                    dest[at].append(bi*streams+int(packed[bi,at,r['cell']]))
                queries=[torch.tensor(x,dtype=torch.long)for x in queries]
                original=T.masked_scan;values={};gens={};weights={};untreated={}
                for mode in ('baseline','remembered','zero','permuted'):
                    calls=[0]
                    def scan(mixer,inputs,keep):
                        calls[0]+=1
                        if calls[0]!=6:return A.reference(mixer,inputs,keep,None,[])
                        captured=[]
                        hook=mixer.core.out_proj.register_forward_hook(lambda m,a,y:captured.append(y.detach().clone())if len(y)<len(inputs)else None)
                        try:y=A.reference(mixer,inputs,keep,queries,[])
                        finally:hook.remove()
                        if mode!='baseline':
                            assert len(captured)==sum(bool(len(x))for x in queries)
                            qi=0
                            for at,q in enumerate(queries):
                                if not len(q):continue
                                v=captured[qi];qi+=1
                                if mode=='zero':v=torch.zeros_like(v)
                                if mode=='permuted':v=v.roll(1,0)
                                y[torch.tensor(dest[at]),at]=v
                        return y
                    try:
                        T.masked_scan=scan
                        with torch.no_grad():out,h,gen=w(s,a)
                    finally:T.masked_scan=original
                    assert calls[0]==6
                    values[mode]=out[index[:,0],index[:,1],index[:,2]].clone()
                    gens[mode]=F.layer_norm(gen,(192,))[index[:,0],index[:,1],index[:,2]].clone()
                    weights[mode]=w.last_weights[index[:,0],index[:,1],index[:,2],5].clone()
                    if mode=='baseline':baseline=out.clone()
                    untreated[mode]=float((out[~treated]-baseline[~treated]).abs().max())
                    assert untreated[mode]<=1e-6,untreated[mode]
                truth=torch.stack([pool['tokens'][r['row'],r['target'],r['cell']].float()for r in rec])
                sight=torch.stack([pool['tokens'][r['row'],r['sighting'],r['oldcell']].float()for r in rec])
                part={'records':rec,'output_error':{k:(v-truth).square().mean(1)for k,v in values.items()},
                      'generator_error':{k:(v-truth).square().mean(1)for k,v in gens.items()},
                      'generate_weight':weights,'sighting_copy_error':(sight-truth).square().mean(1),'untreated_max':untreated}
                store.save(f'batch_{start}',part,1)
            pieces.append(part);print(json.dumps({'windows_done':start+len(selected[start:start+4])}),flush=True)
        records=[r for p in pieces for r in p['records']];rows=np.array([r['row']for r in records])
        rng=np.random.default_rng(20261006);clusters=np.unique(rows)
        ii=rng.integers(len(clusters),size=(2000,len(clusters)))
        results={}
        for kind in ('output_error','generator_error'):
            error={mode:torch.cat([p[kind][mode]for p in pieces])for mode in pieces[0][kind]}
            contrasts={}
            for mode in ('remembered','zero','permuted'):
                diff=(error[mode]-error['baseline']).numpy()
                numer=np.array([diff[rows==r].sum()for r in clusters]);denom=np.array([(rows==r).sum()for r in clusters])
                boot=numer[ii].sum(1)/denom[ii].sum(1)
                contrasts[mode]={'error_minus_baseline':float(diff.mean()),'interval95':np.quantile(boot,[.025,.975]).tolist()}
            results[kind]={'mean':{k:float(v.mean())for k,v in error.items()},'contrasts':contrasts}
        result={'scope':__doc__,'contract':spec,'windows':len(selected),'cells':len(records),
                'same_slot':sum(r['same_slot']for r in records),'moved_slot':sum(not r['same_slot']for r in records),
                'scores':results,'sighting_copy_mean_error':float(torch.cat([p['sighting_copy_error']for p in pieces]).mean()),
                'generate_weight_mean':{k:float(torch.cat([p['generate_weight'][k]for p in pieces]).mean())for k in pieces[0]['generate_weight']},
                'non_treated_output_max_error':max(max(p['untreated_max'].values())for p in pieces)}
        store.save('result',result,1);R.atomic_json(HERE/'evals/e18_reroute_s7_at12000.json',result)
        print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
