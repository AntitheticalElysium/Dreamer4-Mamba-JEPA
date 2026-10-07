"""CPU conditioning of saved E19 router/generator traces by health and motion.

No model forward. Physical current health comes from visible simulator metadata;
generator health uses the saved oracle-readout delta plus that run's current
HUD readout. The two baselines are distinguished. Existing inspected diagnosis
roots, living k>=3, >=2 ordinary damage versus unchanged. Paired descriptive
strata, not a causal claim or new judgement block. Atomic arm journals.
"""
import json
from pathlib import Path
import torch
import e19_diagnose as D

T,R=D.T,D.R
HERE=Path(__file__).resolve().parent


def describe(z,current,physical,m):
    n=int(m.sum())
    if not n:return {'n':0}
    def moment(v):
        x=v[m];return {'mean':float(x.mean()),'median':float(x.median()),'q10_q90':x.quantile(torch.tensor([.1,.9])).tolist()}
    ret={'n':n,'physical_current_health':moment(physical.float()),'read_current_health':moment(current),
         'generator_read_health':moment(current+z['delta']['gen63']),
         'generator_health_change':moment(z['delta']['gen63']),
         'actual_health_change':moment(z['delta']['actual']),
         'generate_weight':moment(z['weight']['generate']),
         'generator_closer_than_best_copy':int((z['error']['gen'][m]<z['error']['best_copy'][m]).sum()),
         'generator_L1_over_self':float(z['error']['gen'][m].sum()/z['error']['self'][m].sum().clamp_min(1e-12)),
         'actual_drawn':int((z['delta']['actual'][m]<-1.5).sum()),
         'force_generate_drawn':int((z['delta']['gen63'][m]<-1.5).sum())}
    return ret


def main():
    torch.set_num_threads(3);meta,fit,seeds=T.split()
    masks={k:v[:,0]for k,v in D.D.DR.masks(meta).items()};shift,ok=D.true_shifts(meta)
    valid=masks['valid']&masks['k3'];assert (ok|~valid).all()
    hp=(torch.cat([meta['root_visible'][:,None],meta['future_visible'][:,0]],1)[... ,1512]*9).round()[:,:D.H].long()
    hit=masks['drop2'];same=masks['dh']==0
    sources={str(p.resolve()):R.file_hash(p)for p in [Path(__file__),Path(D.__file__),Path(T.__file__),HERE/'h16_resume.py']}
    files={}
    for s in (7,8):
        for arm in 'BC':
            name=f'e19_{arm}_s{s}_fmamba_from36000'
            files[name]=(HERE/'evals'/(name+'__e19_router_trace.pt'),HERE/'evals'/(name+'__e19_health_per_root.pt'))
    spec={'scope':__doc__,'sources':sources,'inputs':{str(p):R.file_hash(p)for pair in files.values()for p in pair},
          'metadata':R.file_hash(T.META),'masks':{k:R.tensor_hash(v)for k,v in masks.items()},
          'runtime':{'torch':str(torch.__version__),'precision':'CPU FP32','threads':3}}
    store=R.Store(HERE/'evals/resume/e19_fmamba__health_conditioning',spec)
    with store.lock():
        done=store.load('result')
        if done is not None:print(json.dumps(done,indent=2),flush=True);return
        out={}
        for name,(tp,pp)in files.items():
            result=store.load(name)
            if result is None:
                z=torch.load(tp,map_location='cpu',mmap=True,weights_only=False)
                p=torch.load(pp,map_location='cpu',mmap=True,weights_only=False)
                assert torch.equal(z['seed'],meta['seed'])and torch.equal(p['seed'],meta['seed'])
                current=p['current_hp'];rows={}
                for h in range(1,10):
                    for motion,sm in [('all',torch.ones_like(valid)),('scroll',shift!=0),('stationary',shift==0)]:
                        for event,em in [('damage',hit),('unchanged',same)]:
                            m=valid&(hp==h)&sm&em
                            rows[f'hp{h}_{motion}_{event}']=describe(z,current,hp,m)
                fresh=masks['adjacent']&~masks['win']&~masks['adjwin']
                for label,m in {'fresh_damage':valid&hit&fresh,'scroll_damage':valid&hit&(shift!=0),
                                 'stationary_damage':valid&hit&(shift==0),'unchanged':valid&same}.items():
                    rows[label]=describe(z,current,hp,m)
                result={'name':name,'strata':rows,'current_health_reader_max_error':float((current[valid]-hp[valid]).abs().max()),
                        'trace_reconstruction_max':z['reconstruction_max_abs']}
                store.save(name,result,1)
            out[name]=result
        result={'scope':__doc__,'contract':spec,'arms':out}
        store.save('result',result,1);R.atomic_json(HERE/'evals/e19_fmamba__health_conditioning.json',result)
        print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
