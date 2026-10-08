"""Frozen E17 reader train/generalization diagnosis, declared Oct7 before scoring.

Original Mamba7/8 trajectory heads, three seeds, selected checkpoint versus final
update4000 on EXACT same feature caches. Evaluate conditional-hazard likelihood,
H16 Brier, action-contrast errors and safe choice on FIT, DEV-A and DEV-B.
This tests whether continued reader training fits FIT while worsening held-out
decisions; it cannot infer information absence or causally identify world error.
Different splits can have different difficulty; report prior/opportunity/label
prevalence and compare selected-v-final within each split. No fitting/world/GPU.
CPU2 threads, exact original half-rounded normalization, atomic32-root chunks.
Source/input-bound resume and reproduction of every selected official DEV-B
choice. Logs only EDA, no notebook updates from code. DEV-B reused/exploratory.
"""
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import h16_resume as R
import check_h16_traj as H
import check_h16_signal as HS
import e17_h16_diagnose as D
import e17_h16_error_split as ES

HERE=Path(__file__).resolve().parent


def main():
    torch.set_num_threads(2)
    fp=HERE/'evals/resume/e17_h16_final'
    fc=json.loads((fp/'contract.json').read_text());fs=R.Store(fp,fc)
    assert fs.load('result') is not None
    meta={s:torch.load(D.CACHE/f'{s}_meta.pt',mmap=True,weights_only=False) for s in ('fit','dev')}
    p,_,split=HS.load(full=True);labels={'fit':p[split==0],'dev':p[split==1]}
    inputs={};worlds={}
    for seed in (7,8):
        name=f'corrt_rawlong_teacher_s{seed}_fmamba_L16b40_from36000'
        ds=list(D.RESUME.glob(name+'__w15__det__*'));assert len(ds)==1
        path=ds[0];c=json.loads((path/'contract.json').read_text())
        for kind in ('sources','checkpoints'):
            for f,h in c[kind].items():assert R.file_hash(f)==h,f
        ws=R.Store(path,c);saved=[ws.load(f'head_trajectory_{s}') for s in range(3)]
        for key in [f'head_trajectory_{s}' for s in range(3)]+['result']:
            r=json.loads((path/f'{key}.json').read_text());inputs[str(path/r['file'])]=R.file_hash(path/r['file'])
        for sp in meta:
            f=path/f'features_{sp}.f16';inputs[str(f)]=R.file_hash(f)
            assert fc['inputs'][str(f)]==inputs[str(f)]
            assert torch.equal(labels[sp][...,-1],meta[sp]['p16'])
        worlds[seed]=(path,c,saved,ws.load('result'),fs.load(f'mamba{seed}_norm'))
    contract={'scope':__doc__,'sources':{f:R.file_hash(f) for f in (__file__,H.__file__,HS.__file__,ES.__file__,D.__file__,R.__file__)},
              'inputs':inputs,'final_contract':fs.contract,'labels':{k:R.tensor_hash(v) for k,v in labels.items()},
              'runtime':{'torch':str(torch.__version__),'threads':2,'device':'CPU'}}
    store=R.Store(HERE/'evals/resume/e17_h16_generalization',contract)
    with store.lock():
        done=store.load('result')
        if done is not None:print(json.dumps(done),flush=True);return
        fit_opp=labels['fit'][...,-1].amax(1)>labels['fit'][...,-1].amin(1)
        prior=int(labels['fit'][fit_opp,:,-1].mean(0).argmin())
        out={};raw={}
        for seed,(path,c,saved,official,norm) in worlds.items():
            models=[]
            for ss in saved:
                assert ss['step']==4000
                mm=[]
                for kind in ('best_state','model'):
                    m=H.Hazard(768,16);m.load_state_dict(ss[kind]);mm.append(m.eval())
                models.append(mm)
            by_split={};raw[seed]={}
            for sp in ('fit','dev'):
                progress=json.loads((path/f'features_{sp}.progress.json').read_text())
                assert progress['next']==len(labels[sp])
                xx=torch.from_numpy(np.memmap(path/f'features_{sp}.f16',dtype=np.float16,mode='c',shape=progress['layout']['shape']))
                pieces=[]
                for lo in range(0,len(xx),32):
                    key=f'm{seed}_{sp}_{lo}'
                    batch=store.load(key)
                    if batch is None:
                        x=((xx[lo:lo+32].float()-norm['mu'])/norm['sd']).half().float().flatten(0,1)
                        with torch.no_grad():
                            h=torch.stack([torch.stack([F.softplus(m(x)).view(-1,17,16) for m in mm]) for mm in models])
                        batch={'lo':lo,'hazard':h};store.save(key,batch,lo+len(xx[lo:lo+32]))
                    assert batch['lo']==lo;pieces.append(batch['hazard'])
                    if lo%320==0:print(json.dumps({'world_seed':seed,'split':sp,'roots':lo+batch['hazard'].shape[2],'total':len(xx)}),flush=True)
                # [head, selected/final, root, action, depth]
                hazard=torch.cat(pieces,dim=2)
                pp=labels[sp]
                subsets={'FIT':torch.ones(len(pp),dtype=torch.bool)} if sp=='fit' else {'DEV_A':meta['dev']['seed']%2==0,'DEV_B':meta['dev']['seed']%2==1}
                for part,use in subsets.items():
                    q=pp[use];h=hazard[:,:,use]
                    pred=1-torch.exp(-h.cumsum(-1));p16=q[...,-1]
                    opp=p16.amax(1)>p16.amin(1)
                    prev=F.pad(q,(1,0))[...,:-1];alive=1-prev;delta=q-prev
                    # softplus(z)=negative log survival. Exact expected time-to-event NLL.
                    log_dead=torch.log((-torch.expm1(-h)).clamp_min(1e-30))
                    nll=(delta[None,None]*-log_dead+(1-q)[None,None]*h).mean((-1,-2,-3))
                    choices=h.sum(-1).argmin(-1)
                    truth=p16.unsqueeze(0).unsqueeze(0).expand(3,2,-1,-1)
                    safe=1-truth.gather(-1,choices[...,None]).squeeze(-1)
                    seeds=meta[sp]['seed'][use][opp].numpy()
                    detail={}
                    for j,name in enumerate(('selected','final4000')):
                        sr=safe[:,j,opp].mean(0).numpy()
                        detail[name]={'safe':float(sr.mean()),'per_head_seed':safe[:,j,opp].mean(1).tolist(),
                            'hazard_BCE':float(nll[:,j].mean()),'P16_brier':float((pred[:,j,...,-1]-p16).square().mean()),
                            'allk_brier':float((pred[:,j]-q).square().mean()),'P16_action_error':ES.metrics(pred[:,j,...,-1],p16,opp)}
                    if part=='DEV_B':
                        for hs in range(3):
                            assert torch.equal(choices[hs,0],official['raw_scores'][f'trajectory_{hs}'].argmin(-1))
                    contrast=D.paired(safe[:,1,opp].mean(0).numpy(),safe[:,0,opp].mean(0).numpy(),seeds)
                    by_split[part]={'roots':int(use.sum()),'opportunities':int(opp.sum()),'label_mean':float(p16.mean()),
                        'prior':float((1-p16[:,prior])[opp].mean()),'models':detail,'final_minus_selected':contrast}
                    raw[seed][part]={'safe':safe[:,:,opp],'seeds':seeds,'choices':choices,'P16':p16}
                del xx,hazard,pieces
            out[str(seed)]=by_split
        result={'scope':__doc__,'contract':contract,'worlds':out}
        store.save('rows',raw,1);store.save('result',result,1)
        R.atomic_json(HERE/'evals/e17_h16_generalization.json',result)
        print(json.dumps({'worlds':out}),flush=True)


if __name__=='__main__':main()
