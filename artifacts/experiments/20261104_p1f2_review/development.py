"""Replay the ten saved ONLINE heads with their own encoders. No fitting."""
import os
os.environ.setdefault('JAX_PLATFORMS','cpu')
os.environ.setdefault('MAMBA_DETERMINISTIC','1')
import importlib,json,gc,time
from pathlib import Path
import numpy as np
import torch
from saccade import model as M,train as TR
C=importlib.import_module('artifacts.experiments.20261103_p1f2.chooser')
P=importlib.import_module('artifacts.experiments.20261103_chooser_development.common')
A=importlib.import_module('artifacts.experiments.20261103_gaze_belief.analyze')
HERE=Path(__file__).resolve().parent;RUN=HERE.parent/'20261103_p1f2'

def stats(meta,forecast,data,ids):
    f=forecast;win=P.WIN;counter=meta['counter'];valid=counter>=0
    idx=np.arange(len(counter));cc=counter.clip(min=0)
    target=data.z['p'][ids].float().cpu().numpy()
    last=data.z['p'][ids-1].float().cpu().numpy()
    pred=f['prediction'].astype(np.float32);noop=f['noop'].astype(np.float32)
    # prediction/noop are the persisted half-precision diagnostic exports, as in
    # the original development campaign. C/S/maps remain full precision exports.
    v=dict(counter_error=((pred[idx,cc]-target[idx,cc])**2).sum(-1),
        counter_copy_error=((last[idx,cc]-target[idx,cc])**2).sum(-1),
        counter_C=((pred[idx,cc]-noop[idx,cc])**2).sum(-1),
        counter_noop_copy_error=((last[idx,cc]-noop[idx,cc])**2).sum(-1),
        counter_alignment=((pred[idx,cc]-last[idx,cc])*(target[idx,cc]-last[idx,cc])).sum(-1),
        counter_C_fraction=((pred[idx,cc]-noop[idx,cc])**2).sum(-1)/((pred-noop)**2).sum((-1,-2)).clip(min=1e-12),
        counter_S_fraction=((pred[idx,cc]-target[idx,cc])**2).sum(-1)/((pred-target)**2).sum((-1,-2)).clip(min=1e-12))
    label=A.metadata(meta);v.update(increment=label['increment'],counter_before=label['counter_before'])
    result={}
    for cl in np.unique(meta['cls']):
        mask=(meta['cls']==cl)&valid
        if not mask.any():continue
        groups={'attempt':mask,'increment':mask&v['increment'],
            'first_increment':mask&v['increment']&(v['counter_before']==0),
            'later_increment':mask&v['increment']&(v['counter_before']>0),
            'no_increment':mask&~v['increment']}
        result[str(cl)]={}
        for name,sel in groups.items():
            if not sel.any():continue
            den=v['counter_copy_error'][sel].mean()
            result[str(cl)][name]=dict(n=int(sel.sum()),episodes=len(np.unique(meta['ep'][sel])),
                **{k:float(x[sel].mean()) for k,x in v.items() if k not in ['increment','counter_before']},
                normalized_error=float(v['counter_error'][sel].mean()/den) if den>1e-12 else None,
                normalized_C=float(v['counter_C'][sel].mean()/den) if den>1e-12 else None,
                normalized_noop_drift=float(v['counter_noop_copy_error'][sel].mean()/den) if den>1e-12 else None,
                normalized_alignment=float(v['counter_alignment'][sel].mean()/den) if den>1e-12 else None)
    return result,v

@torch.no_grad()
def main():
    torch.set_num_threads(4);Q=C.Q;cohorts=P.cohorts()
    n=max(int(q['ep'].max())+1 for q in cohorts.values());results={};parity={}
    for checkpoint in range(1,11):
        start=time.time();path=RUN/f'run/ckpt_{checkpoint:02d}.pt'
        ck=torch.load(path,map_location='cpu',weights_only=False);out=HERE/f'ck{checkpoint:02d}'
        out.mkdir(exist_ok=True);Q.HERE=out;Q.CK=path
        if checkpoint==10:
            cache=out/f'dev_{n}_coarse.pt';source=(RUN/f'readings/head/dev_{n}_coarse.pt').resolve()
            if not cache.exists():cache.symlink_to(source)
        model=M.M0(ck['cfg']).cuda().eval().requires_grad_(False);model.load_state_dict(ck['model'])
        head=Q.Head().cuda().eval().requires_grad_(False);head.load_state_dict(ck['fixation_controller']['head'])
        data=Q.Data(Q.collect(model,ck['cfg'],'dev',n,torch.device('cuda')),torch.device('cuda'))
        entry=dict(checkpoint_sha256=TR.file_sha256(path),world_frames=ck['state']['frames'],
            head_updates=ck['fixation_controller']['updates'],cohorts={})
        for name,meta in cohorts.items():
            ids=data.row_ids(meta['ep'],meta['t']);forecast=Q.predict(head,data,ids)
            counter,extra=stats(meta,forecast,data,ids)
            np.savez_compressed(out/f'{name}.npz',**meta,**forecast,**extra)
            full=P.summarize(meta,forecast['C'],forecast['S'],dict(mse=forecast['mse'],copy_mse=forecast['copy_mse']))
            entry['cohorts'][name]=dict(policy={k:dict(n=v['n'],episodes=v['episodes'],
                combined=v['policies']['existing'],uniform=v['policies']['uniform'],
                components=v['components'],errors=v.get('errors',{})) for k,v in full.items()},counter=counter)
            if checkpoint==10:
                old=dict(np.load(RUN/f'readings/head/{name}.npz'))
                assert all(np.array_equal(meta[k],old[k]) for k in meta)
                parity[name]={k:float(np.max(np.abs(forecast[k].astype(float)-old[k].astype(float)))) for k in ['C','S','prob','mse','prediction','noop']}
                assert max(parity[name].values())==0,parity[name]
        entry['seconds']=time.time()-start;results[str(checkpoint)]=entry
        (HERE/'DEVELOPMENT.json').write_text(json.dumps(results,indent=2)+'\n')
        print('checkpoint',checkpoint,'head updates',entry['head_updates'],'seconds',round(entry['seconds'],1),flush=True)
        for cl in ['drink','wood','stone','coal','iron']:
            q=entry['cohorts']['confirmation']['policy'][cl];s=counter.get(cl,{}).get('increment',{})
            print(cl,'remote',round(q['combined']['remote_mass']['mean'],3),'C top',round(q['components']['C']['remote_argmax'],3),
                'counter error/copy',s.get('normalized_error'),'C/copy',s.get('normalized_C'),flush=True)
        del data,head,model,ck;gc.collect();torch.cuda.empty_cache()
    (HERE/'ENDPOINT_PARITY.json').write_text(json.dumps(parity,indent=2)+'\n')

if __name__=='__main__':main()
