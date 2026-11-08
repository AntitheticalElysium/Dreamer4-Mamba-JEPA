"""Post-result source-observation qualification only, no new forecast or fitting."""
import os
os.environ.setdefault('JAX_PLATFORMS','cpu')
import importlib
import json
from pathlib import Path
import numpy as np
import torch
from saccade import model as M, train as TR, data as D
from saccade.eval import s1 as S

R=importlib.import_module('artifacts.experiments.20261103_p1f16_review.local_uptake')
HERE=Path(__file__).resolve().parent


@torch.no_grad()
def main():
    torch.set_num_threads(4);source,weights=R.fixed_cohort();ck=R.WORLDS['p1f16'];R.U.CK=ck;m,cfg=R.U.load()
    reader=S.load_reader(ck.parent/'s1/results.reader.pt',ck,R.U.DEV)
    labels=S.first_episodes(dict(np.load(S.LABELS/'s0_dev_labels.npz')),250);eps=S.episodes(labels)
    frames=torch.stack([D.RP.episode_fields(eps[e].shard,eps[e].slot)['observations'][t] for e,t in zip(source['episode'],source['t'])]).to(R.U.DEV)
    with TR.autocast(R.U.DEV,cfg):
        p=m.perception(frames,torch.ones(len(frames),2,device=R.U.DEV,dtype=torch.long))
    pred=reader(torch.cat([p.p[:,:63].float().flatten(0,1),p.v.float().repeat_interleave(63,0)],-1))['kind'].argmax(-1).reshape(len(frames),63).cpu().numpy()
    anymiss=[];allmiss=[];records=[]
    for i in range(len(frames)):
        cells=np.unique(R.A.WIN[source['useful_cells'][i]])
        misses=pred[i,cells]!=source['truth'][i,cells]
        anymiss.append(misses.any());allmiss.append(misses.all())
        records.append(dict(episode=int(source['episode'][i]),t=int(source['t'][i]),cells=cells.tolist(),truth=source['truth'][i,cells].tolist(),read=pred[i,cells].tolist()))
    new=dict(np.load(HERE/'p1f16/local.npz'));readable=new['source_readable']==1
    subsets={'fine_readable_and_any_coarse_miss':readable&anymiss,'fine_readable_and_all_coarse_miss':readable&allmiss}
    result=dict(source_only=True,post_result_descriptive=True,rows=records,subsets={})
    for key,mask in subsets.items():
        out={}
        for world in R.WORLDS:
            z=dict(np.load(HERE/world/'local.npz'));a,b=z['weights'][:,0],z['weights'][:,1]
            d=np.einsum('isnwg,iw->isng',z['loss'],a-b);base=np.einsum('isnwg,iw->isng',z['loss'],b)
            out[world]={}
            for band,sl in (('all',slice(None)),('high',slice(-2,None))):
                f=d[:,:,sl,1].mean((1,2))/base[:,:,sl,1].mean((1,2))
                out[world][band]=R.A.estimate(f[mask],z['episode'][mask])
        result['subsets'][key]=out
    result['source_sha256']=TR.file_sha256(__file__);result['native_reader_sha256']=TR.file_sha256(ck.parent/'s1/results.reader.pt')
    (HERE/'source_checks.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result['subsets'],indent=2))


if __name__=='__main__':main()
