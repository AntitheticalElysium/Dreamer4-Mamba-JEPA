"""Empirical online gaze, including train/test separation; no simulator input to training."""
import importlib
import json
from pathlib import Path
import sys
import numpy as np
import torch
from saccade import data as D, train as TR
C=importlib.import_module('artifacts.experiments.20261103_p1f2.chooser')
HERE=Path(__file__).resolve().parent


def main():
    ckpath=Path(sys.argv[1]);ck=torch.load(ckpath,map_location='cpu',weights_only=False)
    state=ck['fixation_controller'];raw=np.memmap(ckpath.parent/'gaze.bin',mode='r',dtype=C.TRACE)
    # A selected earlier checkpoint must not be credited with later gaze/training.
    z=raw[:state['trace_rows']];cfg=ck['cfg'];eps=D.load_split(cfg)['train'];off=D.frame_offsets(eps)
    assert len(z)==ck['state']['frames'] and state['frames']==ck['state']['frames']
    win=C.Q.WIN.numpy();out={}
    for band in range(10):
        q=z[len(z)*band//10:len(z)*(band+1)//10];hist=np.bincount(q['window'],minlength=49)
        cells=win[q['window']];baseline=win[q['uniform']]
        out[str(band+1)]=dict(frames=len(q),window_histogram=hist.tolist(),
            sampled_HUD_cell_share=float((cells>=63).mean()),uniform_HUD_cell_share=float((baseline>=63).mean()),
            chosen_probability_mean=float(q['probability'].mean()),changed_from_paired_uniform=float((q['window']!=q['uniform']).mean()))
    # The first 400 train-label episodes are a diagnostic subset, excluding monitor.
    labels_path=TR.ROOT/'artifacts/eda/s0/s0_train400_labels.npz';lab=dict(np.load(labels_path))
    keys={(e.shard,e.slot):i for i,e in enumerate(eps)};mapping={}
    loff=np.r_[0,np.cumsum(lab['ep_T']+1)]
    for e,key in enumerate(zip(lab['ep_shard'],lab['ep_slot'])):
        if tuple(key) in keys:mapping[keys[tuple(key)]]=e
    sums={};counts={};seen=0
    for start in range(0,len(z),100000):
        q=z[start:start+100000];target=q['target'];ei=np.searchsorted(off,target,side='right')-1
        le=np.array([mapping.get(int(e),-1) for e in ei]);take=(target>=0)&(le>=0)
        q=q[take];target=target[take];ei=ei[take];le=le[take]
        if not len(q):continue
        t=target-off[ei];li=loff[le]+t;pos=lab['s_pos'][li];visible=np.zeros((len(q),81),bool)
        kinds={}
        for kind in ('zombies','cows','skeletons','arrows'):
            rel=lab['s_'+kind+'_p'][li]-pos[:,None,:]+np.array([3,4]);good=lab['s_'+kind+'_m'][li]&(rel[...,0]>=0)&(rel[...,0]<7)&(rel[...,1]>=0)&(rel[...,1]<9)
            grid=np.zeros_like(visible);rr,cc=np.where(good);grid[rr,rel[rr,cc,0]*9+rel[rr,cc,1]]=True
            kinds[kind]=grid;visible|=grid
        for name,grid in dict(entities=visible,**kinds).items():
            present=grid.any(-1);n=int(present.sum())
            for policy,key in (('learned','window'),('paired_uniform','uniform')):
                hit=grid[np.arange(len(q))[:,None],win[q[key]]].any(-1)&present
                tag=name+'_'+policy;sums[tag]=sums.get(tag,0)+int(hit.sum());counts[tag]=counts.get(tag,0)+n
        seen+=len(q)
    result=dict(checkpoint=str(ckpath),sha256=TR.file_sha256(ckpath),world_frames=state['frames'],head_updates=state['updates'],
        deciles=out,training_label_subset_frames=seen,
        entity_contact={k:dict(hits=sums[k],frames_with_kind=counts[k],rate=sums[k]/counts[k] if counts[k] else None) for k in sums},
        note='Descriptive train-subset coverage, not held-out world benefit; first frames without target IDs excluded.')
    dest=HERE/'readings/chooser_trace.json';dest.parent.mkdir(exist_ok=True);dest.write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':main()
