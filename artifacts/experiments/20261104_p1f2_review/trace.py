"""Actual F2 acquisition, paired uniform opportunity, and physical-episode CIs."""
import os
os.environ.setdefault('JAX_PLATFORMS','cpu')
import importlib,json
from pathlib import Path
import numpy as np
import torch
from saccade import data as D
from saccade.eval import s1 as S,gates as G
C=importlib.import_module('artifacts.experiments.20261103_p1f2.chooser')
A=importlib.import_module('artifacts.experiments.20261103_gaze_belief.analyze')
F=importlib.import_module('artifacts.experiments.20261031_learned_fixation_sweep.locus')
HERE=Path(__file__).resolve().parent;RUN=HERE.parent/'20261103_p1f2'

def training_semantics(labels,eall,tall,si,loff):
    """Cell ownership for all labelled training frames, including rare overlaps.

    Renderer draws mobs after the player. One train frame has an entity at the
    player cell, so foreground gets precedence rather than double ownership.
    """
    grid=np.concatenate([x['grid'][1:].reshape(-1,63) for x in A.E]);oldgrid=np.concatenate([x['grid'][:-1].reshape(-1,7,9) for x in A.E])
    oldview=np.concatenate([x['view'][:-1].reshape(-1,7,9) for x in A.E]);view=labels['s_view'][si].reshape(-1,63)
    n=len(grid);cats={}
    for k,kind in enumerate(G.KINDS,1):cats[kind]=np.pad(grid==k,((0,0),(0,18)))
    mob=grid>0;player=np.zeros((n,81),bool);player[:,31]=~mob[:,31];cats['player']=player
    terra=np.pad(~mob,((0,0),(0,18)))&~player
    for name,values in [('terrain_resources',[3,4,5,8,9,10,15,16]),('terrain_structures',[11,12]),('terrain_common',[2,7,13]),('terrain_other',[0,1,6,14])]:
        cats[name]=np.pad(np.isin(view,values),((0,0),(0,18)))&terra
    for name,cells in [('health',[63]),('food',[64]),('drink',[65]),('energy',[66]),('inventory',list(range(67,79))),('blank_hud',[79,80])]:
        a=np.zeros((n,81),bool);a[:,cells]=True;cats[name]=a
    assert np.all(sum(cats.values())==1)
    mobs=np.pad(mob,((0,0),(0,18)));rr,cc=np.indices((7,9));delta=labels['s_pos'][si]-labels['s_pos'][si-1]
    pr=rr[None]+delta[:,0,None,None];pc=cc[None]+delta[:,1,None,None]
    valid=(pr>=0)&(pr<7)&(pc>=0)&(pc<9);pr=pr.clip(0,6);pc=pc.clip(0,8)
    olds=oldgrid[np.arange(n)[:,None,None],pr,pc];oldv=oldview[np.arange(n)[:,None,None],pr,pc]
    clear=(olds==0)&~((pr==3)&(pc==4));edit=(view.reshape(-1,7,9)!=oldv)&valid&clear
    hud=np.zeros((n,81),bool)
    for cell,(key,col) in A.HUD_SRC.items():hud[:,cell]=labels[key][si,col].clip(0,9)!=labels[key][si-1,col].clip(0,9)
    r,c=np.divmod(np.arange(81),9);near=np.maximum(abs(r-3),abs(c-4))<=2
    masks={**cats,'any_entity':mobs,'near_entity':mobs&near[None],
        'entering_map':np.pad((~valid).reshape(n,63),((0,0),(0,18))),
        'terrain_edit':np.pad(edit.reshape(n,63),((0,0),(0,18)))&terra,'changed_hud':hud}
    return dict(categories=cats,masks=masks,light=labels['s_light'][si],
        action=labels['t_action'][A.AOFF[eall]+tall-1],player_foreground_overlaps=int(mob[:,31].sum()))

def main():
    torch.set_num_threads(2);ck=torch.load(RUN/'run/ckpt_10.pt',map_location='cpu',weights_only=False)
    raw=np.memmap(RUN/'run/gaze.bin',dtype=C.TRACE,mode='r',shape=(ck['fixation_controller']['trace_rows'],))
    # Reconstruct time within each actual episode UID, never consecutive rows
    # of interleaved streams. Screen repeats are not world-coordinate revisits.
    order=np.argsort(raw['episode'],kind='stable');uid=raw['episode'][order]
    w=raw['window'][order].astype(int);u=raw['uniform'][order].astype(int);same=uid[1:]==uid[:-1]
    temporal={}
    for name,x in [('learned',w),('paired_uniform',u)]:
        repeat=same&(x[1:]==x[:-1]);dist=abs(x[1:]//7-x[:-1]//7)+abs(x[1:]%7-x[:-1]%7)
        changes=np.r_[True,~repeat];st=np.flatnonzero(changes);length=np.diff(np.r_[st,len(x)])
        temporal[name]=dict(same_screen_window=float(repeat.sum()/same.sum()),mean_screen_center_L1=float(dist[same].mean()),
            run_length_median=float(np.median(length)),run_length_p95=float(np.quantile(length,.95)),
            run_length_max=int(length.max()),runs_ge_5=int((length>=5).sum()))
    del order,uid,w,u
    eps=D.load_split(ck['cfg'])['train'];off=D.frame_offsets(eps)
    labels=dict(np.load(S.LABELS/'s0_train400_labels.npz'));keys={(e.shard,e.slot):i for i,e in enumerate(eps)}
    mapping=np.full(len(eps),-1,int)
    for le,key in enumerate(zip(labels['ep_shard'],labels['ep_slot'])):
        if tuple(key) in keys:mapping[keys[tuple(key)]]=le
    kept=[];positions=[]
    for start in range(0,len(raw),262144):
        q=raw[start:start+262144];ei=np.searchsorted(off,q['target'],side='right')-1
        take=(q['target']>=0)&(mapping[ei]>=0)
        kept.append(np.array(q[take]));positions.append(np.flatnonzero(take)+start)
    q=np.concatenate(kept);position=np.concatenate(positions);ei=np.searchsorted(off,q['target'],side='right')-1
    ep=mapping[ei];tt=q['target']-off[ei];loff=np.r_[0,np.cumsum(labels['ep_T']+1)]
    assert ((tt>=1)&(tt<=labels['ep_T'][ep])).all()
    A.L=labels;A.E=G.episode_grids(labels);A.OFF=loff;A.AOFF=np.r_[0,np.cumsum(labels['ep_T'])]
    eall=np.concatenate([np.full(int(T),e) for e,T in enumerate(labels['ep_T'])])
    tall=np.concatenate([np.arange(1,int(T)+1) for T in labels['ep_T']]);si=loff[eall]+tall
    direction=F.FC.DIRS[labels['s_dir'][si-1]];faced=(3+direction[:,0])*9+4+direction[:,1]
    meta=dict(ep=eall,t=tall,cls=np.full(len(tall),'random'),counter=np.full(len(tall),-1),faced=faced)
    sem=training_semantics(labels,eall,tall,si,loff);lookup=np.full(int(loff[-1]),-1,int);lookup[si]=np.arange(len(si));rows=lookup[loff[ep]+tt]
    assert (rows>=0).all();bins=position//(len(raw)//10);assert bins.max()==9
    masks=sem['masks'];groups={'all':np.ones(len(q),bool),'day':sem['light'][rows]>=.5,'night':sem['light'][rows]<.5,
        'move':np.isin(sem['action'][rows],[1,2,3,4]),'noop':sem['action'][rows]==0,'do':sem['action'][rows]==5}
    groups.update({f'decile_{i+1}':bins==i for i in range(10)})
    # Add correctly located consequence counters, with success/failure separate.
    before=si-1;face=labels['s_view'][before,3+direction[:,0],4+direction[:,1]]
    awake_do=(sem['action']==5)&~labels['s_sleep'][before].astype(bool)
    for cl,(block,cell,tool) in F.ATTEMPTS.items():
        key,col=A.HUD_SRC[cell];attempt=awake_do&(face==block)&(labels[key][before,col]<9)
        if tool is not None:attempt&=labels['s_inv'][before,tool]>0
        inc=labels[key][si,col].clip(0,9)==labels[key][before,col].clip(0,9)+1
        mask=np.zeros((len(si),81),bool);mask[attempt,cell]=True;masks['counter_'+cl]=mask
        for label,cond in [('attempt',attempt),('increment',attempt&inc),('no_increment',attempt&~inc),
                           ('first_increment',attempt&inc&(labels[key][before,col]==0)),
                           ('later_increment',attempt&inc&(labels[key][before,col]>0))]:groups[cl+'_'+label]=cond[rows]
    cat={};cov={};win=F.WIN.numpy();center=F.CENTRES.numpy()[:,0]*9+F.CENTRES.numpy()[:,1]
    windows={'learned':q['window'],'paired_uniform':q['uniform']}
    for kind,mask in sem['categories'].items():
        v=[]
        for name,x in windows.items():
            v.append(mask[rows,center[x]].astype(float));v.append(mask[rows[:,None],win[x]].mean(-1))
        # Local arrays are retained for paired grouping, not a simulated policy.
        values=np.stack(v,-1);cat[kind]={}
        for label,sel in groups.items():
            if label not in ['all','day','night','move','noop','do'] and not label.startswith('decile'):continue
            if not sel.any():continue
            rr=A.estimates(values[sel],np.ones(sel.sum()),ep[sel])
            cat[kind][label]=dict(zip(['learned_center','learned_cell_share','uniform_center','uniform_cell_share'],rr))
    # Crop contact conditional on exposure, with paired differences and all
    # repetitions clustered under the original physical source episode.
    wanted=['any_entity','near_entity','zombies','cows','skeletons','arrows','entering_map','terrain_edit','changed_hud']+[k for k in masks if k.startswith('counter_')]
    for kind in wanted:
        mask=masks[kind];available=mask.any(-1)[rows];hits=[]
        for name,x in windows.items():hits.append(mask[rows[:,None],win[x]].any(-1))
        values=np.stack([hits[0],hits[1],hits[0].astype(float)-hits[1]],-1);cov[kind]={}
        for label,sel in groups.items():
            basic=label in ['all','day','night','move','noop','do'] or label.startswith('decile_')
            if not basic and (not kind.startswith('counter_') or not label.startswith(kind[8:]+'_')):continue
            take=sel&available
            if not take.any():continue
            rr=A.estimates(values[take],np.ones(take.sum()),ep[take]);cov[kind][label]=dict(zip(['learned','paired_uniform','difference'],rr))
    logs=[json.loads(l) for l in (RUN/'run/chooser.jsonl').read_text().splitlines()];clock={}
    for b in range(10):
        ll=[l for l in logs if b*2560000<l['world_frames']<=(b+1)*2560000];weight=np.array([l['frames'] for l in ll])
        clock[str(b+1)]={k:float(np.average([l[k] for l in ll],weights=weight)) for k in ['C','S','entropy','head_loss']}
    result=dict(world_frames=len(raw),labelled_training_occurrences=len(q),unique_labelled_frames=len(np.unique(q['target'])),
        physical_label_episodes=len(np.unique(ep)),categories=cat,coverage=cov,temporal=temporal,logged_clock=clock,
        cell_partition_check=True,player_foreground_overlaps=sem['player_foreground_overlaps'],
        note='Executed training gaze, not a held-out teaching effect. CIs cluster repeated passes by physical source episode. Screen-coordinate dwell only.')
    (HERE/'TRACE.json').write_text(json.dumps(result,indent=2)+'\n')
    np.savez_compressed(HERE/'trace_subset.npz',trace=q,position=position,ep=ep,t=tt,decile=bins)
    print('training occurrences',len(q),'unique frames',result['unique_labelled_frames'],'physical episodes',result['physical_label_episodes'],flush=True)
    print('temporal',temporal,flush=True)
    for kind in wanted[:9]:
        v=cov[kind]['all'];print(kind,'learned',v['learned']['mean'],'uniform',v['paired_uniform']['mean'],'delta',v['difference'],flush=True)

if __name__=='__main__':main()
