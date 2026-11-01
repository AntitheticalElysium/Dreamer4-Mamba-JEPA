"""Reproduce the teacher and inspect full-dev diagnostic target definitions."""
import json
from pathlib import Path

import numpy as np
import torch

from saccade import data as D, train as TR
from saccade.eval import s1


def main():
    torch.set_num_threads(4)
    out=Path(__file__).parent
    split=D.load_split(TR.load_cfg());off=D.frame_offsets(split['train'])
    targets=np.load('artifacts/saccade_m0/targets/change_targets.npy',mmap_mode='r')
    counts={k:{'episodes':len(v),'frames':int(D.frame_offsets(v)[-1])} for k,v in split.items()}
    keys={k:{e.key for e in v} for k,v in split.items()}
    assert all(not(keys[a]&keys[b]) for i,a in enumerate(keys) for b in list(keys)[i+1:])
    assert targets.shape==(int(off[-1]),7,9)
    assert np.isnan(targets[off[:-1]]).all()
    teacher_dir=out/'teacher_reproduction';teacher_dir.mkdir(exist_ok=True)
    model,report=D.fit_teacher(teacher_dir)
    a9=D._a9b();comparisons=[]
    # Check day/night and more than one episode independently of the original smoke.
    L=np.load('artifacts/eda/s0/s0_dev_labels.npz')
    state_off=np.r_[0,np.cumsum(L['ep_T']+1)]
    candidates=[]
    for e in range(len(L['ep_T'])):
        a,b=state_off[e:e+2]
        night=np.nonzero(L['s_light'][a:b]<.5)[0]
        if len(night):
            candidates.append((int(L['ep_shard'][e]),int(L['ep_slot'][e]),max(0,int(night[len(night)//2])-2)))
            if len(candidates)>=3:break
    candidates += [(e.shard,e.slot,0) for e in split['train'][:3]]
    for sh,sl,t in candidates:
        frames=D.RP.episode_fields(sh,sl)['observations'].numpy()[t:t+5]
        X,(pi,ri,qi)=D.pair_features(frames)
        manual=[]
        for j in range(len(frames)-1):
            c0,c1=a9.cells(frames[j]),a9.cells(frames[j+1]);dr,dc=a9.est_shift(c0,c1)
            for r in range(7):
                for q in range(9):
                    sr,sq=r-dr,q-dc
                    if not(0<=sr<7 and 0<=sq<9) or (r,q)==(3,4) or (sr,sq)==(3,4):continue
                    manual.append(a9.feats(c1[r,q][None],c0[sr,sq][None])[0])
        delta=float(np.max(np.abs(X-np.asarray(manual))))
        comparisons.append({'shard':sh,'slot':sl,'start':t,'cells':len(X),'feature_max_abs_delta':delta})
        assert delta==0
    # Stored labels agree with the reproduced teacher on sampled whole episodes.
    checks=[]
    for i in (0,len(split['train'])//2,len(split['train'])-1):
        ep=split['train'][i];fr=D.RP.episode_fields(ep.shard,ep.slot)['observations'].numpy()
        X,(pi,ri,qi)=D.pair_features(fr)
        pred=model.predict(X).astype(np.float16)
        saved=targets[off[i]+pi+1,ri,qi]
        delta=float(abs(pred-saved).max())
        checks.append({'train_episode_index':i,'cells':len(X),'stored_soft_target_max_abs_delta':delta})
        assert delta<.001
    dv=dict(L)
    T=dv['ep_T'];si=np.concatenate([np.arange(a,a+t) for a,t in zip(state_off[:-1],T)])
    dh=dv['s_body'][si+1,0].astype(float)-dv['s_body'][si,0].astype(float)
    reward=dv['t_reward'];unlock=np.rint(reward-.1*dh).astype(int)>0;large=reward>.45
    event={'transitions':len(reward),'unlocks':int(unlock.sum()),'large_rewards':int(large.sum()),
           'false_negative_unlocks':int((unlock&~large).sum()),'false_positive_unlocks':int((~unlock&large).sum()),
           'unlock_recall_of_reward_threshold':float(large[unlock].mean()),
           'perfect_reward_model_unlock_brier_under_current_evaluator':float(np.mean(large!=unlock))}
    dep,_=s1.departed(dv);depcheck={}
    pos=dv['s_pos'].astype(int)
    for k,kind in enumerate(('zombies','cows','skeletons','arrows')):
        rel=dv['s_'+kind+'_p'].astype(int)-pos[:,None,:]
        vis=dv['s_'+kind+'_m']&(abs(rel[...,0])<=3)&(abs(rel[...,1])<=4)
        anyvis=vis.any(1);recent=np.zeros(len(pos),bool)
        for a,b in zip(state_off[:-1],state_off[1:]):
            seen=np.maximum.accumulate(np.where(anyvis[a:b],np.arange(b-a),-10**9))
            age=np.arange(b-a)-seen
            recent[a:b]=~anyvis[a:b]&(age>=1)&(age<=64)
        depcheck[kind]={'observable_recently_seen_now_absent_frames':int(recent.sum()),
                        'current_target_none_among_those':int((recent&(dep[:,k]==0)).sum()),
                        'fraction_target_none':float((dep[recent,k]==0).mean())}
    result={'split_counts':counts,'target_shape':list(targets.shape),'teacher':report,
            'feature_checks':comparisons,'stored_target_checks':checks,'unlock_definition':event,
            'departed_target_scope':depcheck}
    (out/'data_evaluation_results.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
