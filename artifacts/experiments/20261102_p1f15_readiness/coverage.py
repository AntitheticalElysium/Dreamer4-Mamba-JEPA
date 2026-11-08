"""Read-only coverage of the proposed exogenous post-training windows."""
import os
os.environ.setdefault('JAX_PLATFORMS','cpu')
import json
from pathlib import Path
import numpy as np
import craftax.craftax_classic.constants as C
from saccade import data as D, train as TR

HERE=Path(__file__).resolve().parent
def count(path,keep=None):
    f=np.load(path)
    z={k:f[k] for k in ('ep_T','ep_shard','ep_slot','t_exogenous','t_reward','t_terminated','s_ach')}
    ao=np.r_[0,np.cumsum(z['ep_T'])];so=np.r_[0,np.cumsum(z['ep_T']+1)];out={}
    for h in (2,4,8,16):
        rows=[(e,t) for e,N in enumerate(z['ep_T']) for t in range(0,int(N)-h+1,h)
              if (keep is None or (int(z['ep_shard'][e]),int(z['ep_slot'][e])) in keep)
              and z['t_exogenous'][ao[e]+t:ao[e]+t+h].all()]
        inds=np.concatenate([np.arange(ao[e]+t,ao[e]+t+h) for e,t in rows])
        si=np.concatenate([np.arange(so[e]+t,so[e]+t+h) for e,t in rows])
        delta=z['s_ach'][si+1]>z['s_ach'][si]
        out[h]={'windows':len(rows),'episodes':len({e for e,t in rows}),'transitions':len(inds),
            'positive_reward':int((z['t_reward'][inds]>.5).sum()),'death':int(z['t_terminated'][inds].sum()),
            'terminal_counts_by_step':z['t_terminated'][inds].reshape(-1,h).sum(0).tolist(),
            'achievement_counts':{n:int(delta[:,getattr(C.Achievement,n).value].sum()) for n in
                ('COLLECT_WOOD','COLLECT_DRINK','PLACE_TABLE','WAKE_UP','DEFEAT_ZOMBIE','COLLECT_COAL','COLLECT_IRON')}}
    # All sliding starts, so a stride alignment cannot manufacture a coverage hole.
    sliding={}
    names=('COLLECT_WOOD','COLLECT_DRINK','PLACE_TABLE','WAKE_UP','DEFEAT_ZOMBIE','COLLECT_COAL','COLLECT_IRON')
    for h in (2,4,8,16):
        totals=np.zeros(9,dtype=np.int64);appearances=np.zeros(9,dtype=np.int64)
        nw=ne=nt=0;terminal_by_step=np.zeros(h,dtype=np.int64)
        for e,N in enumerate(z['ep_T']):
            if keep is not None and (int(z['ep_shard'][e]),int(z['ep_slot'][e])) not in keep:continue
            ex=z['t_exogenous'][ao[e]:ao[e+1]]
            bad=np.r_[0,np.cumsum(~ex)]
            starts=np.flatnonzero((bad[h:]-bad[:-h])==0)
            if not len(starts):continue
            mult=(np.bincount(starts,minlength=N+1)-np.bincount(starts+h,minlength=N+1)).cumsum()[:-1]
            delta=z['s_ach'][so[e]+1:so[e+1]]>z['s_ach'][so[e]:so[e+1]-1]
            flags=np.stack([z['t_reward'][ao[e]:ao[e+1]]>.5,z['t_terminated'][ao[e]:ao[e+1]],
                            *[delta[:,getattr(C.Achievement,n).value] for n in names]],-1)
            totals+=flags[mult>0].sum(0)
            appearances+=(flags*mult[:,None]).sum(0)
            terminal_by_step+=flags[starts[:,None]+np.arange(h)[None],1].sum(0)
            nw+=len(starts);ne+=1;nt+=int((mult>0).sum())
        sliding[h]={'windows':nw,'episodes':ne,'unique_transitions':nt,
            'unique_positive_reward':int(totals[0]),'unique_death':int(totals[1]),
            'terminal_counts_by_step':terminal_by_step.tolist(),
            'unique_achievement_counts':dict(zip(names,totals[2:].tolist())),
            'expected_achievement_exposures_in_24000_uniform_windows':dict(zip(names,(appearances[2:]/nw*24000).tolist()))}
    return {'canonical_stride_H':out,'all_sliding_starts':sliding}

def main():
    keep={(e.shard,e.slot) for e in D.load_split(TR.load_cfg())['train']}
    out={'dev':count('artifacts/eda/s0/s0_dev_labels.npz'),
         'train_excluding_monitor':count('artifacts/eda/s0/s0_train_labels.npz',keep)}
    (HERE/'exogenous_coverage.json').write_text(json.dumps(out,indent=1))
    p=Path('artifacts/experiments/20261030_fovea_placement/map.json');z=json.loads(p.read_text())
    assert z['q']==.5
    classes={k:{'uniform_coverage':v['covered_prior'],'guided_q1_expected':2*v['covered_guided']-v['covered_prior'],
                'target_frames':v['frames']} for k,v in z['predicted_coverage'].items() if k!='all_targets'}
    classes['all_targets']={'uniform_coverage':z['predicted_coverage']['all_targets']['prior'],
        'guided_q1_expected':2*z['predicted_coverage']['all_targets']['guided']-z['predicted_coverage']['all_targets']['prior']}
    (HERE/'coverage.json').write_text(json.dumps({'scope':'Expected event-crop coverage from saved training corpus/rule, not Adam impact',
        'attempt_frames_fraction':z['frames_in_attempt_contexts'],'classes':classes},indent=1))
    print(json.dumps(out,indent=1))

if __name__=='__main__':main()
