"""CPU descriptive census of saved labels and actual F2 training exposure.

No simulator rollout, fitted model, or scientific intervention. State/transition
alignment is asserted per episode against the actual training manifest.
"""
import hashlib
import gzip
import json
from pathlib import Path
import sys

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from saccade import data as D
from craftax.craftax_classic.constants import Achievement

HERE = Path(__file__).resolve().parent
LABELS = ROOT / 'artifacts/eda/s0/s0_train_labels.npz'
TRACE = ROOT / 'artifacts/experiments/20261103_p1f2/run/gaze.bin'
HEAD_LOG = TRACE.with_name('chooser.jsonl')
DTYPE = np.dtype([('target','<i4'),('episode','<i4'),('window','u1'),
                  ('uniform','u1'),('C','u1'),('S','u1'),('probability','<f4')])
DIRS = np.array([[0,0],[0,-1],[0,1],[-1,0],[1,0]])
COLLECT = {'drink':(3,'s_body',2,None), 'wood':(5,'s_inv',0,None),
           'sapling':(2,'s_inv',5,None), 'stone':(4,'s_inv',1,6),
           'coal':(8,'s_inv',2,6), 'iron':(9,'s_inv',3,7)}


def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(1<<20),b''): h.update(b)
    return h.hexdigest()


def main():
    cfg=yaml.safe_load((ROOT/'saccade/m0.yaml').read_text())
    split=D.load_split(cfg);train=split['train'];off=D.frame_offsets(train)
    z=np.load(LABELS)
    keys=('ep_T','ep_shard','ep_slot','ep_eps','s_body','s_inv','s_view',
          's_sleep','s_dir','s_pos','s_light','s_ach','s_zombies_p',
          's_zombies_m','s_zombies_cd','t_action','t_reward','t_terminated')
    L={k:z[k] for k in keys}
    so=np.r_[0,np.cumsum(L['ep_T']+1)];ao=np.r_[0,np.cumsum(L['ep_T'])]
    meta={(int(a),int(b)):i for i,(a,b) in enumerate(zip(L['ep_shard'],L['ep_slot']))}
    rows=[];names=None;flags=None;value_bands={name:{} for name in COLLECT}
    for e,ep in enumerate(train):
        j=meta[ep.shard,ep.slot];T=int(L['ep_T'][j]);assert T==ep.T
        g=np.arange(so[j],so[j]+T);ng=g+1;tr=np.arange(ao[j],ao[j]+T)
        raw=D.RP.episode_fields(ep.shard,ep.slot)
        for label,field in [('t_action','actions_taken'),('t_reward','rewards'),('t_terminated','terminated')]:
            assert np.array_equal(L[label][tr],raw[field].numpy()),(e,label)
        h=L['s_body'][g,0];hn=L['s_body'][ng,0]
        adj=(np.abs(L['s_zombies_p'][g]-L['s_pos'][g,None]).sum(-1)==1)&L['s_zombies_m'][g].astype(bool)
        awake=~L['s_sleep'][g].astype(bool);do=(L['t_action'][tr]==5)&awake
        delta=DIRS[L['s_dir'][g]]
        face=L['s_view'][g,3+delta[:,0],4+delta[:,1]]
        terminated=L['t_terminated'][tr].astype(bool);dead=hn<=0
        risk=(h<=2)&adj.any(-1)
        events={'damage':hn<h, 'death':dead, 'terminated':terminated,
                'risk':risk,'risk_damage':risk&(hn<h), 'risk_death':risk&dead,
                'risk_ready':risk&((L['s_zombies_cd'][g]<=0)&adj).any(-1),
                'adjacent':(h>=3)&adj.any(-1)&awake,
                'low_health':h<=2,'positive_reward':L['t_reward'][tr]>0}
        for name,(block,key,col,tool) in COLLECT.items():
            before=L[key][g,col].astype(int);after=L[key][ng,col].astype(int)
            attempt=do&(face==block)&(before<9)
            if tool is not None: attempt&=L['s_inv'][g,tool]>0
            inc=after>before
            events[name+'_attempt']=attempt
            events[name+'_increment']=inc
            events[name+'_attempt_success']=attempt&inc
            events[name+'_first']=inc&(before==0)
            for value,count in zip(*np.unique(before[inc],return_counts=True)):
                key=str(int(value));value_bands[name][key]=value_bands[name].get(key,0)+int(count)
        unlock=L['s_ach'][ng].astype(bool)&~L['s_ach'][g].astype(bool)
        for a in Achievement:
            events['unlock_'+a.name]=unlock[:,a.value]
        assert np.array_equal(terminated,dead), ('termination differs from death',e)
        if names is None:
            names=list(events);assert len(names)<=64
            flags=np.zeros(off[-1],np.uint64)
        assert list(events)==names
        f=np.zeros(T,np.uint64)
        for bit,name in enumerate(names):f|=events[name].astype(np.uint64)<<bit
        flags[off[e]+1:off[e+1]]=f
        rows.append(dict(episode=e,epsilon=round(float(L['ep_eps'][j]),2),transitions=T,
                         counts={k:int(v.sum()) for k,v in events.items()},
                         after_prefix={k:int(v[7:].sum()) for k,v in events.items()},
                         night={k:int((v&(L['s_light'][g]<.5)).sum()) for k,v in events.items()}))
    ntr=sum(x['transitions'] for x in rows)
    def summarize(rr):
        total=sum(x['transitions'] for x in rr)
        return dict(episodes=len(rr),transitions=total,events={name:dict(
            unique=sum(x['counts'][name] for x in rr),
            episodes=sum(x['counts'][name]>0 for x in rr),
            after_prefix=sum(x['after_prefix'][name] for x in rr),
            night=sum(x['night'][name] for x in rr),
            per_million=1e6*sum(x['counts'][name] for x in rr)/total) for name in names})
    out={'labels_sha256':sha(LABELS),'script_sha256':sha(Path(__file__)),
         'training_manifest_sha256':sha(D.RP.STORE/'manifest.json'),
         'monitor_episodes':len(split['monitor']),'summary':summarize(rows),
         'by_epsilon':{str(v):summarize([x for x in rows if x['epsilon']==v]) for v in sorted(set(x['epsilon'] for x in rows))},
         'increment_prior_value_counts':value_bands,
         'transition_alignment':'Every selected label action/reward/termination array equals its raw stored episode array; state T+1 versus transition T lengths asserted. All termination flags equal after-health <= 0.'}
    assert ntr==int(off[-1]-len(train))
    trace=np.memmap(TRACE,mode='r',dtype=DTYPE)
    seen=np.zeros(off[-1],np.int64);reset=0
    for start in range(0,len(trace),1_000_000):
        target=trace['target'][start:start+1_000_000]
        reset+=int((target<0).sum());target=target[target>=0]
        assert target.max()<off[-1]
        seen+=np.bincount(target,minlength=off[-1])
    assert seen.sum()+reset==len(trace)
    assert (seen[off[:-1]]==0).all()
    expected_seen=expected_head_exposure(trace,len(seen))
    out['trace']={'sha256':sha(TRACE),'rows':len(trace),'episode_starts':reset,
        'prediction_targets':int(seen.sum()),'event_exposure':{}}
    for bit,name in enumerate(names):
        hit=(flags&(np.uint64(1)<<np.uint64(bit)))!=0
        out['trace']['event_exposure'][name]={'repeated_world_targets':int(seen[hit].sum()),
          'unique_seen':int((seen[hit]>0).sum()),'unique_available':int(hit.sum()),
          'mean_exposures':float(seen[hit].mean()) if hit.any() else None,
          'expected_head_samples':float(expected_seen[hit].sum())}
    out['head_sampling_note']='Exact conditional expectation from each saved chunk eligible population and logged head-update count, with 256 uniform replacement draws per head update. Actual CUDA RNG not replayed. Repeated world exposures are not independent data.'
    (HERE/'COVERAGE.json').write_text(json.dumps(out,indent=2)+'\n')
    with gzip.open(HERE/'episode_counts.json.gz','wt') as f:json.dump(rows,f,separators=(',',':'))
    for name in ['drink_attempt','wood_attempt','stone_attempt','coal_attempt','iron_attempt','coal_increment','iron_increment','damage','death','risk','risk_damage','risk_death']:
        print(name,out['summary']['events'][name],out['trace']['event_exposure'][name],flush=True)
    print('summary',out['summary']['episodes'],ntr,'trace',len(trace),'monitor',len(split['monitor']),flush=True)


def expected_head_exposure(trace,n_frames):
    expected=np.zeros(n_frames,np.float64);start=0;updates=0
    for line in HEAD_LOG.read_text().splitlines():
        row=json.loads(line);stop=int(row['world_frames']);now=int(row['head_updates'])
        assert stop-start==row['frames'] and now>=updates
        target=trace['target'][start:stop];target=target[target>=0]
        if len(target):np.add.at(expected,target,256*(now-updates)/len(target))
        start=stop;updates=now
    assert start==len(trace) and updates==25000
    assert np.isclose(expected.sum(),256*updates)
    return expected


if __name__=='__main__':main()
