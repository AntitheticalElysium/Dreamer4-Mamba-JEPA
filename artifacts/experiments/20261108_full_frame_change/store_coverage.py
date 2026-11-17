"""Full-training-store verification. Simulator labels audit coverage; they never construct targets."""
import json
import numpy as np
from saccade import data as D,train as TR
from .phase_pairs import HERE

def main():
    cfg=TR.load_cfg();eps=D.load_split(cfg)['train'];directory=D.targets_dir(cfg)
    targets=np.load(directory/'change_targets.npy',mmap_mode='r');D.validate_change_targets(targets,eps,directory)
    with np.load(D.ROOT/'artifacts/eda/s0/s0_train_labels.npz') as z:
        L={k:z[k] for k in ('ep_T','ep_shard','ep_slot','s_body')}
    lo=np.r_[0,np.cumsum(L['ep_T']+1)]
    lookup={(int(sh),int(sl)):i for i,(sh,sl) in enumerate(zip(L['ep_shard'],L['ep_slot']))}
    stats={name:{'transitions':0,'changes':0,'changes_with_nonzero_target':0,'changes_above_half':0,'unchanged_with_nonzero_target':0} for name in ('health','food','drink','energy')}
    off=D.frame_offsets(eps)
    for i,e in enumerate(eps):
        k=lookup[e.shard,e.slot];assert L['ep_T'][k]==e.T
        body=L['s_body'][lo[k]:lo[k+1]].astype(int);changed=np.diff(body,axis=0)!=0
        p=targets[off[i]+1:off[i+1],7,:4,0]
        assert np.isfinite(p).all() and np.isnan(targets[off[i]]).all()
        assert np.isfinite(targets[off[i]+1:off[i+1],3,4,0]).all()
        for j,name in enumerate(stats):
            s=stats[name];y=changed[:,j];z=p[:,j]
            s['transitions']+=len(y);s['changes']+=int(y.sum());s['changes_with_nonzero_target']+=int((y&(z>0)).sum());s['changes_above_half']+=int((y&(z>.5)).sum());s['unchanged_with_nonzero_target']+=int((~y&(z>0)).sum())
    total=[{'finite':0,'above_half':0,'sum':0.} for _ in range(2)]
    for i in range(0,len(targets),8192):
        a=targets[i:i+8192]
        for mode in range(2):
            z=a[:,:,:,mode];s=total[mode];s['finite']+=int(np.isfinite(z).sum());s['above_half']+=int((z>.5).sum());s['sum']+=float(np.nansum(z,dtype=np.float64))
    for s in total:s['positive_fraction_above_half']=s['above_half']/s['finite'];s['mean_soft_target']=s['sum']/s['finite']
    out={'episodes':len(eps),'transitions':int(sum(e.T for e in eps)),'target_shape':list(targets.shape),'body_cues':stats,'modes':dict(zip(D.CHANGE_MODES,total))}
    (HERE/'STORE_COVERAGE.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2),flush=True)

if __name__=='__main__':main()
