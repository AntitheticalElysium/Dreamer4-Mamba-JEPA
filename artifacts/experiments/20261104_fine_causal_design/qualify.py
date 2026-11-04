"""Source-only qualification of visible, consequential arrow cues. No model selection."""
import os
os.environ.setdefault('JAX_PLATFORMS', 'cpu')
os.environ.setdefault('OMP_NUM_THREADS', '2')
from pathlib import Path
import json, time
import numpy as np
import torch
import torch.nn.functional as F
import jax.numpy as jnp
from saccade import data as D

HERE = Path(__file__).resolve().parent

def pooled(x):
    x=torch.as_tensor(np.array(x)).reshape(-1,9,7,9,7,3).permute(0,1,3,5,2,4).reshape(-1,3,7,7).float()
    return F.adaptive_avg_pool2d(x,3).round().reshape(-1,81,27)

def main():
    torch.set_num_threads(2)
    names=['ep_T','ep_shard','ep_slot','s_arrows_p','s_arrows_m','s_pos','s_light','s_body','s_sleep','t_action']
    f=np.load('artifacts/eda/s0/s0_dev_labels.npz'); L={k:f[k] for k in names}
    so=np.r_[0,np.cumsum(L['ep_T']+1)]; ao=np.r_[0,np.cumsum(L['ep_T'])]
    rows=[]
    for e,T in enumerate(L['ep_T'][:400]):
        sl=slice(so[e],so[e+1]); pos=L['s_pos'][sl]; p=L['s_arrows_p'][sl]; mask=L['s_arrows_m'][sl]
        d=p-pos[:,None]; vis=mask&(abs(d[...,0])<=3)&(abs(d[...,1])<=4)
        first=vis&~np.r_[np.zeros((1,mask.shape[1]),bool),vis[:-1]]
        for t in range(8,int(T)):
            if L['s_sleep'][so[e]+t]: continue
            for j in np.flatnonzero(first[t]&(abs(d[t]).sum(-1)==1)):
                rows.append(dict(e=int(e),t=t,j=int(j),light=float(L['s_light'][so[e]+t]),
                    shard=int(L['ep_shard'][e]),slot=int(L['ep_slot'][e]),action=int(L['t_action'][ao[e]+t]),
                    delta=d[t,j].tolist(),dh=float(L['s_body'][so[e]+t+1,0]-L['s_body'][so[e]+t,0])))
    # Fixed bounded catalogue: 16 dark and 8 daylight cases, in corpus order.
    chosen=[r for r in rows if r['light']<.25][:16]+[r for r in rows if r['light']>=.5][:8]
    (HERE/'catalogue.json').write_text(json.dumps(dict(total_first_adjacent=len(rows),
        dark=sum(r['light']<.25 for r in rows),chosen=chosen),indent=2)+'\n')
    env,params,_,step,frame=D.RP.env_and_render(); out=[]; arrays=[]
    for n,r in enumerate(chosen):
        st=D.RP.advance_to(r['shard'],r['slot'],r['t']); stored=D.RP.episode_fields(r['shard'],r['slot'])['observations'][r['t']].numpy()
        actual=np.asarray(frame(st)); assert np.array_equal(actual,stored)
        toward=-np.asarray(r['delta']); away=-toward
        variants=[]; nxt=[]; stats=[]
        # The two worlds differ ONLY in the visible arrow heading. They are valid
        # rendered states, not claimed to have an identical feasible past.
        key=D.RP._slot_keys(r['shard'],r['slot'])[1][r['t']]
        for direction in [toward,away]:
            s=st.replace(arrow_directions=st.arrow_directions.at[r['j']].set(jnp.asarray(direction)))
            variants.append(np.asarray(frame(s)))
            _,ns,reward,term,_=step(key,s,jnp.asarray(0,jnp.int32))
            nxt.append(np.asarray(frame(ns)))
            stats.append(dict(health=float(ns.player_health),reward=float(reward),terminated=bool(term)))
        pp=pooled(np.stack(variants)); diff=pp[0]-pp[1]
        cf=dict(**r,render_exact=True,original_direction=np.asarray(st.arrow_directions[r['j']]).tolist(),
            coarse_equal=bool((diff==0).all()),coarse_changed=int((diff!=0).sum()),coarse_max=float(diff.abs().max()),
            raw_changed=int((variants[0]!=variants[1]).sum()),next=stats,
            health_contrast=stats[1]['health']-stats[0]['health'])
        out.append(cf);arrays.append(dict(current=np.stack(variants),future=np.stack(nxt)))
        print(n,cf,flush=True)
    torch.save(dict(rows=out,arrays=arrays),HERE/'source_pairs.pt')
    (HERE/'qualification.json').write_text(json.dumps(dict(rows=out,non_claim='Counterfactual headings are valid rendered states; a shared feasible past and fine-specificity are not assumed.'),indent=2)+'\n')

if __name__=='__main__': main()
