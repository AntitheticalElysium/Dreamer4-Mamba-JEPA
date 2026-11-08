"""Source-only query metadata. Labels here are never model inputs."""
import json
from pathlib import Path
import numpy as np
import torch
from saccade.eval import s1 as S
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2]

def main():
    torch.set_num_threads(2)
    z=torch.load(ROOT/'artifacts/experiments/20261103_fusion_retention/p1f16/dev.pt',map_location='cpu',mmap=True,weights_only=False)['current']
    raw=torch.load(HERE/'dev_raw.pt',map_location='cpu',mmap=True,weights_only=False)
    q=torch.where((z['localg']>=0).flatten())[0].numpy();f=q//63;c=q%63
    L=dict(np.load(S.LABELS/'s0_dev_labels.npz'));so=np.r_[0,np.cumsum(L['ep_T']+1)];e=z['episode'][f].numpy();t=z['t'][f].numpy();s=so[e]+t;y=z['truth'][f,c].numpy()
    j=np.full(len(q),-1,np.int8);seen=np.full(len(q),-1,np.int16);hp=np.zeros(len(q),np.int8);cd=np.zeros(len(q),np.int16)
    for kind,ky in [('zombies',1),('cows',2),('skeletons',3),('arrows',4)]:
        for ii in np.flatnonzero(y==ky):
            world=L['s_pos'][s[ii]]+np.array(divmod(int(c[ii]),9))-np.array([3,4]);mm=L[f's_{kind}_m'][s[ii]];pos=L[f's_{kind}_p'][s[ii]]
            ids=np.flatnonzero(mm&(pos==world).all(-1));assert len(ids)>0;k=int(ids[-1]);j[ii]=k
            start=max(int(so[e[ii]]),int(s[ii]-8));rel=L[f's_{kind}_p'][start:s[ii],k]-L['s_pos'][start:s[ii]]
            seen[ii]=int((L[f's_{kind}_m'][start:s[ii],k]&(abs(rel[:,0])<=3)&(abs(rel[:,1])<=4)).sum())
            hp[ii]=L[f's_{kind}_hp'][s[ii],k];cd[ii]=L[f's_{kind}_cd'][s[ii],k]
    out=dict(query=q,frame=f,cell=c,episode=e,time=t,kind=y,heading=raw['heading'][f,c].numpy(),j=j,previous8_visible=seen,
        hp=hp,cd=cd,light=L['s_light'][s],shard=L['ep_shard'][e],slot=L['ep_slot'][e],health=L['s_body'][s,0],sleep=L['s_sleep'][s])
    np.savez_compressed(HERE/'query_meta.npz',**out)
    stats={str(k):dict(queries=int((y==k).sum()),night=int(((y==k)&(out['light']<.5)).sum()),new_night=int(((y==k)&(out['light']<.5)&(seen==0)).sum())) for k in range(5)}
    (HERE/'source_counts.json').write_text(json.dumps(stats,indent=2)+'\n');print(json.dumps(stats,indent=2))

if __name__=='__main__':main()
