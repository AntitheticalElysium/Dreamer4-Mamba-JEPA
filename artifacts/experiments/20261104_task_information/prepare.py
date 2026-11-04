"""Natural raw sensor inputs and visible arrow headings; no world/reader training."""
import os
os.environ.setdefault('JAX_PLATFORMS','cpu')
import json,sys
from pathlib import Path
import numpy as np
import torch
import jax,jax.numpy as jnp
from saccade import data as D,model as M
from saccade.eval import s1 as S
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2]
SHA='34bda77fe5ec5226c4f003301c972d918e2d4a4a554a5a6f279fcabbe922b045'

def main(split):
    torch.set_num_threads(2)
    cached=torch.load(ROOT/f'artifacts/experiments/20261103_fusion_retention/p1f16/{split}.pt',map_location='cpu',mmap=True,weights_only=False)
    assert cached['checkpoint_sha256']==SHA;z=cached['current'];n=len(z['p'])
    L=S.first_episodes(dict(np.load(S.LABELS/('s0_train400_labels.npz' if split=='train' else 's0_dev_labels.npz'))),400)
    env,params,_,_,_=D.RP.env_and_render()
    def scan(reset_key,keys,actions):
        _,s=env.reset(reset_key,params)
        def body(st,ka):
            nxt=env.step(ka[0],st,ka[1],params)[1]
            return nxt,nxt.arrow_directions
        _,dirs=jax.lax.scan(body,s,(keys,actions))
        return jnp.concatenate([s.arrow_directions[None],dirs])
    run=jax.jit(scan)
    coarse=torch.empty(n,81,27,dtype=torch.uint8);fine=torch.empty(n,9,147,dtype=torch.uint8);heading=np.full((n,63),-1,np.int8)
    off=0;so=np.r_[0,np.cumsum(L['ep_T']+1)];census=[]
    for e,T in enumerate(L['ep_T']):
        T=int(T);sh,slot=int(L['ep_shard'][e]),int(L['ep_slot'][e]);raw=D.RP.episode_fields(sh,slot)
        assert torch.equal(z['episode'][off:off+T],torch.full((T,),e))
        assert torch.equal(z['t'][off:off+T],torch.arange(T))
        assert np.array_equal(z['action'][off:off+T].numpy(),raw['actions_taken'].numpy())
        frames=raw['observations'][:T];fix=z['fix'][off:off+T]
        for st in range(0,T,256):
            f=frames[st:st+256];pp=f.reshape(-1,9,7,9,7,3).permute(0,1,3,5,2,4).reshape(-1,3,7,7).float()
            pooled=torch.nn.functional.adaptive_avg_pool2d(pp,3).round().byte().reshape(len(f),81,27)
            coarse[off+st:off+st+len(f)]=pooled
            fine[off+st:off+st+len(f)]=M.cell_pixels(M.crop(f,fix[st:st+len(f)]))
        rk,keys=D.RP._slot_keys(sh,slot);b=next(x for x in D.RP.BUCKETS if x>=T);acts=np.zeros(b,np.int32);acts[:T]=raw['actions_taken'].numpy()
        dirs=np.asarray(run(rk,keys[:b],jnp.asarray(acts)))[:T]
        a=int(so[e]);pos=L['s_arrows_p'][a:a+T]-L['s_pos'][a:a+T,None];vis=L['s_arrows_m'][a:a+T]&(abs(pos[...,0])<=3)&(abs(pos[...,1])<=4)
        for j in range(3):
            tt=np.flatnonzero(vis[:,j]);vec=dirs[tt,j];codes=(vec[:,0]==1).astype(int)+(vec[:,1]==-1)*2+(vec[:,1]==1)*3
            assert np.isin(vec,[-1,0,1]).all() and (abs(vec).sum(-1)==1).all()
            cells=(pos[tt,j,0]+3)*9+pos[tt,j,1]+4;heading[off+tt,cells]=codes
        assert np.array_equal(heading[off:off+T]>=0,z['truth'][off:off+T].numpy()==4)
        census.append(dict(episode=e,frames=T,visible_arrows=int(vis.sum())))
        off+=T
        if (e+1)%40==0:print(split,e+1,'/400',off,'frames',flush=True)
    assert off==n
    torch.save(dict(raw=coarse,fine_pixels=fine,heading=torch.from_numpy(heading),checkpoint_sha256=SHA),HERE/f'{split}_raw.pt')
    (HERE/f'{split}_census.json').write_text(json.dumps(census,indent=2)+'\n')

if __name__=='__main__':main(sys.argv[1])
