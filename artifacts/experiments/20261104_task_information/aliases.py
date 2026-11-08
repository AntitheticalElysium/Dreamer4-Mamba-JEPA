"""Search a stronger structural witness: same real coarse sensor, different fine cue.

Only natural newly revealed ore cells; alternatives change ore identity, preserve
solidity, and use the unchanged renderer/noise key. No pixels are edited.
"""
import os
os.environ.setdefault('JAX_PLATFORMS','cpu')
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import jax,jax.numpy as jnp
from saccade import data as D
from saccade.eval import s1 as S
HERE=Path(__file__).resolve().parent

def main():
    torch.set_num_threads(2);L=S.first_episodes(dict(np.load(S.LABELS/'s0_dev_labels.npz')),250);so=np.r_[0,np.cumsum(L['ep_T']+1)];sources=[]
    offsets=np.array([[r,c] for r in range(-3,4) for c in range(-4,5)])
    for e,T in enumerate(L['ep_T']):
        seen=set()
        for t in range(int(T)):
            s=int(so[e]+t);worlds=L['s_pos'][s]+offsets;new=np.array([tuple(w) not in seen for w in worlds]);seen.update(map(tuple,worlds))
            if t<8 or L['s_light'][s]>=.08 or L['s_sleep'][s] or L['s_body'][s,0]<=0 or L['s_inv'][s,6]<=0:continue
            view=L['s_view'][s].reshape(-1);ok=new&np.isin(view,[4,8,9,10])
            occupied=set()
            for kind in ['zombies','cows','skeletons','arrows']:
                for p in L[f's_{kind}_p'][s][L[f's_{kind}_m'][s]]:occupied.add(tuple(p))
            for c in np.flatnonzero(ok):
                if tuple(worlds[c]) in occupied:continue
                sources.append(dict(episode=e,time=t,cell=int(c),world=worlds[c].tolist(),kind=int(view[c]),light=float(L['s_light'][s]),shard=int(L['ep_shard'][e]),slot=int(L['ep_slot'][e])))
    sources=sources[:1000];(HERE/'alias_sources.json').write_text(json.dumps(sources,indent=2)+'\n');print('source-only ore cells',len(sources),flush=True)
    _,_,_,_,frame=D.RP.env_and_render();types=jnp.array([4,8,9,10],jnp.int32)
    render=jax.jit(lambda st,w:jax.vmap(lambda ty:frame(st.replace(map=st.map.at[w[0],w[1]].set(ty))))(types))
    out=[];last=None;st=None
    for i,r in enumerate(sources):
        key=(r['episode'],r['time'])
        if key!=last:st=D.RP.advance_to(r['shard'],r['slot'],r['time']);last=key
        images=np.asarray(render(st,jnp.array(r['world'])));actual=[4,8,9,10].index(r['kind'])
        raw=D.RP.episode_fields(r['shard'],r['slot'])['observations'][r['time']].numpy();assert np.array_equal(images[actual],raw)
        cells=torch.tensor(images).reshape(-1,9,7,9,7,3).permute(0,1,3,5,2,4).reshape(-1,3,7,7).float()
        coarse=F.adaptive_avg_pool2d(cells,3).round().byte().reshape(4,81,27)
        for alt in range(4):
            if alt!=actual and torch.equal(coarse[actual],coarse[alt]) and not np.array_equal(images[actual],images[alt]):
                out.append(dict(meta=dict(**r,alternative=[4,8,9,10][alt]),current=images[[actual,alt]]));print('RAW COARSE ALIAS',len(out),r,'alt',[4,8,9,10][alt],flush=True)
        if (i+1)%100==0:print('alias search',i+1,'/',len(sources),flush=True)
    torch.save(out,HERE/'aliases.pt');print('raw sensor aliases found',len(out),flush=True)

if __name__=='__main__':main()
