"""Exact stored-state renderer repeats, CPU only. No new physical transitions."""
import os
os.environ.setdefault('JAX_PLATFORMS','cpu')
import json
from pathlib import Path
import numpy as np
import jax
from saccade import data as D
from saccade.eval import s1 as S
HERE=Path(__file__).resolve().parent

def main():
    L=S.first_episodes(dict(np.load(S.LABELS/'s0_dev_labels.npz')),12)
    so=np.r_[0,np.cumsum(L['ep_T']+1)];rows=[]
    # Lowest-light state in each fixed episode before step 256, capped at eight scenes.
    # Initial onset-only renders were preserved separately: light .495 gives negligible static.
    # This selection uses renderer conditions only and preceded ANY model scoring of repeats.
    for e,T in enumerate(L['ep_T']):
        ts=np.arange(8,min(int(T),255)+1)
        ts=ts[L['s_light'][so[e]+ts]<.5]
        if len(ts):rows.append((e,int(ts[np.argmin(L['s_light'][so[e]+ts])])) )
    if len(rows)<8:
        # Fill with predeclared evenly spaced reliability scenes, including daylight (a zero-effect control).
        for e,T in enumerate(L['ep_T']):
            for t in np.unique(np.linspace(8,min(int(T),255),4,dtype=int)):
                if (e,int(t)) not in rows:rows.append((e,int(t)))
                if len(rows)>=8:break
            if len(rows)>=8:break
    rows=rows[:8];frames=[];checks=[];eps=S.episodes(L)
    _,_,_,_,render=D.RP.env_and_render()
    for e,t in rows:
        ep=eps[e];state=D.RP.advance_to(ep.shard,ep.slot,t)
        orig=np.asarray(render(state));stored=D.RP.episode_fields(ep.shard,ep.slot)['observations'][t].numpy()
        err=int(np.abs(orig.astype(int)-stored.astype(int)).max());assert err==0,(e,t,err)
        draws=[np.asarray(render(state.replace(state_rng=jax.random.PRNGKey(611460+k)))) for k in range(4)]
        frames.append(np.stack(draws));checks.append({'e':e,'t':t,'original_max_pixel_error':err,'light':float(state.light_level),
            'repeat_pixel_rms':float(np.std(np.stack(draws).astype(float),axis=0).mean())})
        print(checks[-1],flush=True)
    np.savez_compressed(HERE/'static_repeats.npz',rows=np.array(rows),frames=np.stack(frames))
    (HERE/'static_repeats.json').write_text(json.dumps(checks,indent=1))

if __name__=='__main__':main()
