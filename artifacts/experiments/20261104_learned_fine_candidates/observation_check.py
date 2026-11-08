"""Check reader reliability over independent real rendering noise, keeping the successor world fixed.

Declared after the route result: source/future one-frame readability is not a
bound on recognition across night-static draws. This checks the instrument,
not a model repair or a new source-selection criterion.
"""
import os
os.environ.setdefault('JAX_PLATFORMS','cpu')
import json,sys
from pathlib import Path
import numpy as np
import torch
from saccade import data as D
from craftax.craftax_classic.constants import BlockType
HERE=Path(__file__).resolve().parent

def build():
    import jax,jax.numpy as jnp
    _,_,_,step,frame=D.RP.env_and_render()
    source=torch.load(HERE/'sources.pt',map_location='cpu',weights_only=False)['rows']
    qual=json.loads((HERE/'qualification.json').read_text());ids=[]
    for a in qual:
        if a['meta']['family']!='terrain':continue
        if (not a['meta']['missed'] and a['coarse_ok'] and a['target_ok']) or (a['meta']['missed'] and a['source_missed_verified'] and a['fine_ok'] and a['absent_coarse_ok'] and a['target_ok']):ids.append(a['index'])
    rows=[]
    for i in ids:
        r=source[i]['meta'];st=D.RP.advance_to(r['shard'],r['slot'],r['t']);world=np.array(r['world'])
        absent=st.replace(map=st.map.at[world[0],world[1]].set(BlockType.GRASS.value))
        key=D.RP._slot_keys(r['shard'],r['slot'])[1][r['t']];frames=[]
        for state in [st,absent]:
            ns=step(key,state,jnp.asarray(r['action'],jnp.int32))[1]
            frames.append(np.stack([np.asarray(frame(ns.replace(state_rng=jax.random.PRNGKey(616000+100*i+j)))) for j in range(64)]))
        rows.append(dict(index=i,meta=r,frames=np.stack(frames)))
        print('render qualification',len(rows),'/',len(ids),flush=True)
    torch.save(rows,HERE/'observation_frames.pt')

@torch.no_grad()
def score():
    import importlib
    N=importlib.import_module('artifacts.experiments.20261104_learned_fine_candidates.native')
    m,cfg,z,reader,_=N.load();rows=torch.load(HERE/'observation_frames.pt',map_location='cpu',weights_only=False);out=[]
    source=torch.load(HERE/'sources.pt',map_location='cpu',weights_only=False)['rows']
    for r in rows:
        f=torch.tensor(r['frames'].reshape(-1,63,63,3),device=N.DEV)
        values=[]
        for frames in f.split(64):
            with N.TR.autocast(N.DEV,cfg):p=m.perception(frames,torch.tensor([7,1],device=N.DEV).expand(len(frames),-1))
            values.append(N.event(reader,tuple(t[:,None] for t in p.parts()),r['meta'])['event'])
        event=torch.cat(values).reshape(2,64).float().mean(1).tolist()
        fr=torch.tensor(source[r['index']]['current'],device=N.DEV);fx=N.N.gazes(r['meta']['cell'])[:1].expand(2,-1)
        slot=int((N.M.window(fx[:1])[0]==r['meta']['cell']).nonzero()[0])
        with N.TR.autocast(N.DEV,cfg):
            P=m.perception(fr,fx);mean,_=m.pixels(m.tsp.reconstruct(P.g,P.v,N.M.window(fx)),P.v,N.M.window(fx))
            off=int(torch.where((z['episode']==r['meta']['e'])&(z['t']==r['meta']['t']))[0])
            prior=z['prior'][off].to(N.DEV).float().expand(2,-1)
            tsp,_=m.pixels(m.tsp.predict(prior,P.p,P.v,N.M.window(fx)),P.v,N.M.window(fx))
        target=N.M.cell_pixels(N.M.crop(fr,fx)).float()[:,slot];pred=(mean[:,slot].float()+1)*127.5
        contrast=target[0]-target[1];mask=contrast!=0
        errors=((pred[:,None,mask]-target[None,:,mask])/255).square().mean(-1)
        tp=(tsp[:,slot].float()+1)*127.5;te=((tp[:,None,mask]-target[None,:,mask])/255).square().mean(-1)
        recovered=float(((pred[0]-pred[1])*contrast).sum()/contrast.square().sum())
        out.append(dict(index=r['index'],meta=r['meta'],present_recognition=event[0],absent_recognition=1-event[1],
            fine_alternative_pixel_mse=errors.cpu().tolist(),fine_recovered_pixel_contrast=recovered,
            precrop_tsp_alternative_pixel_mse=te.cpu().tolist(),precrop_tsp_alternative_choice=te.argmin(-1).cpu().tolist(),
            precrop_tsp_recovered_pixel_contrast=float(((tp[0]-tp[1])*contrast).sum()/contrast.square().sum()),
            fine_mae=[float(x) for x in (pred-target).abs().mean(-1)],source_rgb_contrast_rms=float(contrast.square().mean().sqrt())))
    (HERE/'observation_check.json').write_text(json.dumps(out,indent=2)+'\n')
    for k in ['stone','tree']:
        for missed in [False,True]:
            a=[r for r in out if r['meta']['kind']==k and r['meta']['missed']==missed]
            print(k,'source missed',missed,'n',len(a),'present',np.mean([r['present_recognition'] for r in a]),'absent',np.mean([r['absent_recognition'] for r in a]),flush=True)

if __name__=='__main__':
    torch.set_num_threads(2)
    build() if sys.argv[1]=='build' else score()
