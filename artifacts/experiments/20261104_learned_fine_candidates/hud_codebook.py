"""Exact renderer atlas + frozen native periphery inversion, no learned reader."""
import os
os.environ.setdefault('JAX_PLATFORMS','cpu')
import json,importlib,sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from saccade import data as D,model as M,train as TR
from saccade.eval import s1 as S
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2]
FIELDS=[('health',63,'s_body',0),('food',64,'s_body',1),('drink',65,'s_body',2),('energy',66,'s_body',3),
 ('sapling',67,'s_inv',5),('wood',68,'s_inv',0),('stone',69,'s_inv',1),('coal',70,'s_inv',2),('iron',71,'s_inv',3),
 ('diamond',72,'s_inv',4),('wood_pickaxe',73,'s_inv',6),('stone_pickaxe',74,'s_inv',7),('iron_pickaxe',75,'s_inv',8),
 ('wood_sword',76,'s_inv',9),('stone_sword',77,'s_inv',10),('iron_sword',78,'s_inv',11)]

def build():
    import jax,jax.numpy as jnp
    _,_,reset,_,frame=D.RP.env_and_render();_,st=reset(jax.random.PRNGKey(615001));atlas=[]
    for name,cell,arr,col in FIELDS:
        patches=[];r,c=divmod(cell,9)
        for v in range(10):
            s=st.replace(**{'player_'+name:jnp.asarray(v)}) if arr=='s_body' else st.replace(inventory=st.inventory.replace(**{name:jnp.asarray(v)}))
            f=np.asarray(frame(s));patches.append(f[r*7:(r+1)*7,c*7:(c+1)*7])
        atlas.append(np.stack(patches))
    np.save(HERE/'hud_atlas.npy',np.stack(atlas));print('atlas built',flush=True)

@torch.no_grad()
def check():
    N=importlib.import_module('artifacts.experiments.20261104_fine_causal_design.test_native');m,cfg=N.load();dev=N.DEV
    torch.set_num_threads(2);L=dict(np.load(S.LABELS/'s0_dev_labels.npz'));so=np.r_[0,np.cumsum(L['ep_T']+1)]
    z=torch.load(ROOT/'artifacts/experiments/20261103_fusion_retention/p1f16/dev.pt',map_location='cpu',weights_only=False,mmap=True)['current']
    si=so[z['episode'].numpy()]+z['t'].numpy();truth=np.stack([L[arr][si,col] for name,cell,arr,col in FIELDS],1).clip(0,9).astype(int)
    ids=set(np.random.default_rng(615003).choice(len(si),4096,replace=False).tolist())
    for c in range(16):
        for v in np.unique(truth[:,c]):ids.update(np.flatnonzero(truth[:,c]==v)[:4].tolist())
    ids=np.array(sorted(ids));atlas=np.load(HERE/'hud_atlas.npy');cells=torch.tensor([f[1] for f in FIELDS])
    # Validate the renderer atlas on natural corpus pixels, independent of latent matching.
    exact=np.zeros((len(ids),16),bool)
    for e in np.unique(z['episode'][ids].numpy()):
        rows=np.flatnonzero(z['episode'][ids].numpy()==e);ix=ids[rows]
        shard,slot=z['episode_keys'][int(e)];fr=D.RP.episode_fields(int(shard),int(slot))['observations'][z['t'][ix]].numpy()
        patches=fr.reshape(-1,9,7,9,7,3).transpose(0,1,3,2,4,5).reshape(-1,81,7,7,3)[:,cells.numpy()]
        exact[rows]=(patches==atlas[np.arange(16)[None],truth[ix]]).all((2,3,4))
    pix=torch.tensor(atlas,device=dev).permute(0,1,4,2,3).reshape(-1,3,7,7).float()
    pooled=F.adaptive_avg_pool2d(pix,3).round().reshape(16,10,27)/127.5-1
    correct=[];gap=[]
    for a in range(0,len(ids),128):
        ix=ids[a:a+128];v=z['v'][ix].to(dev).float();h=pooled[None].expand(len(ix),-1,-1,-1)
        with TR.autocast(dev,cfg):
            for layer,film in zip(m.perception.p_layers,m.perception.film):
                gain,shift=film(v)[:,None,None].chunk(2,-1);h=F.gelu(layer(h))*(1+gain)+shift
            codes=M.out_map(m.perception.p_out(h),m.perception.s)
        # Match the saved native token, not a token computed from its truth label.
        p=z['p'][ix][:,cells].to(dev).float();dist=(codes-p[:,:,None]).square().mean(-1)
        yh=dist.argmin(-1).cpu().numpy();correct.append(yh==truth[ix]);d=dist.sort(-1).values;gap.append((d[...,1]-d[...,0]).cpu().numpy())
    correct=np.concatenate(correct);gap=np.concatenate(gap)
    out=dict(frames=len(ids),rows=len(ids)*16,method='Enumerated native renderer glyphs -> frozen periphery given actual lighting -> nearest exact token code. No fitted weights.',
        fields={name:dict(pixel_atlas_matches=int(exact[:,j].sum()),latent_correct=int(correct[:,j].sum()),rows=len(ids),
            min_runnerup_squared_distance_gap=float(gap[:,j].min()),values=sorted(np.unique(truth[ids,j]).tolist())) for j,(name,*_) in enumerate(FIELDS)})
    (HERE/'hud_codebook.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2),flush=True)

if __name__=='__main__':build() if sys.argv[1]=='build' else check()
