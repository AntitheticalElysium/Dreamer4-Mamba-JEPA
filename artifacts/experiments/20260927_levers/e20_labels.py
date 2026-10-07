"""Terminal-aware factual health labels, independently checked against HUD pixels.

reward = achievement_count + .1*dh is ambiguous at fractional .1: +1 with one
achievement count and -9 with the next. A genuine death cannot increase health.
Only that terminal ambiguity is corrected. Exact renderer templates independently
read all ten displayed health values; no learned/fork readout supplies these labels.
"""
import os
os.environ.setdefault('JAX_PLATFORMS','cpu')

import numpy as np
import torch


def health_change(rewards, terminated):
    from exposure import health_change as legacy
    dh=legacy(rewards)
    ambiguous=torch.as_tensor(terminated).bool()&(dh==1)
    dh=dh.clone();dh[ambiguous]=-9
    return dh,ambiguous


def templates():
    from craftax.craftax_classic.constants import load_all_textures_given_size
    tex=load_all_textures_given_size(7)
    pictures=[]
    for hp in range(10):
        p=np.zeros((7,7,3),np.float32)
        if hp>0:p[:5,:5]=np.asarray(tex['health_texture'])
        alpha=np.asarray(tex['number_textures_alpha'][hp])
        p[2:6,2:6]=p[2:6,2:6]*(1-alpha)+np.asarray(tex['number_textures'][hp])
        pictures.append(p.astype(np.uint8))
    return np.stack(pictures)


def decode(frames, pictures):
    raw=np.ascontiguousarray(np.asarray(frames)[:,49:56,:7,:])
    # Choose the smallest byte signature distinguishing all ten fixed templates.
    flat=pictures.reshape(10,-1);selected=[];groups=np.zeros(10,dtype=np.int64)
    while len(np.unique(groups))<10:
        scored=[]
        for j in range(flat.shape[1]):
            if j not in selected:
                scored.append((len(np.unique(np.column_stack([groups,flat[:,j]]),axis=0)),j))
        _,j=max(scored);selected.append(j)
        _,groups=np.unique(flat[:,selected],axis=0,return_inverse=True)
    signature=flat[:,selected]
    values=raw.reshape(len(raw),-1)[:,selected]
    match=(values[:,None,:]==signature[None]).all(-1)
    if not bool((match.sum(1)==1).all()):
        raise RuntimeError('HUD glyph signature missing/ambiguous')
    hp=match.argmax(1)
    wrong=np.where((raw!=pictures[hp]).reshape(len(raw),-1).any(1))[0]
    if len(wrong):
        raise RuntimeError(f'HUD pixel template mismatch at frames {wrong[:8].tolist()}')
    return torch.from_numpy(hp.copy()).long()
