"""Reconstruction of the rejected single-winner selector atop v2's unchanged dominant-motion estimate.
Retained for reproducing SINGLE_PLANE_QUALIFICATION; never used by canonical training."""
import numpy as np
from saccade import data as D

def correspondence(frames):
    c,sources,valid=D.change_correspondence(frames)
    identity_source,global_source=sources[:,:,:,0],sources[:,:,:,1]
    cost0=np.abs(c[1:]-identity_source).mean((-1,-2,-3))
    cost1=np.abs(c[1:]-global_source).mean((-1,-2,-3));gv=valid[:,:,:,1]
    def surround(x,v):
        z=[];x=np.pad(x,((0,0),(1,1),(1,1)));v=np.pad(v,((0,0),(1,1),(1,1)))
        for dr in (-1,0,1):
            for dc in (-1,0,1):
                if dr==dc==0:continue
                a=x[:,1+dr:10+dr,1+dc:10+dc];b=v[:,1+dr:10+dr,1+dc:10+dc]
                z.append(np.where(b,a,np.nan))
        return np.nanmedian(z,axis=0)
    choose=(surround(cost0,np.ones_like(gv))<surround(cost1,gv)-1e-6)|(cost0==0)|~gv
    source=np.where(choose[:,:,:,None,None,None],identity_source,global_source)
    exact=np.abs(c[1:]-source).max((-1,-2,-3))==0
    return c,source,exact
