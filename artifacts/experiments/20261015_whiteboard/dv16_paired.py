"""R50: DV16 paired differences between worlds (secondary, not preregistered; 2026-10-16; CPU, saved dv16 rows only)

Per root, expected safe at 16 of the argmin-risk action (uniform over ties), for each world's saved DV16 risks (HD / HL / ZA), with
all 17 first actions and with SLEEP excluded; paired differences PI - D parent and IC - D parent, bootstrap 95% intervals over roots
(4,000 resamples). The D parent is the regenerating Mamba belief on the same spatial scaffold (no copies): the "ordinary evolving
latent belief" of the user's sharpened question.
Usage: dv16_paired.py -> prints
"""
import sys
import torch
sys.path.insert(0,'../20261012_oracle'); sys.path.insert(0,'.'); sys.path.insert(0,'../20260927_levers')
import dv16 as DV
roots,_=DV.load_roots()
P=torch.stack([r['P'] for r in roots])[...,2]
OUT=DV.OUT
def per_root(rows,key,excl=None):
    v=[]
    for i,row in enumerate(rows):
        risk=row[key].clone(); p=P[i].clone()
        if excl is not None:
            keep=torch.ones(17,dtype=torch.bool); keep[excl]=False; risk=risk[keep]; p=p[keep]
        m=risk==risk.min(); v.append(1-float(p[m].mean()))
    return torch.tensor(v)
g=torch.Generator().manual_seed(1)
def boot(d):
    idx=torch.randint(len(d),(4000,len(d)),generator=g); b=d[idx].mean(1)
    return round(float(d.mean()),4), round(float(b.quantile(.025)),4), round(float(b.quantile(.975)),4)
D=torch.load(OUT/'dv16_D_parent_ref.pt',weights_only=False)
for w in ('pi_s1','pi_s2','ic_s1','ic_s2'):
    R=torch.load(OUT/f'dv16_{w}.pt',weights_only=False)
    for key in ('HD','HL','ZA'):
        for excl,lab in ((None,'all'),(6,'no-sleep')):
            print(w,'-D',key,lab,boot(per_root(R,key,excl)-per_root(D,key,excl)))
