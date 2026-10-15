"""R50: DV16 paired differences between worlds (secondary, not preregistered; 2026-10-16; CPU, saved dv16 rows only)

Per root, expected safe at 16 of the argmin-risk action (uniform over ties), for each world's saved DV16 risks (HD / HL / ZA), with
all 17 first actions and with SLEEP excluded; paired differences PI - D parent and IC - D parent, bootstrap 95% intervals over roots
(4,000 resamples). The D parent is the regenerating Mamba belief on the same spatial scaffold (no copies): the "ordinary evolving
latent belief" of the user's sharpened question.
Part 2 (R51): the escape-direction sub-choice (noop + 4 moves): within-root pairwise AUC of each world's imagined risk against the
real P16 ordering (ties 0.5; roots with no real spread contribute nothing), per root, and paired bootstrap of AUC differences vs D
(PI sampled, PI's deterministic afterstate pi_ref, and pi_hudgen: pi_ref with the 18 HUD cells always regenerated).
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

P5 = P[:, :5]
def auc_root(risk):
    out = []
    for i in range(len(risk)):
        num = den = 0.
        for a in range(5):
            for b in range(5):
                if P5[i, a] > P5[i, b]:
                    den += 1; num += float(risk[i, a] > risk[i, b]) + .5 * float(risk[i, a] == risk[i, b])
        out.append(num / den if den else float('nan'))
    return torch.tensor(out)
for key in ('HL', 'HD'):
    A = {w: auc_root(torch.stack([r[key] for r in torch.load(OUT / f'dv16_{w}.pt', weights_only=False)])) for w in ('pi_s1', 'pi_s2', 'pi_ref_s1', 'pi_ref_s2', 'pi_hudgen_s1', 'pi_hudgen_s2', 'ic_s1', 'ic_s2', 'D_parent_ref')}
    ok = ~A['D_parent_ref'].isnan()
    for w in ('pi_s1', 'pi_s2', 'pi_ref_s1', 'pi_ref_s2', 'pi_hudgen_s1', 'pi_hudgen_s2', 'ic_s1', 'ic_s2'):
        print(key, 'AUC5', w, round(float(A[w][ok].mean()), 4), '- D', boot(A[w][ok] - A['D_parent_ref'][ok]), 'roots', int(ok.sum()))
