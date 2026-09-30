"""Seed-clustered, held-out rescore of immutable 6k per-root rows.

Distinct from teval's published all-root one-step and rollout aggregates. It
uses only the fixed 43 held-out diagnostic seeds and no model refitting.
"""
import json,hashlib
from pathlib import Path
import numpy as np,torch
ROOT=Path('/home/antithetical/EPITA/PERSO/DynamicHorizons-Mamba-JEPA')
BASE=ROOT/'artifacts/experiments/20260927_levers/evals'
meta=torch.load(ROOT/'artifacts/eda/diagnosis_rollouts_v1/meta.pt',weights_only=False)
seeds=meta['seed']; uniq=seeds.unique();train=uniq[torch.randperm(len(uniq),generator=torch.Generator().manual_seed(0))[:int(.7*len(uniq))]]
test=~torch.isin(seeds,train);ids=seeds[test].numpy();u=np.unique(ids);rs=np.random.default_rng(20260930)
# Shared seed draws for paired arm contrasts; duplicated seeds carry all roots/actions.
B=4000; draw=rs.integers(0,len(u),size=(B,len(u)));w=np.zeros((B,len(u)),dtype=np.int16)
for i in range(B):w[i]=np.bincount(draw[i],minlength=len(u))
ci=lambda x:[float(np.quantile(x,.025)),float(np.quantile(x,.975))]
res={'panel_sha256':hashlib.sha256((ROOT/'artifacts/eda/diagnosis_rollouts_v1/meta.pt').read_bytes()).hexdigest(),'test_roots':int(test.sum()),'test_seeds':len(u),'bootstrap_seed':20260930,'bootstrap_replicates':B,'arms':{}}
ratios={}
for pool in ('raw','ldad1','ldad10'):
 name=f'int_corrg_{pool}_suffix_s7_fmamba_u18000_at6000';p=BASE/f'{name}_per_root.pt';d=torch.load(p,weights_only=False)
 e=d['onestep_err'][test].double().numpy();c=d['onestep_copy'][test].double().numpy();cl=d['class'][test].numpy();seed_ix=np.searchsorted(u,ids);V=d['V']
 arm={'per_root_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'onestep':{},'rollout':{}};ratios[pool]={}
 for j,k in enumerate(('moved','blocked','interact','sleep','idle','all')):
  mask=(cl==j) if j<5 else np.ones_like(cl,dtype=bool)
  ee=np.bincount(seed_ix,weights=(e*mask).sum(1),minlength=len(u));cc=np.bincount(seed_ix,weights=(c*mask).sum(1),minlength=len(u))
  r=ee.sum()/cc.sum();br=(w@ee)/(w@cc);ratios[pool][k]=br
  arm['onestep'][k]={'n':int(mask.sum()),'err_over_copy':float(r),'ci95_seed':ci(br),'error_over_V':float(e[mask].mean()/V),'copy_over_V':float(c[mask].mean()/V)}
 alive=d['alive'][test].numpy();g=d['gen_err'][test].double().numpy();t=d['tf_err'][test].double().numpy();
 for depth in (1,2,4,8,16):
  j=depth-1;m=alive[:,j];gg=np.bincount(seed_ix,weights=g[:,j]*m,minlength=len(u));tt=np.bincount(seed_ix,weights=t[:,j]*m,minlength=len(u));nn=np.bincount(seed_ix,weights=m,minlength=len(u))
  bg=(w@gg)/(w@nn)/V;bt=(w@tt)/(w@nn)/V
  arm['rollout'][str(depth)]={'n':int(m.sum()),'generated_over_V':float(gg.sum()/nn.sum()/V),'generated_ci95_seed':ci(bg),'teacher_over_V':float(tt.sum()/nn.sum()/V),'recursive_minus_teacher_over_V':float((gg.sum()-tt.sum())/nn.sum()/V),'recursive_minus_teacher_ci95_seed':ci(bg-bt)}
 res['arms'][pool]=arm
res['contrasts']={}
for a,b in (('ldad1','raw'),('ldad10','raw'),('ldad1','ldad10')):
 res['contrasts'][f'{a}_minus_{b}']={k:{'estimate':res['arms'][a]['onestep'][k]['err_over_copy']-res['arms'][b]['onestep'][k]['err_over_copy'],'ci95_seed':ci(ratios[a][k]-ratios[b][k])}for k in ('moved','blocked','all')}
p=ROOT/'artifacts/experiments/20260929_mamba_integration/audit6k.json';p.write_text(json.dumps(res,indent=2)+'\n')
print(json.dumps({'arms':{a:{'one':{k:v for k,v in d['onestep'].items() if k in ('moved','blocked','all')},'depth16':d['rollout']['16']}for a,d in res['arms'].items()},'contrasts':res['contrasts']},indent=2))
