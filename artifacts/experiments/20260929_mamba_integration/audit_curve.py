"""Held-out, paired-seed 6/12/18k learning-curve audit from immutable per-root rows."""
import json,hashlib
from pathlib import Path
import numpy as np,torch
ROOT=Path('/home/antithetical/EPITA/PERSO/DynamicHorizons-Mamba-JEPA');BASE=ROOT/'artifacts/experiments/20260927_levers/evals'
meta=torch.load(ROOT/'artifacts/eda/diagnosis_rollouts_v1/meta.pt',weights_only=False);seeds=meta['seed'];uall=seeds.unique();train=uall[torch.randperm(len(uall),generator=torch.Generator().manual_seed(0))[:int(.7*len(uall))]];test=~torch.isin(seeds,train)
ids=seeds[test].numpy();u=np.unique(ids);idx=np.searchsorted(u,ids);B=5000;rng=np.random.default_rng(20260930);draw=rng.integers(0,len(u),size=(B,len(u)));w=np.stack([np.bincount(q,minlength=len(u)) for q in draw]);ci=lambda v:[float(np.quantile(v,.025)),float(np.quantile(v,.975))]
out={'panel_sha256':hashlib.sha256((ROOT/'artifacts/eda/diagnosis_rollouts_v1/meta.pt').read_bytes()).hexdigest(),'test_roots':int(test.sum()),'test_seeds':len(u),'bootstrap_seed':20260930,'bootstrap_replicates':B,'arms':{},'within_arm_budget_contrasts':{},'across_arm_ratios_descriptive':{}}
boots={}
for arm in ('raw','ldad1','ldad10'):
 out['arms'][arm]={};boots[arm]={}
 for budget in (6000,12000,18000):
  name=f'int_corrg_{arm}_suffix_s7_fmamba_u18000'+(f'_at{budget}' if budget<18000 else '')
  path=BASE/f'{name}_per_root.pt';d=torch.load(path,weights_only=False);e=d['onestep_err'][test].double().numpy();c=d['onestep_copy'][test].double().numpy();cl=d['class'][test].numpy();V=d['V'];row={'per_root_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'onestep':{},'rollout':{}};boots[arm][budget]={}
  for j,k in enumerate(('moved','blocked','interact','sleep','idle','all')):
   m=cl==j if j<5 else np.ones_like(cl,dtype=bool);ee=np.bincount(idx,weights=(e*m).sum(1),minlength=len(u));cc=np.bincount(idx,weights=(c*m).sum(1),minlength=len(u));r=ee.sum()/cc.sum();br=(w@ee)/(w@cc);boots[arm][budget][k]=br
   row['onestep'][k]={'n':int(m.sum()),'error_over_copy':float(r),'ci95_seed':ci(br),'error_over_V':float(e[m].mean()/V),'copy_over_V':float(c[m].mean()/V)}
  alive=d['alive'][test].numpy();g=d['gen_err'][test].double().numpy();tf=d['tf_err'][test].double().numpy()
  for depth in (1,4,8,16):
   j=depth-1;m=alive[:,j];gg=np.bincount(idx,weights=g[:,j]*m,minlength=len(u));tt=np.bincount(idx,weights=tf[:,j]*m,minlength=len(u));nn=np.bincount(idx,weights=m,minlength=len(u));bg=(w@gg)/(w@nn)/V;bt=(w@tt)/(w@nn)/V
   row['rollout'][str(depth)]={'n':int(m.sum()),'generated_over_V':float(gg.sum()/nn.sum()/V),'generated_ci95_seed':ci(bg),'teacher_over_V':float(tt.sum()/nn.sum()/V),'recursive_minus_teacher_over_V':float((gg.sum()-tt.sum())/nn.sum()/V),'recursive_minus_teacher_ci95_seed':ci(bg-bt)}
  out['arms'][arm][str(budget)]=row
 for lo,hi in ((6000,12000),(12000,18000),(6000,18000)):
  key=f'{arm}_{hi}_minus_{lo}';out['within_arm_budget_contrasts'][key]={k:{'estimate':out['arms'][arm][str(hi)]['onestep'][k]['error_over_copy']-out['arms'][arm][str(lo)]['onestep'][k]['error_over_copy'],'ci95_seed':ci(boots[arm][hi][k]-boots[arm][lo][k])}for k in ('moved','blocked','all')}
for budget in (6000,12000,18000):
 for a,b in (('ldad1','raw'),('ldad10','raw'),('ldad1','ldad10')):
  key=f'{a}_minus_{b}_{budget}';out['across_arm_ratios_descriptive'][key]={k:{'estimate':out['arms'][a][str(budget)]['onestep'][k]['error_over_copy']-out['arms'][b][str(budget)]['onestep'][k]['error_over_copy'],'ci95_seed':ci(boots[a][budget][k]-boots[b][budget][k])}for k in ('moved','blocked','all')}
p=ROOT/'artifacts/experiments/20260929_mamba_integration/audit_curve.json';p.write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps({'test_roots':out['test_roots'],'test_seeds':out['test_seeds'],'arms':{a:{k:{'one':{c:round(v['error_over_copy'],3) for c,v in x['onestep'].items()},'gen16':round(x['rollout']['16']['generated_over_V'],3),'tf16':round(x['rollout']['16']['teacher_over_V'],3)}for k,x in d.items()}for a,d in out['arms'].items()},'within_arm_budget_contrasts':out['within_arm_budget_contrasts']},indent=2))
