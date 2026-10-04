"""Global object-count proxies for spatial-adjacency labels on the CLS probe's exact 54k rows."""
import json,random,sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[3]
sys.path[:0]=[str(ROOT),str(ROOT/'artifacts/experiments/20260921_readout_ladder')]
from frozen_ladder import strata

def auc(score,label):
 pos=score[label].float();neg=score[~label].float().sort().values
 lo=torch.searchsorted(neg,pos,right=False).float();hi=torch.searchsorted(neg,pos,right=True).float()
 return float(((lo+hi)/2).mean()/len(neg))
files=list((ROOT/'artifacts/eda/observe_fresh_v5').glob('seed-*.pt'))
random.Random(407).shuffle(files)
rows=[]
for f in files:
 rows.extend(torch.load(f,weights_only=False))
 if len(rows)>=1200:break
v=torch.stack([r['visible'] for r in rows]).float();s=strata(v)
tiles=v[:,:1071].reshape(-1,7,9,17).argmax(-1)
mobs=v[:,1071:1512].reshape(-1,7,9,7)
report={'rows':len(rows),'counts':{}}
for name,count in [('lava',(tiles==14).sum((1,2)).float()),('zombie',mobs[...,0].sum((1,2)).float())]:
 y=s[f'{name}_adjacent'];pres=count>0
 report['counts'][name]={'adjacent':int(y.sum()),'any_visible':int(pres.sum()),
                         'global_count_auc':auc(count,y),'presence_auc':auc(pres.float(),y),
                         'P_adjacent_given_visible':float(y[pres].float().mean())}
Path(__file__).with_suffix('.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report),flush=True)
