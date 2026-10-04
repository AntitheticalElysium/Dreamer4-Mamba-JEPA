"""Exploratory factual scroll-estimator accuracy near visible zombies on inspected one-step forks."""
import hashlib,json,sys
from pathlib import Path
import torch
ROOT=Path('/home/antithetical/EPITA/PERSO/DynamicHorizons-Mamba-JEPA')
HERE=Path(__file__).resolve().parent
sys.path[:0]=[str(ROOT),str(ROOT/'artifacts/experiments/20260927_levers'),
              str(ROOT/'artifacts/experiments/20260926_diagnosis')]
import teval as E
from onestep import classify
from scroll import estimate,MOVE_SHIFT

def main():
    torch.set_num_threads(8)
    meta,_,_=E.split();cls,_=classify(meta)
    cache=torch.load(Path(str(E.CACHE).format('raw')),map_location='cpu',weights_only=False,mmap=True)
    z=meta['root_visible'][:,1071:1512].reshape(-1,7,9,7)[...,0]>0
    near=z[:,2:5,3:6].flatten(1).any(1)
    root=cache['ctx'][:,-1].float()
    out={'status':'complete','strata':{},'raw_cache_sha256':hashlib.sha256(Path(str(E.CACHE).format('raw')).read_bytes()).hexdigest(),
         'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    pred=torch.empty(len(root),4,dtype=torch.long)
    for start in range(0,len(root),32):
        sl=slice(start,min(len(root),start+32))
        pred[sl]=estimate(root[sl,None],cache['one'][sl,1:5].float())
    truth=torch.zeros_like(pred)
    for a,sh in MOVE_SHIFT.items():truth[:,a-1]=torch.where(cls[:,a]==0,sh,0)
    for name,m in [('zombie_near',near),('no_zombie_near',~near),('all',torch.ones_like(near))]:
        y=cls[m,1:5];p=pred[m];t=truth[m]
        valid=(y==0)|(y==1)
        row={'roots':int(m.sum()),'move_attempts':int(valid.sum()),'agreement':float((p[valid]==t[valid]).float().mean())}
        for group,c in [('moved',0),('blocked',1)]:
            mask=y==c
            row[group+'_n']=int(mask.sum());row[group+'_agreement']=float((p[mask]==t[mask]).float().mean())
            row[group+'_miss_or_false']=float(((p[mask]==0) if c==0 else (p[mask]!=0)).float().mean())
        out['strata'][name]=row
    HERE.joinpath('scroll_hazard.json').write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps(out,indent=2),flush=True)
if __name__=='__main__':main()
