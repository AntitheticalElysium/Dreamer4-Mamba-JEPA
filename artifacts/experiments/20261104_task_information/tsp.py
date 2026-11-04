"""Export the existing native pre-crop prediction; no current fine input."""
import os
os.environ.setdefault('JAX_PLATFORMS','cpu');os.environ.setdefault('MAMBA_DETERMINISTIC','1')
import importlib,sys
from pathlib import Path
import torch
N=importlib.import_module('artifacts.experiments.20261104_fine_causal_design.test_native')
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2]

@torch.no_grad()
def main():
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=True
    m,cfg=N.load()
    for split in ['train','dev']:
        a=torch.load(ROOT/f'artifacts/experiments/20261103_fusion_retention/p1f16/{split}.pt',map_location='cpu',mmap=True,weights_only=False)
        assert a['checkpoint_sha256']==N.SHA;z=a['current'];out=[]
        for st in range(0,len(z['p']),1024):
            p,v,b,fix=[z[k][st:st+1024].to(N.DEV).float() for k in ['p','v','prior','fix']];fix=fix.long()
            with N.TR.autocast(N.DEV,cfg):
                h=m.tsp.predict(b,p,v,N.M.window(fix));mean,_=m.pixels(h,v,N.M.window(fix))
            out.append(mean.half().cpu())
        torch.save(dict(mean=torch.cat(out),checkpoint_sha256=N.SHA),HERE/f'{split}_tsp.pt')
        print(split,'native pre-glimpse mean exported',len(z['p']),flush=True)

if __name__=='__main__':main()
