"""CPU rebuild of the exact teval TRAIN-seed zombie ridge; compare stored E19 mask to pool tokens."""
import hashlib
import json
import sys
from pathlib import Path

import torch

torch.set_num_threads(4)
sys.path.insert(0,'artifacts/experiments/20260927_levers')
import teval as T

HERE=Path(__file__).parent
cachepath=Path(str(T.CACHE).format('raw'))
assert cachepath.exists(), 'Never regenerate missing encoder data in this verification'
cache=torch.load(cachepath,weights_only=False,mmap=True)
meta,train_roots,train_seeds=T.split()
seeds=meta['seed']
val_roots=torch.isin(seeds,train_seeds[:len(train_seeds)//5])
fit_roots=train_roots&~val_roots
toks=torch.cat([cache['ctx'][:,-1:],cache['fut']],1).float()
vis=torch.cat([meta['root_visible'][:,None],meta['future_visible'][:,0]],1)
f=T.facts_of(vis)
cell=toks.flatten(0,1)[:,T.MAP].flatten(0,1)
rows=lambda m:m[:,None].expand(-1,17).flatten()
cr=lambda m:m[:,None].expand(-1,63).flatten()
sub=torch.randperm(len(cell),generator=torch.Generator().manual_seed(0))[:400_000]
fit=torch.zeros(len(cell),dtype=torch.bool)
val=torch.zeros_like(fit)
fit[sub]=cr(rows(fit_roots))[sub]
val[sub]=cr(rows(val_roots))[sub]
print(json.dumps({'stage':'fit_exact_zombie_ridge','fit_cells':int(fit.sum()),'validation_cells':int(val.sum())}),flush=True)
zombie=T.ridge(cell,f['zombie'].flatten()[:,None],fit,val)
pool=torch.load('artifacts/eda/spatial_pool_v1/pool.pt',weights_only=False,mmap=True)
target=torch.load('artifacts/eda/hpctx_labels_v1.pt',weights_only=False,mmap=True)['beside']
chunks=[]
beside=[22,30,32,40]
for i in range(0,len(target),128):
    x=pool['tokens'][i:i+128,1:,:, :][:,:,beside].float()
    v=zombie(x.flatten(0,2)).view(len(x),5,4).amax(-1)>.3
    chunks.append(v)
pred=torch.cat(chunks)
out={'mask_disagreements':int((pred!=target).sum()),'mask_elements':target.numel(),
     'mask_true':int(target.sum()),'rebuilt_true':int(pred.sum()),'threshold':.3,
     'fit_cells':int(fit.sum()),'validation_cells':int(val.sum()),
     'scope':'Exact regeneration from existing Raw cache/TRAIN-seed probe and pool; no GPU, no new evaluation block.'}
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as s:
        for b in iter(lambda:s.read(8<<20),b''):
            h.update(b)
    return h.hexdigest()
out['sources']={str(p):sha(p) for p in [Path(T.__file__),Path(__file__)]}
out['mask_sha256']=sha('artifacts/eda/hpctx_labels_v1.pt')
out['cache_sha256']=sha(cachepath)
out['meta_sha256']=sha(T.META)
out['pool_sha256']=json.loads((HERE/'inputs.json').read_text())['datasets']['artifacts/eda/spatial_pool_v1/pool.pt']['sha256']
tmp=HERE/'health_mask.tmp'
tmp.write_text(json.dumps(out,indent=2)+'\n')
tmp.replace(HERE/'health_mask.json')
print(json.dumps(out),flush=True)
assert out['mask_disagreements']==0, out
