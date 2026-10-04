"""Exploratory registration of the actual first temporal-Mamba input, not only encoder patches.

Frozen prior matched corrt/fmamba Raw world, factual diagnosis panel already inspected.
The first factored layer's h=n2(x+space(n1(x))) is measured at fixed vs terrain-aligned slots.
"""
import hashlib,json,sys
from pathlib import Path
import torch
ROOT=Path('/home/antithetical/EPITA/PERSO/DynamicHorizons-Mamba-JEPA')
HERE=Path(__file__).resolve().parent
sys.path[:0]=[str(ROOT),str(ROOT/'artifacts/experiments/20260927_levers'),str(HERE)]
import teval as E
import tworld as T
from registration_scope import terrain_decision, shifted

CKPT=ROOT/'artifacts/eda/levers_tworlds_v1/corrt_raw_suffix_s7_fmamba.pt'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

@torch.no_grad()
def main():
    torch.set_num_threads(8)
    payload=torch.load(CKPT,map_location='cpu',weights_only=False)
    assert payload['args']['backbone']=='fmamba' and payload['args']['head']=='corrt'
    source=Path(T.__file__).read_text()
    now='POOLS = {"raw": ROOT / "artifacts/eda/spatial_pool_v1", "tc": ROOT / "artifacts/eda/spatial_pool_tc_v1",\n         "ldad10": ROOT / "artifacts/eda/spatial_pool_ldad10_v1",\n         "ldad1": ROOT / "artifacts/eda/spatial_pool_ldad1_v1"}'
    former='POOLS = {"raw": ROOT / "artifacts/eda/spatial_pool_v1", "tc": ROOT / "artifacts/eda/spatial_pool_tc_v1",\n         "ldad10": ROOT / "artifacts/eda/spatial_pool_ldad10_v1"}'
    assert now in source and hashlib.sha256(source.replace(now,former).encode()).hexdigest()==payload['script_sha256']
    model=T.TWorld('corrt',backbone='fmamba').cpu().eval();model.load_state_dict(payload['world'])
    layer=model.layers[0]
    cache=torch.load(Path(str(E.CACHE).format('raw')),map_location='cpu',weights_only=False,mmap=True)
    meta,_,_=E.split()
    pairs=[]
    for i in range(len(cache['ctx'])):
        for k in (0,1):
            act=int(cache['fut_a'][i,k])
            if not 1<=act<=4:continue
            v0=meta['root_visible'][i] if k==0 else meta['future_visible'][i,0,k-1]
            v1=meta['future_visible'][i,0,k]
            shift,_,_=terrain_decision(v0,v1,act)
            if shift>0:pairs.append((i,k,shift,int(meta['seed'][i])))
    def first_h(s,a,t):
        x=torch.cat([model.action(a)[:,None],model.embed(s)],1)+model.space+model.time[t]
        n=layer.n1(x)
        return layer.n2(x+layer.space(n,n,n,need_weights=False)[0])[:,1:64]
    total={'raw_fixed':0.,'raw_aligned':0.,'mixer_fixed':0.,'mixer_aligned':0.,
           'near_raw_fixed':0.,'near_raw_aligned':0.,'near_mixer_fixed':0.,'near_mixer_aligned':0.}
    by_seed={}
    for start in range(0,len(pairs),32):
        batch=pairs[start:start+32]
        s0=torch.stack([cache['ctx'][i,-1] if k==0 else cache['fut'][i,k-1] for i,k,_,_ in batch]).float()
        s1=torch.stack([cache['fut'][i,k] for i,k,_,_ in batch]).float()
        a0=torch.tensor([int(cache['fut_a'][i,k]) for i,k,_,_ in batch]);a1=torch.tensor([int(cache['fut_a'][i,k+1]) for i,k,_,_ in batch])
        h0=first_h(s0,a0,3+batch[0][1]) if all(k==batch[0][1] for _,k,_,_ in batch) else None
        h1=first_h(s1,a1,4+batch[0][1]) if all(k==batch[0][1] for _,k,_,_ in batch) else None
        if h0 is None:
            h0=torch.cat([first_h(s0[j:j+1],a0[j:j+1],3+k) for j,(_,k,_,_) in enumerate(batch)],0)
            h1=torch.cat([first_h(s1[j:j+1],a1[j:j+1],4+k) for j,(_,k,_,_) in enumerate(batch)],0)
        for j,(i,k,shift,seed) in enumerate(batch):
            bucket=by_seed.setdefault(seed,{q:0. for q in total})
            for prefix,u,v in [('raw',s0[j,:63],s1[j,:63]),('mixer',h0[j],h1[j])]:
                old,new,fixed=shifted(u.reshape(7,9,-1),v.reshape(7,9,-1),shift)
                df=(fixed-new).square().mean(-1);da=(old-new).square().mean(-1)
                dr,dc=((0,0),(-1,0),(1,0),(0,-1),(0,1))[shift]
                rr=torch.arange(max(0,-dr),7-max(0,dr))[:,None]
                cc=torch.arange(max(0,-dc),9-max(0,dc))[None,:]
                near=(rr>=2)&(rr<=4)&(cc>=3)&(cc<=5)
                values={f'{prefix}_fixed':float(df.mean()),f'{prefix}_aligned':float(da.mean()),
                        f'near_{prefix}_fixed':float(df[near].mean()),f'near_{prefix}_aligned':float(da[near].mean())}
                for q,value in values.items():total[q]+=value;bucket[q]+=value
    n=len(pairs);out={'status':'complete','pairs':n,'seeds':len(by_seed),'means':{q:v/n for q,v in total.items()},
                     'relative_reduction':{},'ci95':{},'checkpoint_sha256':sha(CKPT),
                     'checkpoint_model_source_sha256':payload['script_sha256'],
                     'raw_cache_sha256':sha(str(E.CACHE).format('raw')),'source_sha256':sha(__file__)}
    groups=list(by_seed.values());g=torch.Generator().manual_seed(20260929)
    for p in ('raw','mixer','near_raw','near_mixer'):
        out['relative_reduction'][p]=1-total[p+'_aligned']/total[p+'_fixed']
        bs=[]
        for _ in range(1000):
            ix=torch.randint(len(groups),(len(groups),),generator=g)
            f=sum(groups[int(j)][p+'_fixed'] for j in ix);a=sum(groups[int(j)][p+'_aligned'] for j in ix)
            bs.append(1-a/f)
        out['ci95'][p]=torch.tensor(bs).quantile(torch.tensor([.025,.975])).tolist()
    HERE.joinpath('mixer_registration.json').write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps(out,indent=2),flush=True)
if __name__=='__main__':main()
