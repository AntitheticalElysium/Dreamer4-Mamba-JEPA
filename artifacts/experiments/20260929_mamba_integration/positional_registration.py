"""Exploratory stage-by-stage registration within frozen first Mamba layer."""
import hashlib,json,sys
from pathlib import Path
import torch
ROOT=Path('/home/antithetical/EPITA/PERSO/DynamicHorizons-Mamba-JEPA')
HERE=Path(__file__).resolve().parent
sys.path[:0]=[str(ROOT),str(ROOT/'artifacts/experiments/20260927_levers'),str(HERE)]
import teval as E
import tworld as T
from registration_scope import terrain_decision,shifted
CKPT=ROOT/'artifacts/eda/levers_tworlds_v1/corrt_raw_suffix_s7_fmamba.pt'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
@torch.no_grad()
def main():
    torch.set_num_threads(8)
    payload=torch.load(CKPT,map_location='cpu',weights_only=False)
    source=Path(T.__file__).read_text()
    now='POOLS = {"raw": ROOT / "artifacts/eda/spatial_pool_v1", "tc": ROOT / "artifacts/eda/spatial_pool_tc_v1",\n         "ldad10": ROOT / "artifacts/eda/spatial_pool_ldad10_v1",\n         "ldad1": ROOT / "artifacts/eda/spatial_pool_ldad1_v1"}'
    former='POOLS = {"raw": ROOT / "artifacts/eda/spatial_pool_v1", "tc": ROOT / "artifacts/eda/spatial_pool_tc_v1",\n         "ldad10": ROOT / "artifacts/eda/spatial_pool_ldad10_v1"}'
    assert now in source and hashlib.sha256(source.replace(now,former).encode()).hexdigest()==payload['script_sha256']
    model=T.TWorld('corrt',backbone='fmamba').cpu().eval();model.load_state_dict(payload['world'])
    layer=model.layers[0]
    cache=torch.load(Path(str(E.CACHE).format('raw')),map_location='cpu',weights_only=False,mmap=True)
    meta,_,_=E.split();by_seed={};counts={}
    stages=('projected','plus_screen_pos','plus_time_pos','mamba_input')
    def features(s,a,t):
        proj=model.embed(s)
        screen=proj+model.space[1:]
        time=screen+model.time[t]
        action=model.action(a)[:,None]+model.space[:1]+model.time[t]
        x=torch.cat([action,time],1)
        n=layer.n1(x);h=layer.n2(x+layer.space(n,n,n,need_weights=False)[0])
        return dict(zip(stages,(proj[:,:63],screen[:,:63],time[:,:63],h[:,1:64])))
    for k in (0,1):
        pairs=[]
        for i in range(len(cache['ctx'])):
            act=int(cache['fut_a'][i,k]);
            if not 1<=act<=4:continue
            v0=meta['root_visible'][i] if k==0 else meta['future_visible'][i,0,k-1]
            v1=meta['future_visible'][i,0,k]
            shift,_,_=terrain_decision(v0,v1,act)
            if shift>0:pairs.append((i,shift,int(meta['seed'][i])))
        counts[str(k)]=len(pairs)
        for start in range(0,len(pairs),32):
            chunk=pairs[start:start+32]
            s0=torch.stack([cache['ctx'][i,-1] if k==0 else cache['fut'][i,k-1] for i,_,_ in chunk]).float()
            s1=torch.stack([cache['fut'][i,k] for i,_,_ in chunk]).float()
            a0=torch.tensor([int(cache['fut_a'][i,k]) for i,_,_ in chunk])
            a1=torch.tensor([int(cache['fut_a'][i,k+1]) for i,_,_ in chunk])
            f0,f1=features(s0,a0,3+k),features(s1,a1,4+k)
            for j,(i,shift,seed) in enumerate(chunk):
                d=by_seed.setdefault(seed,{q:[0.,0.] for q in stages})
                for q in stages:
                    old,new,fixed=shifted(f0[q][j].reshape(7,9,-1),f1[q][j].reshape(7,9,-1),shift)
                    d[q][0]+=float((fixed-new).square().mean());d[q][1]+=float((old-new).square().mean())
    rows=list(by_seed.values());g=torch.Generator().manual_seed(20260929);out={'status':'complete','pairs':sum(counts.values()),'pairs_by_transition':counts,'seeds':len(rows),'stages':{},
         'checkpoint_sha256':sha(CKPT),'checkpoint_model_source_sha256':payload['script_sha256'],
         'cache_sha256':sha(str(E.CACHE).format('raw')),'source_sha256':sha(__file__)}
    for q in stages:
        fixed=sum(r[q][0] for r in rows);aligned=sum(r[q][1] for r in rows)
        bs=[]
        for _ in range(1000):
            ix=torch.randint(len(rows),(len(rows),),generator=g)
            a=sum(rows[int(j)][q][1] for j in ix);f=sum(rows[int(j)][q][0] for j in ix)
            bs.append(1-a/f)
        out['stages'][q]={'fixed_mse':fixed/out['pairs'],'aligned_mse':aligned/out['pairs'],
                          'relative_reduction':1-aligned/fixed,'seed_ci95':torch.tensor(bs).quantile(torch.tensor([.025,.975])).tolist()}
    HERE.joinpath('positional_registration.json').write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps(out,indent=2),flush=True)
if __name__=='__main__':main()
