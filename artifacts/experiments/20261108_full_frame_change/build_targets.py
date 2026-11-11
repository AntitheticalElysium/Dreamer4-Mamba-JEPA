"""Build the NEW full training store efficiently, reusing historical scores only for byte-identical detector inputs.
The legacy schema is interpreted solely as a computation cache. It never excludes a new target location or mode.
Checksums and direct canonical recomputation verify the reuse; old artifacts are never overwritten."""
import json,shutil,time
import numpy as np
import lightgbm as lgb
from saccade import data as D,train as TR

def main():
    cfg=TR.load_cfg();eps=D.load_split(cfg)['train'];off=D.frame_offsets(eps);outdir=D.targets_dir(cfg)
    outdir.mkdir(parents=True,exist_ok=True)
    if (outdir/'change_targets.npy').exists():raise FileExistsError('new target store already exists')
    olddir=D.ROOT/'artifacts/saccade_m0/targets';old=np.load(olddir/'change_targets.npy',mmap_mode='r')
    assert old.shape[0]==off[-1] and old.shape[-1]==D.GRID
    model=lgb.Booster(model_file=str(olddir/'change_teacher.txt'));A=D._a9b();H=old.shape[-2]
    out=np.lib.format.open_memmap(outdir/'change_targets.pending.npy','w+',np.float16,(int(off[-1]),D.GRID,D.GRID,2));out[:]=np.nan
    rr,cc=np.meshgrid(np.arange(H),np.arange(D.GRID),indexing='ij')
    counts={'exact':0,'reused':0,'predicted':0};start=time.time()
    for i,e in enumerate(eps):
        frames=D.RP.episode_fields(e.shard,e.slot)['observations'].numpy();c,source,valid=D.change_correspondence(frames)
        exact=(np.abs(c[1:,:,:,None]-source).max((-1,-2,-3))==0)&valid
        p=np.full(valid.shape,np.nan,np.float32);p[exact]=0
        # Reconstruct the historical cache's source correspondence from its own grid height, not a new region rule.
        pc=c[:,:H];cost=[]
        for dr,dc in A.SHIFTS:
            r0,r1=max(0,-dr),min(H,H-dr);c0,c1=max(0,-dc),min(D.GRID,D.GRID-dc)
            cost.append(np.abs(pc[:-1,r0:r1,c0:c1]-pc[1:,r0+dr:r1+dr,c0+dc:c1+dc]).mean((1,2,3,4,5)))
        shift=np.asarray(A.SHIFTS)[np.stack(cost,1).argmin(1)]
        sr=rr[None]-shift[:,0,None,None];sq=cc[None]-shift[:,1,None,None]
        prior=pc[:-1][np.arange(e.T)[:,None,None],sr.clip(0,H-1),sq.clip(0,D.GRID-1)]
        values=old[off[i]+1:off[i+1]]
        same=(source[:,:H]==prior[:,:,:,None]).all((-1,-2,-3))
        reuse=same&np.isfinite(values[:,:,:,None])&valid[:,:H]&~exact[:,:H]
        p[:,:H][reuse]=np.broadcast_to(values[:,:,:,None],same.shape)[reuse]
        todo=valid&~np.isfinite(p);pi,ri,qi,mode=np.nonzero(todo)
        if len(pi):p[pi,ri,qi,mode]=model.predict(A.feats(c[pi+1,ri,qi],source[pi,ri,qi,mode]),num_threads=8)
        out[off[i]+1:off[i+1]]=p
        counts['exact']+=int(exact.sum());counts['reused']+=int(reuse.sum());counts['predicted']+=len(pi)
        if i%100==0:print(f'{i}/{len(eps)} episodes, {time.time()-start:.1f}s, {counts}',flush=True)
    out.flush()
    checks=[]
    for i in (0,len(eps)//2,len(eps)-1):
        e=eps[i];f=D.RP.episode_fields(e.shard,e.slot)['observations'].numpy()
        direct=D.change_probabilities(f,model).astype(np.float16);cached=out[off[i]+1:off[i+1]]
        ok=np.array_equal(direct,cached,equal_nan=True);checks.append({'episode':e.key,'bitwise_equal':ok,'frames':e.T})
        assert ok,'cache reuse differs from direct canonical recomputation'
    shutil.copy2(olddir/'change_teacher.txt',outdir/'change_teacher.txt')
    meta={'format':D.CHANGE_FORMAT,'shape':list(out.shape),'episodes_sha256':__import__('hashlib').sha256('\n'.join(e.key for e in eps).encode()).hexdigest(),
          'semantic_exclusions':[],'modes':list(D.CHANGE_MODES),'construction':counts,'recomputation_checks':checks,
          'teacher_sha256':TR.file_sha256(olddir/'change_teacher.txt'),'legacy_cache_sha256':TR.file_sha256(olddir/'change_targets.npy'),
          'elapsed_seconds':time.time()-start}
    (outdir/'change_targets.pending.npy').replace(outdir/'change_targets.npy')
    (outdir/'change_targets.json').write_text(json.dumps(meta,indent=2))
    D.validate_change_targets(out,eps,outdir)
    print(json.dumps(meta,indent=2),flush=True)

if __name__=='__main__':main()
