"""Pre-training oracle erasure audit; uses only observation-derived change and real outcome targets."""
import os
os.environ.setdefault("JAX_PLATFORMS","cpu")
os.environ.setdefault("OMP_NUM_THREADS","4")
import json,math,sys
from pathlib import Path
import numpy as np
import torch
import lightgbm as lgb
from saccade import data as D,train as TR
P=Path(__file__).resolve().parent;E=P.parent
TEACH=D.ROOT/"artifacts/saccade_full_frame_change/targets/change_teacher.txt"

def entropy(x):
    z=np.clip(x,1e-300,1.0)
    return -(x*np.log(z)).sum(-1)

def b_gap(p):
    assert p.shape[0]==2
    r=np.stack([p,1-p],-1)
    return np.maximum(entropy(r.mean(0))-entropy(r).mean(0),0)

def scores(change,reward,term,edges):
    # Both labels available: do not let a different geometric mask create a score contrast.
    common=np.isfinite(change).all(0)
    gap=np.zeros_like(change[0],dtype=float)
    gap[common]=b_gap(change[:,common].astype(float))
    sym=lambda x:np.sign(x)*np.log1p(abs(x))
    bins=np.clip(np.searchsorted(edges,sym(reward),side="right")-1,0,len(edges)-2)
    rg=(bins[0]!=bins[1])*math.log(2)
    tg=(term[0]!=term[1])*math.log(2)
    # Pool the valid BCE entries across both members just as the training reduction would.
    # A coordinate scored in only one member has zero alias cost: the pooled optimum can match it.
    v=[float(2*gap.sum()/np.isfinite(change).sum()),float(rg.mean()),float(tg.mean())]
    return {"change":v[0],"reward":v[1],"termination":v[2],"aux":float(np.mean(v)),
            "largest_cell_time_gap":float(gap.max()),"steps_with_change_gap":(np.flatnonzero(gap.sum((1,2,3))>1e-10)+1).tolist(),
            "change_gap_by_step":gap.sum((1,2,3)).tolist(),"gap_sum_by_plane":gap.sum((0,1,2)).tolist(),
            "geometric_mask_disagreement":int(np.count_nonzero(np.isfinite(change[0])!=np.isfinite(change[1])))}

def main():
    import jax,jax.numpy as jnp
    assert jax.default_backend()=="cpu"
    torch.set_num_threads(2)
    model=lgb.Booster(model_file=str(TEACH))
    class Predict:
        def predict(self,x):return model.predict(x,num_threads=4)
    teacher=Predict()
    ck=torch.load(E/"20261106_p1f4/run/ckpt_10.pt",mmap=True,map_location="cpu",weights_only=False)
    edges=ck["model"]["outcome.edges"].numpy()
    bankp=E/"20261108_full_frame_change/phase_bank.pt"
    bank=torch.load(bankp,map_location="cpu",weights_only=False)
    L=dict(np.load(D.ROOT/"artifacts/eda/s0/s0_dev_labels.npz"))
    _,_,_,step,render=D.RP.env_and_render()
    def roll(st,keys,acts):
        fr=[np.asarray(render(st))];r=[];d=[]
        for k,a in zip(keys,acts):
            _,st,re,te,_=step(k,st,jnp.asarray(a,jnp.int32))
            fr.append(np.asarray(render(st)));r.append(float(re));d.append(bool(te))
        return np.array(fr),np.array(r),np.array(d)
    def target(f):return np.stack([D.change_probabilities(z,teacher) for z in f])
    assert abs(b_gap(np.array([0.,1.]))-math.log(2))<1e-12
    assert b_gap(np.array([.3,.3]))==0
    out=[];only=int(sys.argv[1]) if len(sys.argv)>1 else None
    for rec in bank:
        i=rec["e"];t=rec["t0"]
        if only is not None and i!=only:continue
        st=D.RP.advance_to(int(L["ep_shard"][i]),int(L["ep_slot"][i]),t)
        kk=jax.random.split(jax.random.PRNGKey(110812+i),50)
        f,r,d=map(np.stack,zip(*(roll(st.replace(player_hunger=jnp.asarray(v,st.player_hunger.dtype)),kk,[0]*40) for v in (20.,12.))))
        n=rec["frames"].shape[1]
        assert np.array_equal(f[:,:n],rec["frames"]) and np.array_equal(r[:,:n-1],rec["reward"])
        assert np.array_equal(d[:,:n-1],rec["term"])
        assert np.array_equal(f[0,17:25],f[1,17:25]) and np.array_equal(r[0,16:24],r[1,16:24])
        if d.any():
            out.append({"e":i,"excluded":"termination"});continue
        x=target(f[:,24:41]);rs=r[:,24:40];ts=d[:,24:40]
        row={"e":i,"early":{},"late":{},"exposure":{}}
        for h in (1,2,4,8,16):
            row["early"][str(h)]={"dense":scores(x[:,:h],rs[:,:h],ts[:,:h],edges),"endpoint":scores(x[:,h-1:h],rs[:,h-1:h],ts[:,h-1:h],edges)}
        # Same teacher targets; change only the fork/available prefix.
        assert np.array_equal(f[0,24:32],f[1,24:32])
        for h in (1,2,4,8):
            row["late"][str(h)]=scores(x[:,7:7+h],rs[:,7:7+h],ts[:,7:7+h],edges)
        # Use the same underlying trajectory, outcome and score in both rendering branches.
        for a in (0,1,3,6):
            ff=np.repeat(f[1:2,24:41],2,axis=0)
            ff[1,8:]=np.minimum(ff[1,8:].astype(np.int16)+a,255).astype(np.uint8)
            row["exposure"][str(a)]=scores(target(ff),np.repeat(rs[1:2],2,0),np.repeat(ts[1:2],2,0),edges)
        assert row["exposure"]["0"]["aux"]<1e-12
        out.append(row)
        print(i,row["early"]["16"]["dense"]["aux"],row["exposure"]["3"]["aux"],flush=True)
    # Reproduce an independently qualified actual achievement alias, not a synthetic reward label.
    st=D.RP.advance_to(395,15,159)
    kk=jax.random.split(jax.random.PRNGKey(822448839),8).reshape(2,4,2)[0]
    _,nx,_,_,_=step(kk[0],st,jnp.asarray(5,jnp.int32))
    earned=nx.achievements&~st.achievements
    assert np.flatnonzero(np.asarray(earned)).tolist()==[4]
    fa,ra,da=map(np.stack,zip(*(roll(s,kk,[5,6,16,3]) for s in (st,st.replace(achievements=st.achievements|earned)))))
    assert np.array_equal(fa[0],fa[1]) and np.array_equal(da[0],da[1])
    assert ra.tolist()==[[1.,0.,0.,0.],[0.,0.,0.,0.]]
    xa=target(fa)
    alias={str(h):scores(xa[:,:h],ra[:,:h],da[:,:h],edges) for h in (1,4)}
    report={"target_teacher_sha256":TR.file_sha256(TEACH),"bank_sha256":TR.file_sha256(bankp),
            "reward_alias":alias,"rows":out}
    (P/("RAW.json" if only is None else f"SCENE_{only}.json")).write_text(json.dumps(report,indent=2))
if __name__=="__main__":main()
