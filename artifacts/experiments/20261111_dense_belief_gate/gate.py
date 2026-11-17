"""Dense latent MSE erasure gate; frozen CPU simulator + GPU Mamba, no world updates."""
import os
os.environ.setdefault("JAX_PLATFORMS","cpu")
import json,importlib
from pathlib import Path
import numpy as np
import torch
from saccade import data as D,model as M,train as TR
P=Path(__file__).resolve().parent; E=P.parent
B=importlib.import_module("artifacts.experiments.20261111_bootstrap_target_audit.check")
DEV=torch.device("cuda")
FORK=24; H=16

def replay():
    import jax,jax.numpy as jnp
    assert jax.default_backend()=="cpu"
    bankpath=E/"20261108_full_frame_change/phase_bank.pt"
    bank=torch.load(bankpath,map_location="cpu",weights_only=False)
    L=dict(np.load(D.ROOT/"artifacts/eda/s0/s0_dev_labels.npz"))
    _,_,_,step,render=D.RP.env_and_render()
    def roll(s,keys,a):
        f=[np.asarray(render(s))];r=[];d=[];food=[int(s.player_food)]
        for k,x in zip(keys,a):
            _,s,re,de,_=step(k,s,jnp.asarray(x,jnp.int32))
            f.append(np.asarray(render(s)));r.append(float(re));d.append(bool(de));food.append(int(s.player_food))
        return np.asarray(f),np.asarray(r),np.asarray(d),np.asarray(food)
    rows=[];excluded=[]
    for r in bank:
        i=r["e"]; s=D.RP.advance_to(int(L["ep_shard"][i]),int(L["ep_slot"][i]),r["t0"])
        kk=jax.random.split(jax.random.PRNGKey(110812+i),50)
        f,rw,d,food=map(np.stack,zip(*(roll(s.replace(player_hunger=jnp.asarray(v,s.player_hunger.dtype)),kk,[0]*40) for v in (20.,12.))))
        n=len(r["frames"][0]);p=len(r["prefix_frames"])
        assert np.array_equal(f[:,:n],r["frames"]) and np.array_equal(food[:,:n],r["food"])
        assert np.array_equal(rw[:,:n-1],r["reward"]) and np.array_equal(d[:,:n-1],r["term"])
        assert np.array_equal(f[0,17:25],f[1,17:25]) and np.array_equal(rw[0,16:24],rw[1,16:24])
        assert np.array_equal(f[0,40],f[1,40])
        if d.any():
            excluded.append(i);continue
        assert (np.argmax(food[:,25:]<food[:,24:25],axis=1)+1).tolist()==[8,16]
        frames=np.concatenate([np.broadcast_to(r["prefix_frames"],(2,*r["prefix_frames"].shape)),f],1)
        a=np.concatenate([np.broadcast_to(r["prefix_actions"],(2,p)),np.zeros((2,41),dtype=np.int64)],1)
        rew=np.concatenate([np.broadcast_to(np.r_[0.,r["prefix_reward"]],(2,p+1)),rw],1)
        done=np.concatenate([np.broadcast_to(np.r_[False,r["prefix_term"]],(2,p+1)),d],1)
        fx=np.broadcast_to(D.fixation_prior(np.random.default_rng(12000+i),(len(frames[0]),)),(2,len(frames[0]),2))
        rows.append(dict(e=i,f=frames,a=a,r=rew,d=done,fx=fx,t=p+24))
        print("replay",i,flush=True)
    s=D.RP.advance_to(395,15,159)
    kk=jax.random.split(jax.random.PRNGKey(822448839),8).reshape(2,4,2)[0]
    _,n,_,_,_=step(kk[0],s,jnp.asarray(5,jnp.int32));earned=n.achievements&~s.achievements
    assert np.flatnonzero(np.asarray(earned)).tolist()==[4]
    f,r,d,_=map(np.stack,zip(*(roll(z,kk,[5,6,16,3]) for z in (s,s.replace(achievements=s.achievements|earned)))))
    assert np.array_equal(f[0],f[1]) and np.array_equal(d[0],d[1])
    assert r.tolist()==[[1.,0.,0.,0.],[0.,0.,0.,0.]]
    alias=dict(f=f,a=np.array([[5,6,16,3,0]]*2),r=np.column_stack([np.zeros(2),r]),
               d=np.column_stack([np.zeros(2,dtype=bool),d]),fx=np.full((2,5,2),4),t=0)
    return rows,excluded,alias,TR.file_sha256(bankpath)

def parts(s,y):
    s,y=s.float(),y.float()
    dy=y[0]-y[1]; ds=s[0]-s[1]
    oracle=dy.square().mean(-1)/4
    err=(s-y).square().mean((0,2))
    tied=(s.mean(0,keepdim=True)-y).square().mean((0,2))
    contrast=(ds-dy).square().mean(-1)/4
    shared=(s.mean(0)-y.mean(0)).square().mean(-1)
    assert torch.allclose(err,shared+contrast,atol=1e-6)
    assert torch.allclose(tied-err,oracle-contrast,atol=1e-6)
    assert torch.allclose((y.mean(0,keepdim=True)-y).square().mean((0,2)),oracle,atol=1e-7)
    return dict(oracle_gap=oracle.tolist(),actual_mse=err.tolist(),actual_tied_mse=tied.tolist(),actual_erasure_gap=(oracle-contrast).tolist(),
                dot=(ds*dy).sum(-1).tolist(),teacher_energy=dy.square().sum(-1).tolist())

@torch.no_grad()
def main():
    torch.set_num_threads(2)
    rows,excluded,alias,bank_hash=replay()
    result=dict(bank_sha256=bank_hash,excluded=excluded,checkpoints={})
    root=E/"20261111_p1_multistep_mamba/run"
    for ci in (1,5,10):
        path=root/("ckpt_10.orig_with_mask.pt" if ci==10 else f"ckpt_{ci:02d}.pt")
        ck=torch.load(path,weights_only=False,map_location="cpu",mmap=True)
        cfg=ck["cfg"];m=B.MM.MultiStepM0(cfg).to(DEV).eval();m.load_state_dict(ck["model"])
        def make(r):
            return B.tokens(m,*(r[k] for k in ("f","fx","a","r","d")))[0]
        bank=[]
        for r in rows:
            t=r["t"];split=2*t+1
            with TR.autocast(DEV,cfg):
                x=make(r)
                # Includes matched arrival reward/action/percept tokens for the last eight states.
                assert torch.equal(x[0,2*(t-7):split],x[1,2*(t-7):split])
                y,_=m.belief(x,m.belief.initial(2,DEV))
                _,c=m.belief(x[:,:split],m.belief.initial(2,DEV))
                q,_=m.belief(m.belief.rollout_tokens(B.tensor(r["a"][:,t:t+H],torch.long)),c)
                p=parts(q[:,1::2],y[:,2*(t+1)::2])
                assert len(p["oracle_gap"])==H
                # Counterfactual: only earlier history can distinguish the teachers.
                shared=x.clone();shared[0,split:]=shared[1,split:]
                yy,_=m.belief(shared,m.belief.initial(2,DEV))
                p["same_future_input_teacher_gap"]=(yy[0,2*(t+1)::2].float()-yy[1,2*(t+1)::2].float()).square().mean(-1).div(4).tolist()
                p["e"]=r["e"]
                # The two foil histories and outcomes are identical; brightness alone changes one stream from elapsed step8.
                fs=np.repeat(r["f"][1:2],2,axis=0)
                ff={**r,**{k:np.repeat(r[k][1:2],2,axis=0) for k in ("r","d","a","fx")}}
                f={0:fs}
                for v in (1,3,6):
                    fx=fs.copy();fx[1,t+8:]=np.minimum(255,fx[1,t+8:].astype(np.int16)+v).astype(np.uint8);f[v]=fx
                p["exposure"]={}
                for v,fv in f.items():
                    yy,_=m.belief(make({**ff,"f":fv}),m.belief.initial(2,DEV))
                    g=(yy[0,2*(t+1)::2].float()-yy[1,2*(t+1)::2].float()).square().mean(-1).div(4)
                    if v==0:assert float(g.max())<1e-8
                    p["exposure"][str(v)]=g.tolist()
            bank.append(p)
        with TR.autocast(DEV,cfg):
            x=make(alias)
            assert torch.equal(x[0,:2],x[1,:2])
            yy,_=m.belief(x,m.belief.initial(2,DEV))
            rew_gap=(yy[0,2::2].float()-yy[1,2::2].float()).square().mean(-1).div(4).tolist()
        result["checkpoints"][str(ci)]=dict(sha256=TR.file_sha256(path),rows=bank,reward_alias_oracle_gap=rew_gap)
        (P/"RAW.json").write_text(json.dumps(result,indent=2))
        print("complete checkpoint",ci,flush=True)
if __name__=="__main__":main()
