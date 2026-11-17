"""Frozen bootstrap target audit. No world training or edits."""
import os
os.environ.setdefault("JAX_PLATFORMS","cpu")
import importlib,json,sys
from pathlib import Path
import numpy as np
import torch
from saccade import data as D,model as M,train as TR
from saccade.eval import s1 as S

P=Path(__file__).resolve().parent; E=P.parent
MM=importlib.import_module("artifacts.experiments.20261111_p1_multistep_mamba.model_multistep")
DEV=torch.device("cuda")

def tensor(x,dtype=None):
    return torch.as_tensor(np.array(x,copy=True),device=DEV,dtype=dtype)

def tokens(model, f,fix, a,r,d):
    p=model.perception(tensor(f),tensor(fix))
    return model.belief.tokens(model.belief.percept_embed(p),tensor(r,torch.float32),tensor(d,torch.bool),tensor(a,torch.long)),p

def loss_parts(s,y):
    s,y=s.float(),y.float()
    ds=s[0]-s[1];dy=y[0]-y[1]
    shared=(s.mean(0)-y.mean(0)).square().mean()
    contrast=(ds-dy).square().mean()/4
    pair_mean=shared+dy.square().mean()/4
    assert torch.allclose((s-y).square().mean(),shared+contrast,atol=1e-6)
    return dict(mse=float(shared+contrast),mse_if_pair_mean=float(pair_mean),
                mse_saved_by_contrast=float(pair_mean-shared-contrast),
                teacher_contrast_energy=float(dy.square().mean()/4),
                contrast_gain=float((ds@dy)/dy.square().sum().clamp_min(1e-12)),
                student_contrast_rms=float(ds.square().mean().sqrt()))

@torch.no_grad()
def pairs(model,cfg):
    bank=torch.load(E/"20261108_full_frame_change/phase_bank.pt",weights_only=False)
    out=[]
    for i,r in enumerate(bank):
        n=len(r["prefix_frames"]); e=r["end"]; T=n+e+2
        f=np.concatenate([np.broadcast_to(r["prefix_frames"],(2,*r["prefix_frames"].shape)),r["frames"][:,:e+2]],1)
        a=np.concatenate([np.broadcast_to(r["prefix_actions"],(2,n)),np.zeros((2,e+2),np.int64)],1)
        rw=np.concatenate([np.broadcast_to(np.r_[0.,r["prefix_reward"]],(2,n+1)),r["reward"][:,:e+1]],1)
        td=np.concatenate([np.broadcast_to(np.r_[False,r["prefix_term"]],(2,n+1)),r["term"][:,:e+1]],1)
        fx=np.broadcast_to(D.fixation_prior(np.random.default_rng(12000+i),(T,)),(2,T,2))
        assert r["food"][0,e+1] < r["food"][1,e+1]
        with TR.autocast(DEV,cfg):
            x,pp=tokens(model,f,fx,a,rw,td)
            y,_=model.belief(x,model.belief.initial(2,DEV))
            target=y[:,-2]
            # Teacher endpoint positive control: the actual observed digit changes its embedding.
            center=pp.p[:, -2,64].float()
            end=pp.p[:, -1,64].float()
            assert torch.equal(center[0],center[1]) and (end[0]-end[1]).norm()>.1
            before=y[:,-4]
            control=x.clone();control[0,-2]=control[1,-2]
            yc,_=model.belief(control,model.belief.initial(2,DEV))
            row=dict(e=r["e"],teacher_before_rms=float((before[0]-before[1]).float().square().mean().sqrt()),
                     teacher_equal_endpoint_input_rms=float((yc[0,-2]-yc[1,-2]).float().square().mean().sqrt()),
                     teacher_after_rms=float((target[0]-target[1]).float().square().mean().sqrt()),h={})
            for h in (2,4,8):
                t=T-1-h; split=2*t+1
                assert np.array_equal(f[0,t-7:t+1],f[1,t-7:t+1])
                assert np.array_equal(a[0,t:t+h],a[1,t:t+h])
                _,carry=model.belief(x[:,:split],model.belief.initial(2,DEV))
                s,_=model.belief(model.belief.rollout_tokens(tensor(a[:,t:t+h],torch.long)),carry)
                bad,_=model.belief(x[:,split:-1],carry,torch.ones((2,2*h),device=DEV,dtype=torch.long))
                # Same pair of genuine starting histories. Only the observed teacher loses its prefix in the bad fork.
                row["h"][str(h)]={"observed":loss_parts(s[:,-1],target),"unrebased":loss_parts(s[:,-1],bad[:,-1]),
                                   "oracle":loss_parts(target,target)}
        out.append(row)
    return out

@torch.no_grad()
def ordinary(model,cfg):
    L=dict(np.load(S.LABELS/"s0_dev_labels.npz"))
    so=np.r_[0,np.cumsum(L["ep_T"]+1)];ao=np.r_[0,np.cumsum(L["ep_T"])]
    rng=np.random.default_rng(111104); out=[]
    eps=S.episodes(L)
    for i in rng.permutation(len(eps)):
        e=eps[i]
        if e.T<64:continue
        t=int(rng.integers(32,e.T-16))
        h=8 if len(out)%2==0 else 16
        f=D.RP.episode_fields(e.shard,e.slot)["observations"][:t+h+1].numpy()
        a=np.r_[L["t_action"][ao[i]:ao[i]+t+h],0][None]
        r=np.r_[0.,L["t_reward"][ao[i]:ao[i]+t+h]][None]
        d=np.r_[False,L["t_terminated"][ao[i]:ao[i]+t+h]][None]
        fix=D.fixation_prior(np.random.default_rng([12011,e.shard,e.slot]),(1,len(f)))
        with TR.autocast(DEV,cfg):
            x,_=tokens(model,f[None],fix,a,r,d)
            y,_=model.belief(x,model.belief.initial(1,DEV))
            target=y[:,-2].float()
            s=2*t+1
            _,full=model.belief(x[:,:s],model.belief.initial(1,DEV))
            _,recent=model.belief(x[:,2*(t-7):s],model.belief.initial(1,DEV))
            masked=model.belief.rollout_tokens(tensor(a[:,t:t+h],torch.long))
            sa,_=model.belief(masked,full)
            sb,_=model.belief(masked,recent)
            errs=[float((z[:,-1].float()-target).square().mean()) for z in (sa,sb)]
        out.append(dict(ep=int(i),t=t,h=h,full=errs[0],recent=errs[1]))
        if len(out)==80:break
    return out

def main():
    torch.set_num_threads(2)
    root=E/"20261111_p1_multistep_mamba/run"
    out={}
    for ci in (1,5,10):
        path=root/("ckpt_10.orig_with_mask.pt" if ci==10 else f"ckpt_{ci:02d}.pt")
        ck=torch.load(path,map_location="cpu",weights_only=False,mmap=True)
        model=MM.MultiStepM0(ck["cfg"]).to(DEV).eval();model.load_state_dict(ck["model"])
        out[str(ci)]={"checkpoint_sha256":TR.file_sha256(path),"pairs":pairs(model,ck["cfg"]),"ordinary":ordinary(model,ck["cfg"])}
        (P/"RAW.json").write_text(json.dumps(out,indent=2))
        print("completed checkpoint",ci,flush=True)
if __name__=="__main__": main()
