"""Same-history checkpoint study. Frozen model, no training of world weights."""
import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")
import json
from pathlib import Path
import numpy as np
import torch
from torch import nn
from saccade import data as D, model as M, train as TR
from saccade.eval import s1 as S, gates as G

HERE=Path(__file__).resolve().parent
E=HERE.parent
ROOTS={"hud":E/"20261109_p1_hud_change/run","multi":E/"20261111_p1_multistep_mamba/run"}
CHECKS=(1,3,5,6,8,10)
DEVICE=torch.device("cuda")
KINDS=("hunger","thirst","fatigue","sleep_duration","zombie_side","achievement_drink","achievement_zombie")
WIDTHS=(1,1,1,1,4,1,1)

def bank():
    path=S.LABELS/"s0_dev_labels.npz"
    L=dict(np.load(path))
    idx=np.sort(np.random.default_rng(1112).choice(len(L["ep_T"]),192,replace=False))
    C=np.load(E/"20261107_f4_followups/readings/counters_dev.npz")
    cnt={k:C[k] for k in ("hunger","thirst","fatigue")}
    so=np.r_[0,np.cumsum(L["ep_T"]+1)];ao=np.r_[0,np.cumsum(L["ep_T"])]
    grids=G.episode_grids(L)
    rows=[];xx=[];yy=[];ep=[];oo=[]
    for k,g in enumerate(grids):
        if k not in idx:continue
        a,b=so[k:k+2];c,d=ao[k:k+2];T=d-c
        body=L["s_body"][a:b].astype(int);sleep=L["s_sleep"][a:b]
        inv=L["s_inv"][a:b];ach=L["s_ach"][a:b]
        step=np.arange(T+1)
        prev_action=np.r_[-1,L["t_action"][c:d]]
        act=np.r_[L["t_action"][c:d],0]
        rw=np.r_[0.,L["t_reward"][c:d]]
        term=np.r_[False,L["t_terminated"][c:d]]
        observation=np.column_stack([body,inv,sleep,L["s_light"][a:b],rw,term,np.eye(18)[prev_action+1]])
        take=np.arange(8,T)
        recent=np.concatenate([observation[take-j] for j in range(8)]+[np.eye(17)[act[take]]],axis=1)
        y=np.full((T+1,len(KINDS)),np.nan);oracle=np.full((T+1,4),np.nan)
        last_sleep=np.maximum.accumulate(np.where(sleep,step,-1))
        for j,(name,col) in enumerate(zip(KINDS[:3],(1,2,3))):
            delta=np.r_[0,np.diff(body[:,col])]
            last=np.maximum.accumulate(np.where(delta!=0,step,-1));age=step-last
            ok=(last>=0)&(age>8)&(~sleep)&(last_sleep<last)&(body[:,col]>0)&(body[:,col]<9)
            ok&=delta[last.clip(0)]==-1
            y[ok,j]=cnt[name][a:b][ok];oracle[ok,j]=age[ok]
        last_awake=np.maximum.accumulate(np.where(~sleep,step,-1))
        dur=step-last_awake;ok=sleep&(dur>8)
        y[ok,3]=dur[ok];oracle[ok,3]=dur[ok]
        dep,side=S.departed_episode(g["grid"])
        ok=(dep[:,0]>=3)&(side[:,0]>=0)
        y[ok,4]=side[ok,0]
        y[:,5]=ach[:,4];y[:,6]=ach[:,8] # S0 achievement order, verified against enum below
        es=S.episodes(L)[k]
        rows.append((es,take,act,rw,term))
        xx.append(recent);yy.append(y[take]);oo.append(oracle[take]);ep.extend([k]*len(take))
    import craftax.craftax_classic.constants as CCF
    # Use named enum values; the numeric columns here are diagnostic labels only.
    assert CCF.Achievement.COLLECT_DRINK.value==4 and CCF.Achievement.DEFEAT_ZOMBIE.value==8
    return rows,np.concatenate(xx).astype("f"),np.concatenate(yy).astype("f"),np.asarray(ep),np.concatenate(oo)

@torch.no_grad()
def encode(path,rows,initial=None):
    ck=torch.load(path,weights_only=False,map_location="cpu",mmap=True)
    cfg=ck["cfg"]
    if initial is None:
        m=M.M0(cfg).to(DEVICE).eval()
        m.load_state_dict({k:v for k,v in ck["model"].items() if k!="belief.mask_token"})
    else:
        import importlib
        cls=(M.M0 if initial=="hud" else importlib.import_module("artifacts.experiments.20261111_p1_multistep_mamba.model_multistep").MultiStepM0)
        torch.manual_seed(1)
        m=cls(cfg).to(DEVICE).eval()
    b=[];bp=[]
    for e,t,a,r,d in rows:
        f=D.RP.episode_fields(e.shard,e.slot)["observations"].numpy()
        n=len(f);rng=np.random.default_rng([cfg["p1"]["monitor"]["fixed_fixation_seed"],e.shard,e.slot])
        fix=torch.as_tensor(D.fixation_prior(rng,(n,)),device=DEVICE)[None]
        # Perception batched by frame. Recurrent scan is the full causal episode, not a cropped chunk.
        with TR.autocast(DEVICE,cfg):
            P=[m.perception(torch.as_tensor(f[i:i+512],device=DEVICE)[None],fix[:,i:i+512]) for i in range(0,n,512)]
            p=M.Percept(*(torch.cat(z,1) for z in zip(*[v.parts() for v in P])),fix)
            emb=m.belief.percept_embed(p)
            tok=m.belief.tokens(emb,torch.as_tensor(r,dtype=torch.float32,device=DEVICE)[None],torch.as_tensor(d,device=DEVICE)[None],torch.as_tensor(a.astype(np.int64),device=DEVICE)[None])
            out,_=m.belief(tok,m.belief.initial(1,DEVICE))
        b.append(out[0,2*t].float().cpu().numpy());bp.append(out[0,2*t+1].float().cpu().numpy())
    return np.concatenate(b),np.concatenate(bp)

def scores(y,p):
    ans={}
    for j,name in enumerate(KINDS):
        m=np.isfinite(y[:,j]);v=y[m,j]
        z=p[j][m]
        if j<4:
            ans[name]=float(1-((v-z[:,0])**2).sum()/((v-v.mean())**2).sum())
        elif j==4:ans[name]=float((z.argmax(1)==v).mean())
        else:ans[name]=G.average_precision(v>0,z[:,0])
    return ans

def ridge(X,Y,fold):
    X=torch.as_tensor(X,device=DEVICE)
    yy=torch.as_tensor(Y,device=DEVICE)
    out=[np.zeros((len(Y),w),"f") for w in WIDTHS]
    for j,w in enumerate(WIDTHS):
        valid=np.isfinite(Y[:,j])
        for f in range(4):
            tr=(fold!=f)&valid;te=(fold==f)&valid
            xx=X[tr];xz=X[te];y=yy[tr,j]
            if w>1:y=torch.nn.functional.one_hot(y.long(),w).float()
            else:y=y[:,None]
            mean=xx.mean(0);sd=xx.std(0).clamp_min(1e-5)
            xx=(xx-mean)/sd;xz=(xz-mean)/sd
            mu=y.mean(0)
            wgt=torch.linalg.solve(xx.T@xx+10*torch.eye(xx.shape[1],device=DEVICE),xx.T@(y-mu))
            out[j][te]=(xz@wgt+mu).cpu().numpy()
    return scores(Y,out)

def nonlinear(X,Y,fold,ep):
    X=torch.as_tensor(X,device=DEVICE); yy=torch.as_tensor(Y,device=DEVICE)
    out=[np.zeros((len(Y),w),"f") for w in WIDTHS]
    loss_fn=nn.functional
    for f in range(4):
        te=fold==f
        val=(fold==(f+1)%4)
        tr=~(te|val)
        mu=X[tr].mean(0);sd=X[tr].std(0).clamp_min(1e-5)
        x=((X-mu)/sd).clamp(-20,20)
        ym=torch.nanmean(yy[tr],0);ss=torch.nanmean((yy[tr]-ym).square(),0).sqrt().clamp_min(1e-5)
        torch.manual_seed(54321+f)
        net=nn.Sequential(nn.Linear(X.shape[1],128),nn.GELU(),nn.Linear(128,128),nn.GELU(),nn.Linear(128,sum(WIDTHS))).to(DEVICE)
        opt=torch.optim.AdamW(net.parameters(),lr=.001)
        rng=torch.Generator(device=DEVICE).manual_seed(99+f)
        it=torch.as_tensor(np.flatnonzero(tr),device=DEVICE)
        iv=torch.as_tensor(np.flatnonzero(val),device=DEVICE)
        def loss(i):
            z=net(x[i]).split(WIDTHS,dim=-1); v=yy[i]
            tot=[]
            for j,q in enumerate(z):
                m=v[:,j].isfinite()
                if not m.any():continue
                if j<4:tot.append((q[m,0]-(v[m,j]-ym[j])/ss[j]).square().mean())
                elif j==4:tot.append(loss_fn.cross_entropy(q[m],v[m,j].long()))
                else:tot.append(loss_fn.binary_cross_entropy_with_logits(q[m,0],v[m,j]))
            return torch.stack(tot).mean()
        best=1e9
        for epoch in range(18):
            for i in it[torch.randperm(len(it),generator=rng,device=DEVICE)].split(2048):
                opt.zero_grad(set_to_none=True);loss(i).backward();opt.step()
            with torch.no_grad():
                v=float(loss(iv))
                if v<best:
                    best=v
                    for j,q in enumerate(net(x[te]).split(WIDTHS,dim=-1)):
                        if j<4:q=q*ss[j]+ym[j]
                        out[j][te]=q.cpu().numpy()
    return scores(Y,out)

def study_data():
    torch.set_num_threads(2)
    rows,X,Y,ep,oracle=bank()
    eid=np.unique(ep);perm=np.random.default_rng(1113).permutation(eid)
    fold_map=dict(zip(perm,np.arange(len(perm))%4))
    fold=np.array([fold_map[e] for e in eid])[np.searchsorted(eid,ep)]
    return rows,X,Y,ep,oracle,eid,fold

def initial():
    rows,X,Y,ep,oracle,eid,fold=study_data()
    out={}
    for arm,root in ROOTS.items():
        b,bp=encode(root/"ckpt_01.pt",rows,initial=arm)
        out[arm]={"ridge_b":ridge(b,Y,fold),"ridge_bp":ridge(bp,Y,fold),
                  "ridge_recent_bp":ridge(np.concatenate([X,bp],1),Y,fold),
                  "mlp_recent_bp":nonlinear(np.concatenate([X,bp],1),Y,fold,ep)}
    (HERE/"UNTRAINED.json").write_text(json.dumps(out,indent=2))
    print(json.dumps(out,indent=2),flush=True)

def main():
    rows,X,Y,ep,oracle,eid,fold=study_data()
    ref={}
    for j,name in enumerate(KINDS[:4]):
        v=Y[:,j];m=np.isfinite(v);z=oracle[m,j]
        ref[name]={"rows":int(m.sum()),"episodes":len(np.unique(ep[m])),"exact":float((v[m]==z).mean()),
                   "r2":float(1-((v[m]-z)**2).sum()/((v[m]-v[m].mean())**2).sum())}
    report={"n_episodes":len(eid),"n_rows":len(ep),"episodes":eid.tolist(),"oracle":ref,"checkpoints":{}}
    report["recent"]={"ridge":ridge(X,Y,fold),"mlp":nonlinear(X,Y,fold,ep)}
    print("reference",ref,flush=True)
    for i in CHECKS:
        for name,root in ROOTS.items():
            p=root/f"ckpt_{i:02d}.pt"
            b,bp=encode(p,rows)
            out={"sha256":TR.file_sha256(p),"ridge_b":ridge(b,Y,fold),"ridge_bp":ridge(bp,Y,fold),
                 "ridge_recent_bp":ridge(np.concatenate([X,bp],1),Y,fold),
                 "mlp_recent_bp":nonlinear(np.concatenate([X,bp],1),Y,fold,ep)}
            report["checkpoints"][f"{name}_{i}"]=out
            (HERE/"TRAJECTORY.json").write_text(json.dumps(report,indent=2))
            print(name,i,out["ridge_bp"],out["mlp_recent_bp"],flush=True)

if __name__=="__main__":
    import sys
    (initial if "--untrained" in sys.argv else main)()
