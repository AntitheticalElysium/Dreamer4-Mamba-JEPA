"""Frozen auxiliary and fixed-cohort belief audit."""
import os
os.environ.setdefault("JAX_PLATFORMS","cpu")
import importlib,json
import numpy as np
import torch
from saccade import model as M,train as TR
from .native import RUNS,P,dev
B=importlib.import_module("artifacts.experiments.20261111_bootstrap_target_audit.check")
G=importlib.import_module("artifacts.experiments.20261111_dense_belief_gate.gate")
T=importlib.import_module("artifacts.experiments.20261111_belief_trajectory.trajectory")

def auxiliary(rows,path):
    ck=torch.load(path,map_location="cpu",mmap=True,weights_only=False);cfg=ck["cfg"]
    m=B.MM.MultiStepM0(cfg).to(dev).eval();m.load_state_dict(ck["model"])
    out=[]
    with torch.no_grad(),TR.autocast(dev,cfg):
        for r in rows:
            t=r["t"];split=2*t+1
            x,_=B.tokens(m,*(r[k] for k in ("f","fx","a","r","d")))
            assert torch.equal(x[0,2*(t-7):split],x[1,2*(t-7):split])
            y,_=m.belief(x,m.belief.initial(2,dev))
            _,carry=m.belief(x[:,:split],m.belief.initial(2,dev))
            q,_=m.belief(m.belief.rollout_tokens(B.tensor(r["a"][:,t:t+16],torch.long)),carry)
            true=y[:,2*(t+1)::2]
            stats=G.parts(q[:,1::2],true)
            assert len(stats["oracle_gap"])==16
            shared=x.clone();shared[0,split:]=shared[1,split:]
            yy,_=m.belief(shared,m.belief.initial(2,dev))
            dy=yy[0,2*(t+1)::2].float()-yy[1,2*(t+1)::2].float()
            stats["same_future_input_teacher_gap"]=(dy.square().mean(-1)/4).tolist()
            stats["e"]=r["e"]
            stats["oracle_and_predicted_contrast_rms"]=[
                (z[0]-z[1]).float().square().mean(-1).sqrt().tolist() for z in (true,q[:,1::2])]
            out.append(stats)
    return {"sha256":TR.file_sha256(path),"rows":out}

def main():
    torch.set_num_threads(2)
    gate,excluded,_,h=G.replay()
    rows,X,Y,ep,oracle,eid,fold=T.study_data()
    r={"same_dev_episodes":eid.tolist(),"row_count":len(Y),"bank_sha256":h,
       "dev_reference":{k:{"rows":int(np.isfinite(Y[:,i]).sum()),
                          "oracle_exact":bool(np.array_equal(Y[np.isfinite(Y[:,i]),i],oracle[np.isfinite(Y[:,i]),i]))}
                        for i,k in enumerate(T.KINDS[:4])},
       "excluded_latent_gate":excluded,"recent":{"ridge":T.ridge(X,Y,fold),"mlp":T.nonlinear(X,Y,fold,ep)},"checkpoints":{}}
    for name,path in RUNS.items():
        b,bp=T.encode(path,rows)
        q={"sha256":TR.file_sha256(path),"ridge_b":T.ridge(b,Y,fold),"ridge_bp":T.ridge(bp,Y,fold),
           "ridge_recent_bp":T.ridge(np.concatenate([X,bp],1),Y,fold),
           "mlp_recent_bp":T.nonlinear(np.concatenate([X,bp],1),Y,fold,ep)}
        if name!="parent":
            q["blind"]=auxiliary(gate,path.with_name("latest.pt"))
        r["checkpoints"][name]=q
        (P/"BELIEF.json").write_text(json.dumps(r,indent=2));print(name,"complete",flush=True)
if __name__=="__main__":main()
