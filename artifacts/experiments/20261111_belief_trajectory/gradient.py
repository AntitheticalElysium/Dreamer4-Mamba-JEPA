"""Frozen loss-routing audit; paired original and rebased teacher only, no optimizer steps."""
import os
os.environ.setdefault("JAX_PLATFORMS","cpu")
import importlib,json
from pathlib import Path
import numpy as np
import torch
from saccade import data as D, model as M, train as TR
HERE=Path(__file__).resolve().parent
E=HERE.parent/"20261111_p1_multistep_mamba"
T=importlib.import_module("artifacts.experiments.20261111_p1_multistep_mamba.trainer")
MM=T.MM
DEV=torch.device("cuda")

@torch.no_grad()
def prefix(model,chunk,c,cache,cfg):
    P=cfg["data"]["context_prefix"];q=TR.tensors(chunk,DEV);f=q["first"]
    with TR.autocast(DEV,cfg):
        pc=model.perception(q["frames"][:,P:],q["fixation"][:,P:])
        y,carry=model.belief(model.belief.tokens(model.belief.percept_embed(pc),q["reward"][:,P:],q["terminated"][:,P:],q["action"][:,P:]),
                            c,f[:,P:].long().cumsum(1).repeat_interleave(2,1))
        nxt=y[:,1::2]
        bp=torch.cat([cache.to(nxt.dtype),nxt[:,:-1]],1)*(~f)[...,None]
        cache=torch.cat([bp[:,-P:],nxt[:,-1:]],1).detach().float()
    return carry,cache

def main():
    torch.set_num_threads(2)
    cfg=TR.load_cfg()
    eps=D.load_split(cfg)["train"]
    offsets=D.frame_offsets(eps)
    targets=np.load(D.targets_dir(cfg)/"change_targets.npy",mmap_mode="r")
    P=cfg["data"]["context_prefix"];N=2
    out={}
    for ci in (1,5,10):
        file=E/"run"/("ckpt_10.orig_with_mask.pt" if ci==10 else f"ckpt_{ci:02d}.pt")
        ck=torch.load(file,weights_only=False,map_location="cpu",mmap=True)
        cfg=ck["cfg"]
        model=MM.MultiStepM0(cfg).to(DEV).train()
        model.load_state_dict(ck["model"]);model.denoiser.checkpoint=True
        ps=[p for n,p in model.belief.named_parameters() if n!="mask_token"]
        ratios=[]
        s=D.Streams(eps,N,P,seed=111104,targets_offsets=offsets)
        c=model.belief.initial(N,DEV);cache=torch.zeros(N,P+1,256,device=DEV)
        for j in range(6):
            steps=64 if j<3 else 256
            chunk=s.next(steps)
            if j==0:
                c,cache=prefix(model,chunk,c,cache,cfg)
                chunk=s.next(steps)
            boot=TR.bootstrap_rows(cfg,1,ck["state"]["update"]+j)[[0,1]]
            seed=42110+j
            gr=torch.Generator().manual_seed(seed)
            H=T.HORIZONS[int(torch.randint(0,4,(),generator=gr))]
            st=int(torch.randint(0,steps-H,(),generator=gr))
            first=chunk["first"][:,P:]
            r={"T":steps,"H":H,"t":st,"bad":first[:,:st+1].any(1).tolist(),
               "valid":(~first[:,st:st+H+1].any(1)).tolist(),"boot":boot.tolist()}
            for tag in ("original","rebased"):
                f=model.belief.forward;calls=0
                def wrapped(x,c,seq_idx=None,backend=None):
                    nonlocal calls
                    calls+=1
                    if tag=="rebased" and calls==2:
                        seq_idx=seq_idx-seq_idx[:,:1]
                    return f(x,c,seq_idx,backend)
                with torch.random.fork_rng(devices=[0]):
                    torch.manual_seed(seed)
                    model.belief.forward=wrapped
                    try:
                        terms,_,_=T.multistep_losses(model,chunk,c,cache,targets,cfg,boot,ck["state"]["update"],torch.Generator().manual_seed(seed))
                    finally:model.belief.forward=f
                    sig=model.harmoniser.log_sigma.detach().exp()
                    losses={k:v/sig[i] for i,(k,v) in enumerate(TR.groups(terms).items())}
                    losses["multi"]=T.LAMBDA_MULTI*terms["multi"]
                    V={}
                    for k,l in losses.items():
                        g=torch.autograd.grad(l,ps,retain_graph=True,allow_unused=True) if l.requires_grad else [None]*len(ps)
                        V[k]=torch.cat([(torch.zeros_like(p) if z is None else z).float().flatten() for z,p in zip(g,ps)])
                    V["base"]=V["G1"]+V["G2"]+V["G3"]
                    norms={k:float(v.norm()) for k,v in V.items()}
                    r[tag]={"loss":{k:float(v.detach()) for k,v in losses.items()},"shared_belief_norm":norms,
                            "aux_to_base_norm":norms["multi"]/max(norms["base"],1e-12),
                            "aux_g1_cos":float(torch.nn.functional.cosine_similarity(V["multi"],V["G1"],dim=0))}
                    del terms,losses,V,g
            c,cache=prefix(model,chunk,c,cache,cfg)
            ratios.append(r)
        out[str(ci)]={"checkpoint_sha256":TR.file_sha256(file),"chunks":ratios}
        (HERE/"GRADIENT.json").write_text(json.dumps(out,indent=2))
        print(ci,[(x["bad"],x["original"]["aux_to_base_norm"],x["rebased"]["aux_to_base_norm"]) for x in ratios],flush=True)
if __name__=="__main__":main()
