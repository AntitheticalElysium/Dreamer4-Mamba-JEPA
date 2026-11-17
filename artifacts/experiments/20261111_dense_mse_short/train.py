"""Two-arm, capped short continuation. Original assets read-only."""
import os
os.environ.setdefault("JAX_PLATFORMS","cpu")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF","expandable_segments:True")
import argparse,hashlib,json,time
from pathlib import Path
import numpy as np
import torch
from saccade import data as D,model as M,train as TR,chooser as C
from .loss import multistep_losses,LAMBDA_MULTI
P=Path(__file__).resolve().parent; E=P.parent
PARENT=E/"20261109_p1_hud_change/run/ckpt_10.pt"
GAZE=PARENT.parent/"gaze.bin"
STEPS=64
BUDGET=384
SEED=1

def atomic_write(path,value):
    t=path.with_suffix(".tmp");t.write_text(json.dumps(value,indent=2));t.replace(path)

def save(path,value):
    t=path.with_suffix(".tmp");torch.save(value,t);t.replace(path)

def fixture():
    cfg=torch.load(PARENT,mmap=True,map_location="cpu",weights_only=False)["cfg"]
    model=M.M0(cfg)
    ck=torch.load(PARENT,mmap=True,map_location="cpu",weights_only=False)
    model.load_state_dict(ck["model"])
    mask=.02*torch.randn(cfg["belief"]["width"],generator=torch.Generator().manual_seed(111206))
    model.belief.register_parameter("mask_token",torch.nn.Parameter(mask))
    model.denoiser.checkpoint=cfg["p1"]["activation_checkpointing"]
    return model,cfg

def replay_gaze(chunk,streams,trace):
    p=streams.prefix;f=chunk["fixation"][:,p:]
    ref=(f[...,0]-1)*7+f[...,1]-1
    assert np.array_equal(trace["target"],chunk["target"][:,p:].reshape(-1))
    assert np.array_equal(trace["episode"],chunk["episode"][:,p:].reshape(-1))
    assert np.array_equal(trace["uniform"],ref.reshape(-1))
    win=trace["window"].astype(int).reshape(f.shape[:-1])
    assert ((win>=0)&(win<49)).all()
    f[...,0]=1+win//7;f[...,1]=1+win%7
    for i,hist in enumerate(streams.hist):
        streams.hist[i]=[(*r[:3],*map(int,chunk["fixation"][i,-len(hist)+j])) for j,r in enumerate(hist)]

def run(mode,out,steps):
    assert os.environ.get("MAMBA_DETERMINISTIC")=="1"
    assert 0<steps<=BUDGET
    torch.set_num_threads(2);torch.manual_seed(SEED)
    dev=torch.device("cuda");model,cfg=fixture();model.to(dev).train()
    opt=torch.optim.AdamW(model.param_groups(),lr=2e-5,betas=tuple(cfg["optimizer"]["betas"]),eps=cfg["optimizer"]["eps"])
    eps=D.load_split(cfg)["train"];offs=D.frame_offsets(eps);p=cfg["data"]["context_prefix"]
    targets=np.load(D.targets_dir(cfg)/"change_targets.npy",mmap_mode="r");D.validate_change_targets(targets,eps,D.targets_dir(cfg))
    n=cfg["data"]["streams"];assert n==16
    stream=D.Streams(eps,n,p,seed=SEED,targets_offsets=offs)
    trace=np.memmap(GAZE,dtype=C.TRACE,mode="r")
    gen=torch.Generator().manual_seed(1001)
    c=model.belief.initial(n,dev);cache=torch.zeros(n,p+1,cfg["belief"]["width"],device=dev)
    out.mkdir(parents=True,exist_ok=True)
    ident={"mode":mode,"updates":steps,"seed":SEED,"parent":TR.file_sha256(PARENT),
           "gaze_prefix_sha256":hashlib.sha256(trace[:steps*n*STEPS].tobytes()).hexdigest(),
           "source":{f.name:TR.file_sha256(f) for f in sorted(P.glob("*.py"))},
           "data":TR.run_provenance(cfg)}
    latest=out/"latest.pt";start=0;log=out/"log.jsonl";identity=out/"identity.json"
    if identity.exists():
        assert json.loads(identity.read_text())==ident,"resume identity mismatch"
    else:
        assert not latest.exists()
        atomic_write(identity,ident)
    if latest.exists():
        s=torch.load(latest,map_location=dev,weights_only=False)
        model.load_state_dict(s["model"]);opt.load_state_dict(s["opt"]);stream.load_state(s["streams"])
        c,cache,start=s["carry"],s["cache"],s["step"]
        torch.set_rng_state(s["cpu"].cpu());torch.cuda.set_rng_state(s["cuda"].cpu());gen.set_state(s["gen"].cpu())
        with log.open("r+b") as f:f.truncate(s["log_bytes"])
    elif log.exists():
        assert log.stat().st_size==0,"partial log without a checkpoint"
    for u in range(start,steps):
        t=time.time();chunk=stream.next(STEPS);pos=u*n*STEPS
        replay_gaze(chunk,stream,trace[pos:pos+n*STEPS])
        boot=TR.bootstrap_rows(cfg,SEED,15625+u)
        for g in opt.param_groups:g["lr"]=2e-5*min(1,(u+1)/32)*g["lr_scale"]
        opt.zero_grad(set_to_none=True);read=[]
        for rows in np.array_split(np.arange(n),2):
            terms,cs,ss=multistep_losses(model,D.select(chunk,rows),M.carry_rows(c,rows),cache[rows],
                                          targets,cfg,boot[rows],15625+u,gen,mode)
            loss=(model.harmoniser(TR.groups(terms))+terms["noise"]+LAMBDA_MULTI*terms["multi"])/2
            assert torch.isfinite(loss)
            loss.backward()
            M.carry_put(c,rows,cs);cache[rows]=ss
            read.append({k:float(v.detach()) for k,v in terms.items()})
        norm=float(torch.nn.utils.clip_grad_norm_(model.parameters(),cfg["optimizer"]["grad_clip"],error_if_nonfinite=True))
        opt.step()
        rec=dict(update=u+1,seconds=time.time()-t,grad_norm=norm,peak_gib=torch.cuda.max_memory_allocated()/2**30,
                 **{k:float(np.mean([a[k] for a in read])) for k in read[0]})
        with log.open("a") as f:f.write(json.dumps(rec)+"\n")
        if u%16==0 or u+1==steps:print(mode,rec,flush=True)
        if (u+1)%64==0 or u+1==steps:
            save(latest,dict(model=model.state_dict(),cfg=cfg,opt=opt.state_dict(),streams=stream.state(),carry=c,cache=cache,step=u+1,
                             cpu=torch.get_rng_state(),cuda=torch.cuda.get_rng_state(),gen=gen.get_state(),log_bytes=log.stat().st_size))
    save(out/"eval.pt",{"model":{k:v for k,v in model.state_dict().items() if k!="belief.mask_token"},
                       "cfg":cfg,"world_seed":SEED,"state":{"frames":n*STEPS*steps,"update":steps},
                       "parent":ident["parent"],"training_sha256":TR.file_sha256(latest)})
    print(mode,"complete",steps,flush=True)
if __name__=="__main__":
    a=argparse.ArgumentParser()
    a.add_argument("mode",choices=("endpoint","dense"));a.add_argument("out",type=Path);a.add_argument("--steps",type=int,default=BUDGET)
    x=a.parse_args();run(x.mode,x.out,x.steps)
