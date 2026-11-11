"""Measure sub-digit phase response through the actual sampler. No model updates."""
import os
os.environ.setdefault("JAX_PLATFORMS","cpu")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF","expandable_segments:True")
import importlib,json
from pathlib import Path
import numpy as np
import torch
from saccade import data as D,model as M,train as TR
P=Path(__file__).resolve().parent; E=P.parent; dev=torch.device("cuda")
RUNS={"parent":E/"20261109_p1_hud_change/run/ckpt_10.pt"}
for size,folder in [(384,"short"),(2000,"2k")]:
    for arm in ("endpoint","dense"):
        RUNS[f"{arm}_{size}"]=E/f"20261111_dense_mse_{folder}/run/{arm}/eval.pt"

def old_data(name):
    if name=="parent":
        p=E/"20261111_two_arm_audit";stem="hud"
    else:
        stem,size=name.split("_")
        p=E/f"20261111_dense_mse_{'short' if size=='384' else '2k'}"
    return torch.load(p/f"phase_{stem}.pt",weights_only=False,map_location="cpu"),json.loads((p/f"PHASE_{stem}.json").read_text())

@torch.no_grad()
def main():
    torch.set_num_threads(2)
    IF=importlib.import_module("artifacts.experiments.20261030_actor_interface.interface")
    path=E/"20261108_full_frame_change/phase_bank.pt"
    bank=torch.load(path,weights_only=False,map_location="cpu")
    result={"bank_sha256":TR.file_sha256(path),"checkpoints":{}}
    K=32
    for name,ck in RUNS.items():
        model,cfg=IF.load(str(ck))
        saved,meta=old_data(name)
        sha=TR.file_sha256(ck)
        assert sha==meta["checkpoint_sha256"]
        rows=[]
        for i,r in enumerate(bank):
            n=len(r["prefix_frames"]);end=r["end"];T=n+end+1
            f=np.concatenate([np.broadcast_to(r["prefix_frames"],(2,*r["prefix_frames"].shape)),r["frames"][:,:end+1]],1)
            a=np.concatenate([np.broadcast_to(r["prefix_actions"],(2,n)),np.zeros((2,end+1),np.int64)],1)
            rw=np.concatenate([np.broadcast_to(np.r_[0.,r["prefix_reward"]],(2,n+1)),r["reward"][:,:end]],1)
            td=np.concatenate([np.broadcast_to(np.r_[False,r["prefix_term"]],(2,n+1)),r["term"][:,:end]],1)
            fx=D.fixation_prior(np.random.default_rng(12000+i),(T,))
            fix=torch.as_tensor(np.broadcast_to(fx,(2,*fx.shape)).copy(),device=dev)
            query=torch.as_tensor(D.fixation_prior(np.random.default_rng(15000+i),()),device=dev)
            act=torch.as_tensor(a,device=dev);prev=torch.cat([act.new_full((2,1),-1),act[:,:-1]],1)
            with TR.autocast(dev,cfg):
                p=model.perception(torch.as_tensor(f,device=dev),fix)
                y,_=model.belief(model.belief.tokens(model.belief.percept_embed(p),torch.as_tensor(rw,device=dev,dtype=torch.float32),torch.as_tensor(td,device=dev),act),model.belief.initial(2,dev))
                bp=torch.cat([torch.zeros_like(y[:,:1]),y[:,1::2][:,:-1]],1)
                truth=model.perception(torch.as_tensor(r["frames"][:,end+1],device=dev),query[None].expand(2,-1)).p[:,64].float()
                delta=truth[0]-truth[1]
                assert delta.norm()>.1 and r["food"][0,end+1]<r["food"][1,end+1]
                shape=[tuple(z.shape[2:]) for z in p.parts()]
                row={"e":r["e"],"steps":{}}
                for lag in (7,4,1,0):
                    t=T-1-lag
                    assert np.array_equal(f[0,t-7:t+1],f[1,t-7:t+1])
                    assert all(torch.equal(z[0,t-7:t+1],z[1,t-7:t+1]) for z in p.parts())
                    assert torch.equal(prev[0,t-7:t+1],prev[1,t-7:t+1]) and act[:,t].tolist()==[0,0]
                    caps=[];ys=[]
                    for br in (0,1):
                        ca=model.denoiser.new_cache();rg=torch.Generator(device=dev).manual_seed(14000+i)
                        for j in range(t-7,t+1):
                            ca=M.context_pass(model.denoiser,tuple(z[br:br+1,j:j+1] for z in p.parts()),prev[br:br+1,j:j+1],
                                model.belief.memory(bp[br:br+1,j:j+1]),fix[br:br+1,j:j+1],ca,cfg,rg)
                        ca={"t":ca["t"],"kv":[tuple(z.expand(K,*z.shape[1:]) for z in kv) for kv in ca["kv"]]}
                        orig=model.denoiser.step;guesses=[]
                        def capture(*args,**kwargs):
                            z=orig(*args,**kwargs)
                            if len(guesses)<cfg["edm"]["sampler"]["steps"]:
                                guesses.append(z[0][0][:,0,64].float().clamp(-3,3))
                            return z
                        model.denoiser.step=capture
                        try:
                            xs=M.sample(model.denoiser,shape,act.new_zeros(K,1),
                                model.belief.memory(y[br:br+1,2*t+1])[:,None].expand(K,-1,-1,-1),
                                query[None,None].expand(K,1,-1),ca,cfg,torch.Generator(device=dev).manual_seed(16000+i))
                        finally:model.denoiser.step=orig
                        caps.append(torch.stack(guesses));ys.append(xs[0][:,0,64].float())
                        if lag==0:
                            assert saved[i]["e"]==r["e"]
                            assert torch.equal(xs[0][:,0].float().cpu(),saved[i]["gen"][br][0]),(name,i,br)
                    proj=lambda z:((z-truth[1])*delta).sum(-1)/delta.square().sum()
                    z0,z1=map(proj,caps);v0,v1=map(proj,ys)
                    row["steps"][str(lag+1)]={"paired_first":(z0[0]-z1[0]).cpu().tolist(),
                          "paired_final":(v0-v1).cpu().tolist(),"mean_first":[float(z0[0].mean()),float(z1[0].mean())],
                          "mean_final":[float(v0.mean()),float(v1.mean())],
                          "mean_difference_per_field_call":(z0-z1).mean(-1).cpu().tolist(),
                          "final_difference_rms_in_digit_steps":float((ys[0]-ys[1]).square().sum(-1).mean().sqrt()/delta.norm())}
            rows.append(row)
        result["checkpoints"][name]={"sha256":sha,"rows":rows,"due_fork_bitwise_equal":True}
        (P/"NATIVE.json").write_text(json.dumps(result,indent=2));print(name,"complete",flush=True)
if __name__=="__main__":main()
