"""Frozen one-step samples, S1 reproduction, coupled sampler experiments and observed-target denoising."""
import math
import time

from .common import *
from saccade.eval.probe import KEEP


@torch.no_grad()
def read(reader, p, v):
    result = []
    for ids in torch.arange(len(p)).split(512):
        x = torch.cat([p[ids, :63].reshape(-1, 16).half(), v[ids].repeat_interleave(63, 0).half()], 1).cuda()
        result.append(reader(x)["kind"].argmax(-1).reshape(len(ids), 63).cpu())
    return torch.cat(result)


def counts(pred):
    return {name: float((pred[:, KEEP] == k).sum(1).float().mean()) for k, name in enumerate(KINDS[1:], 1)}


def geometry(parts):
    out = {}
    for key, x in parts.items():
        if key not in ("p", "g", "v"):
            continue
        z = x.float().reshape(-1, x.shape[-1])
        norms = z.square().mean(-1).sqrt()
        out[key] = {"rms_mean": float(norms.mean()), "rms_quantiles": torch.quantile(norms, torch.tensor([.01,.1,.5,.9,.99])).tolist(),
                    "token_mean_abs": float(z.mean(-1).abs().mean()), "max_abs": float(z.abs().max()),
                    "fraction_at_clamp": float((z.abs() >= 2.99999).float().mean())}
    return out


def prepare(model, enc, eps, ts, cfg):
    P = cfg["data"]["context_prefix"]
    pad = lambda a: (torch.ones_like if a.dtype == torch.long and a.ndim == 2 else torch.zeros_like)(a[:1])
    def win(key):
        return torch.stack([torch.cat([pad(enc[e][key]).expand(max(0, P-t), *enc[e][key].shape[1:]),
                                      enc[e][key][max(0,t-P):t+1]]) for e,t in zip(eps,ts)]).cuda()
    parts = [win(k).float() for k in ("p","g","v")]
    valid = torch.stack([torch.as_tensor(np.r_[np.zeros(max(0,P-t),bool),np.ones(min(P,t)+1,bool)]) for t in ts]).cuda()
    act, fix, bp = win("prev_action"), win("fix"), win("bp").float()
    return parts, act, fix, model.belief.memory(bp), torch.zeros_like(act), valid


@torch.no_grad()
def sample_prepared(model, prep, cfg, steps, churn, context_noise, initial_noise, churn_noise):
    target, act, fix, memory, episode, valid = prep
    e = cfg["edm"]; s = e["sampler"]; B,T = act.shape
    sig = [(s["sigma_max"]**(1/s["rho"]) + i/(steps-1)*(s["sigma_min"]**(1/s["rho"])-s["sigma_max"]**(1/s["rho"])))**s["rho"]
           for i in range(steps)] + [0.]
    context = [x[:,:-1] + e["inference_context_sigma"]*n for x,n in zip(target,context_noise)]
    x = [sig[0]*n for n in initial_noise]
    for k in range(steps):
        gamma = min(churn/steps,math.sqrt(2)-1)
        hat = sig[k]*(1+gamma)
        if gamma:
            x = [a + math.sqrt(hat**2-sig[k]**2)*n for a,n in zip(x,churn_noise[k])]
        sg = torch.full((B,T),e["inference_context_sigma"],device="cuda"); sg[:,-1]=hat
        cs,co,ci,cn = M.precondition(sg,e["sigma_data"])
        seq = [torch.cat([cx,a],1) for cx,a in zip(context,x)]
        with TR.autocast(torch.device("cuda"),cfg):
            Fo = model.denoiser(tuple(M._bcast(ci,a)*a for a in seq),cn,act,memory,fix,episode,valid)
        clean = M.clean_estimate(x,tuple(f[:,-1:] for f in Fo),cs[:,-1:],co[:,-1:],3.)
        x = [a+(sig[k+1]-hat)*(a-d)/hat for a,d in zip(x,clean)]
    return {k:a[:,0].float().cpu() for k,a in zip(("p","g","v"),x)}


if __name__ == "__main__":
    before=fingerprints(); model,cfg=load_model(); h=model_hash(model); reader=s1.load_reader(READER,CKPT,"cuda")
    enc32=torch.load(OUT/"real_dev.pt",map_location="cpu",weights_only=False,mmap=True)
    enc=half_encoding(enc32)
    start=time.time()
    eps,ts,gen=s1.one_step(model,enc,cfg,30000,np.random.default_rng(0))
    real={k:torch.stack([enc[e][k][t] for e,t in zip(eps,ts)]).float() for k in ("p","g","v")}
    kr,kg=read(reader,real["p"],real["v"]),read(reader,gen["p"],gen["v"])
    baseline={"real_counts":counts(kr),"generated_counts":counts(kg),"real_geometry":geometry(real),"generated_geometry":geometry(gen)}
    saved=json.loads((RUN/"s1/results.json").read_text())["s1_3"]["reader_counts"]
    errors={k:{d:abs(baseline[f"{d}_counts"][k]-saved[k][d]["est"]) for d in ("real","generated")} for k in KINDS[1:]}
    assert max(v for e in errors.values() for v in e.values()) < 1e-3,errors
    baseline["s1_count_reproduction_error"]=errors
    save("generation_baseline.json",baseline)
    torch.save({"eps":eps,"ts":ts,"real":real,"generated":gen,"real_reader":kr,"generated_reader":kg},OUT/"baseline_samples.pt")
    print("baseline S1 reproduced",baseline["generated_counts"],flush=True)

    L=labels("dev"); off=np.r_[0,np.cumsum(L["ep_T"]+1)]
    grid=np.concatenate([ep["grid"] for ep in grids(L)]).reshape(-1,63)
    past_idx=off[eps]+ts-1
    rng=np.random.default_rng(423)
    population=np.linspace(0,29999,1024,dtype=int)
    cohorts={}
    for k,name in enumerate(KINDS[1:],1):
        for night in (False,True):
            eligible=np.flatnonzero((grid[past_idx]==k).any(1)&((L["s_light"][past_idx]<.5)==night))
            cohorts[f"past_{name}_{'night' if night else 'day'}"]=np.sort(rng.choice(eligible,min(128,len(eligible)),replace=False))
    panel=np.unique(np.r_[population,*cohorts.values()])
    save("generation_panel.json",{"population_indices":population.tolist(),"cohorts":{k:v.tolist() for k,v in cohorts.items()},
         "panel_indices":panel.tolist(),"selection":"previous-frame entities only; no successor-label selection"})
    arms={"production_3_churn1":(3,1.),"ode_3":(3,0.),"ode_8":(8,0.),"ode_32":(32,0.)}
    all_samples={name:[] for name in arms}; noiseds={s:[] for s in (.1,.5,1.,3.,20/3)}
    chunks=np.array_split(panel,max(1,len(panel)//64))
    for rep in range(4):
        records={name:{k:[] for k in ("p","g","v")} for name in arms}
        for batch,ids in enumerate(chunks):
            prep=prepare(model,enc,eps[ids],ts[ids],cfg)
            target=prep[0]; rg=torch.Generator(device="cuda").manual_seed(9100+rep*1000+batch)
            context_noise=[torch.randn(a[:,:-1].shape,device="cuda",generator=rg) for a in target]
            initial_noise=[torch.randn(a[:,-1:].shape,device="cuda",generator=rg) for a in target]
            churn_noise=[[torch.randn(a[:,-1:].shape,device="cuda",generator=rg) for a in target] for _ in range(3)]
            for name,(steps,churn) in arms.items():
                sample=sample_prepared(model,prep,cfg,steps,churn,context_noise,initial_noise,churn_noise)
                for key in sample:records[name][key].append(sample[key])
            if rep==0:
                # Observed-target denoising uses the full-precision encoder target, with a shared epsilon across sigma.
                true=[torch.stack([enc32[e][k][t] for e,t in zip(eps[ids],ts[ids])]).cuda()[:,None] for k in ("p","g","v")]
                _,act,fix,memory,episode,valid=prep
                for sigma in noiseds:
                    noisy=[a+sigma*n for a,n in zip(true,initial_noise)]
                    context=[a[:,:-1]+cfg["edm"]["inference_context_sigma"]*n for a,n in zip(target,context_noise)]
                    seq=[torch.cat([cx,a],1) for cx,a in zip(context,noisy)]
                    sg=torch.full(act.shape,cfg["edm"]["inference_context_sigma"],device="cuda");sg[:,-1]=sigma
                    cs,co,ci,cn=M.precondition(sg)
                    with torch.no_grad(),TR.autocast(torch.device("cuda"),cfg):
                        Fo=model.denoiser(tuple(M._bcast(ci,a)*a for a in seq),cn,act,memory,fix,episode,valid)
                    Dn=M.clean_estimate(noisy,tuple(f[:,-1:] for f in Fo),cs[:,-1:],co[:,-1:],3.)
                    raw_se=[((f[:,-1:].float()-(x-M._bcast(cs[:,-1:],x)*z)/M._bcast(co[:,-1:],x))**2).cpu()
                            for f,x,z in zip(Fo,true,noisy)]
                    noiseds[sigma].append({"noisy":{k:v[:,0].cpu() for k,v in zip(("p","g","v"),noisy)},
                                           "denoised":{k:v[:,0].cpu() for k,v in zip(("p","g","v"),Dn)},
                                           "raw_se":{k:v[:,0] for k,v in zip(("p","g","v"),raw_se)}})
            if batch%8==0:print("sampler panel",rep,batch,len(chunks),flush=True)
        for name,parts in records.items():all_samples[name].append({k:torch.cat(v) for k,v in parts.items()})
    torch.save({"panel":panel,"eps":eps[panel],"ts":ts[panel],"arms":all_samples,"observed_target_denoising":noiseds},OUT/"sampler_panel.pt")
    summaries={}
    for name,reps in all_samples.items():
        summaries[name]={"all_panel":[],"population":[],"geometry":[]}
        pop=np.isin(panel,population)
        for sample in reps:
            pred=read(reader,sample["p"],sample["v"])
            summaries[name]["all_panel"].append(counts(pred))
            summaries[name]["population"].append(counts(pred[pop]))
            summaries[name]["geometry"].append(geometry(sample))
    save("sampler_summaries.json",summaries)
    assert model_hash(model)==h and fingerprints()==before
    save("generation_run.json",{"seconds":time.time()-start,"panel_histories":len(panel),"draws_per_history":4,
         "frozen_parameter_hash":h,"checkpoint_and_sources_unchanged":True,"peak_gpu_gib":torch.cuda.max_memory_allocated()/2**30})
