"""CPU LayerNorm control for conditional_stat and conditional_dopt.

Preserves previous unconstrained evidence. Applies the world's exact FP32
per-token LayerNorm (eps=1e-5) to empirical statistics before frozen readout.
This is a feasible normalized candidate, NOT the constrained D-loss optimum.
Reports all repeated-layout groups and the 69 nonidentical-target groups
separately, with branch weights and paired root-cluster intervals.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from conditional_dopt import solve

ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/"artifacts/eda/oracle_v1"
MAPC=[i for i in range(63) if i!=31]
NAMES=("zombie","skeleton","cow","arrow")
MODES=("real","mean","median","dopt","real_ln","mean_ln","median_ln","dopt_ln")


def main():
    torch.set_num_threads(4)
    meta=torch.load(ROOT/"artifacts/eda/diagnosis_rollouts_v1/meta.pt",weights_only=False)
    saved=torch.load(OUT/"snap_bank_readers.pt",weights_only=False)
    selection=json.loads((OUT/"conditional_stat.json").read_text())
    h=saved["heads"]["mob"]
    threshold=torch.tensor(saved["thresholds"])
    R=len(meta["seed"])
    fut=np.memmap(ROOT/"artifacts/eda/stochdiag_v1/fut5.f16",dtype=np.float16,mode="r",shape=(R,5,16,81,192))
    rows={s:torch.zeros(R,len(MODES),4,4,dtype=torch.float64) for s in ("all","nonidentical_targets")}
    variances={s:{m:{reg:[] for reg in ("all81","map62","mob_cells","hud18")} for m in MODES} for s in rows}
    group_counts={s:{"groups":0,"roots":set(),"branches":0,"true_mob_branch_cells":[0]*4} for s in rows}
    diagnoses=[]

    def decode(x):
        c=x[:,MAPC].double()
        z=(c-h["mu"].double())/h["sd"].double()
        z=torch.cat((z,torch.ones(*z.shape[:-1],1,dtype=z.dtype)),-1)
        return z@h["w"].double()>threshold

    for g in selection["groups"]:
        r,ss=g["root"],g["members"]
        Y=torch.from_numpy(np.array(fut[r,ss,0])).float()
        nonidentical=not bool((Y==Y[:1]).all())
        xd,diag=solve(Y.double().numpy().reshape(len(ss),-1))
        assert diag["success"],diag
        X={"real":Y,"mean":Y.mean(0,keepdim=True),"median":Y.median(0,keepdim=True).values,
           "dopt":torch.from_numpy(xd.reshape(1,81,192)).float()}
        for m in MODES[:4]:X[m+"_ln"]=F.layer_norm(X[m],(192,),eps=1e-5)
        v=meta["future_visible"][r,ss[0],0].float()
        flags=v[1071:1512].reshape(63,7)>0
        truth=torch.stack((flags[:,0],flags[:,2],flags[:,1],flags[:,3:].any(-1)),-1)[MAPC]
        subset=["all"]+(["nonidentical_targets"] if nonidentical else [])
        for s in subset:
            group_counts[s]["groups"]+=1
            group_counts[s]["roots"].add(r)
            group_counts[s]["branches"]+=len(ss)
            group_counts[s]["true_mob_branch_cells"]=[a+int(b)*len(ss) for a,b in zip(group_counts[s]["true_mob_branch_cells"],truth.sum(0))]
        for m,mode in enumerate(MODES):
            dec=decode(X[mode])
            lab=truth[None].expand(len(dec),-1,-1)
            weight=1 if mode.startswith("real") else len(ss)
            cnt=torch.stack((lab.sum((0,1)),dec.sum((0,1)),(lab&dec).sum((0,1)),(~lab&dec).sum((0,1))),-1)*weight
            var=X[mode].var(-1,unbiased=False)
            values={"all81":var.flatten(),"map62":var[:,MAPC].flatten(),
                    "mob_cells":var[:,MAPC][:,truth.any(-1)].flatten(),"hud18":var[:,63:].flatten()}
            for s in subset:
                rows[s][r,m]+=cnt
                for reg,vv in values.items():variances[s][mode][reg].append(vv.repeat(weight))
        diagnoses.append({"root":r,"members":ss,"nonidentical_targets":nonidentical,
                          "dopt_success":diag["success"],"dopt_mean_readout_identical":bool((decode(X["dopt"])==decode(X["mean"])).all())})
    def metric(t):
        den=t[...,0].clamp_min(1)
        return torch.stack((t[...,1]/den,t[...,2]/den,t[...,3]/den),-1)
    report={"schema":"oracle_conditional_norm_v1","epsilon":1e-5,"dtype":"float32","subsets":{},
            "dopt_mean_readout_identical_groups":sum(q["dopt_mean_readout_identical"] for q in diagnoses),
            "limits":["Applying LayerNorm to the unconstrained optimum is not solving the LayerNorm-constrained optimum.",
                      "Conditioned groups have 2-5 observed futures, one-step only; singleton outcomes excluded."],
            "script_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    for s,rr in rows.items():
        totals=rr.sum(0);met=metric(totals)
        metrics={mode:{name:{"true":int(totals[m,j,0]),"predicted":int(totals[m,j,1]),"true_positive":int(totals[m,j,2]),
                             "false_positive":int(totals[m,j,3]),"census_ratio":float(met[m,j,0]),
                             "recall":float(met[m,j,1]),"fp_per_true":float(met[m,j,2])}
                       for j,name in enumerate(NAMES)} for m,mode in enumerate(MODES)}
        ids=torch.tensor(sorted(group_counts[s]["roots"]))
        rng=torch.Generator().manual_seed(20261012)
        boot=[]
        for _ in range(16):
            idx=torch.randint(len(ids),(250,len(ids)),generator=rng)
            boot.append(metric(rr[ids][idx].sum(1)))
        boot=torch.cat(boot)
        contrasts={}
        for m in range(4,8):
            for ref in (0,m-4):
                delta=met[m]-met[ref]
                ci=torch.quantile(boot[:,m]-boot[:,ref],torch.tensor([.025,.975],dtype=torch.float64),dim=0)
                contrasts[MODES[m]+"_minus_"+MODES[ref]]={name:{measure:{"delta":float(delta[j,k]),"ci95":[float(ci[0,j,k]),float(ci[1,j,k])]}
                                                       for k,measure in enumerate(("census_ratio","recall","fp_per_true"))}
                                                   for j,name in enumerate(NAMES)}
        vs={}
        for mode in MODES:
            vs[mode]={}
            for reg,vals in variances[s][mode].items():
                vv=torch.cat(vals)
                vs[mode][reg]={"n":len(vv),"mean":float(vv.mean()),"q05":float(vv.quantile(.05)),
                               "median":float(vv.median()),"q95":float(vv.quantile(.95)),"min":float(vv.min())}
        coverage={k:len(v) if isinstance(v,set) else v for k,v in group_counts[s].items()}
        report["subsets"][s]={"coverage":coverage,"metrics":metrics,"contrasts":contrasts,"token_variance":vs}
    path=OUT/"conditional_norm.json"
    tmp=path.with_suffix('.json.tmp');tmp.write_text(json.dumps(report,indent=2));tmp.replace(path)
    torch.save({"rows":rows,"group_diagnostics":diagnoses},OUT/"conditional_norm_rows.pt")
    print(json.dumps({s:{"coverage":v["coverage"],"metrics":v["metrics"],"dopt_variance":v["token_variance"]["dopt"],
                        "dopt_ln_variance":v["token_variance"]["dopt_ln"]} for s,v in report["subsets"].items()},indent=2))


if __name__=="__main__":main()
