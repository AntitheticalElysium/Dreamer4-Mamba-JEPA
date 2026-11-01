"""Independent numerical checks of implementation contracts, no learning claims."""
import copy
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch
import yaml

from saccade import data as D, model as M, train as TR
from saccade.eval import s1

OUT = Path(__file__).parent


def target_projection(cfg):
    torch.manual_seed(1)
    per = M.Perception(cfg).eval()
    ep = D.load_split(cfg)["dev"][0]
    fr = D.RP.episode_fields(ep.shard, ep.slot)["observations"]
    ids = np.linspace(0, ep.T, min(ep.T+1, 32), dtype=int)
    frames = fr[ids]
    fix = torch.full((len(ids),2),4,dtype=torch.long)
    with torch.no_grad():
        p = per(frames,fix)
        out = {}
        for key,x in zip(("p","g","v"),p.parts()):
            mapped=M.out_map(x,3.)
            delta=(mapped-x).float()
            out[key]={"rms_delta":float(delta.square().mean().sqrt()),
                      "max_abs_delta":float(delta.abs().max()),
                      "target_rms":float(x.square().mean().sqrt()),
                      "mapped_rms":float(mapped.square().mean().sqrt())}
    # Zero raw EDMLoss with an oracle target does not imply correct deployed D.
    x0=p.p[:1,None]
    sigma=torch.tensor([[.002]])
    cs,co,ci,_=M.precondition(sigma)
    noise=torch.randn_like(x0)
    x=x0+M._bcast(sigma,x0)*noise
    perfect_F=(x0-M._bcast(cs,x0)*x)/M._bcast(co,x0)
    raw_D=M._bcast(cs,x)*x+M._bcast(co,x)*perfect_F
    out["perfect_conditional_denoiser"]={"raw_F_training_mse":0.,
          "clean_D_max_error_before_output_map":float((raw_D-x0).abs().max()),
          "clean_D_max_error_after_output_map":float((M.out_map(raw_D,3)-x0).abs().max())}
    targets=tuple(t[:1,None] for t in p.parts())
    class Perfect:
        def step(self,scaled,cn,*args):
            sig=(4*cn).exp();cs,co,ci,_=M.precondition(sig)
            return tuple((target-M._bcast(cs,z)*z/M._bcast(ci,z))/M._bcast(co,z) for z,target in zip(scaled,targets)),[]
    args=(Perfect(),[tuple(t.shape[2:]) for t in targets],None,torch.zeros(1,1,1),None,None,cfg)
    with torch.no_grad():
        sampled=M.sample(*args,gen=torch.Generator().manual_seed(19))
        original=M.out_map
        try:
            M.out_map=lambda z,s:z.clamp(-s,s)
            clamped=M.sample(*args,gen=torch.Generator().manual_seed(19))
        finally:M.out_map=original
    out['full_zero_error_oracle_sampler']={
        'max_abs_error_current_output_map':max(float((a-b).abs().max()) for a,b in zip(sampled,targets)),
        'max_abs_error_idempotent_range_clamp':max(float((a-b).abs().max()) for a,b in zip(clamped,targets)),
        'scope':'Zero-loss contract check only; does not measure learned or actor performance.'}
    return out


def ap_ties():
    a=np.array([1,1,0,0]); b=a[::-1].copy();s=np.full(4,.5)
    return {"expected_constant_predictor_AP":.5,"positives_first":s1.ap(a,s),"negatives_first":s1.ap(b,s)}


def omitted_history():
    torch.manual_seed(4)
    r={"p":torch.randn(10,81,16),"g":torch.randn(10,9,64),"v":torch.randn(10,8),
       "fix":torch.full((10,2),4),"action":torch.zeros(10,dtype=torch.long),
       "reward":torch.zeros(10),"terminated":torch.zeros(10,dtype=torch.bool),
       "b":torch.randn(10,256),"bp":torch.randn(10,256)}
    nxt={k:r[k][9:10] for k in ("p","g","v")}
    nxt.update(r=torch.zeros(1),term=torch.zeros(1))
    a=s1.tuple_features([r],np.array([0]),np.array([9]),nxt)
    alt=copy.deepcopy(r)
    alt["p"][:8]+=10
    alt["g"]+=10
    alt["reward"][:9]=1
    alt["b"]+=100
    alt["bp"]+=100
    b=s1.tuple_features([alt],np.array([0]),np.array([9]),nxt)
    return {"C2ST_features":a.shape[1],"max_delta_changing_older_history_glimpses_rewards_belief":float(abs(a-b).max())}


def duplicate_yaml():
    tree=yaml.compose(Path("saccade/m0.yaml").read_text())
    dup=[]
    def walk(node,path):
        if isinstance(node,yaml.MappingNode):
            seen={}
            for key,value in node.value:
                if key.value in seen:
                    dup.append({"path":".".join(path+[key.value]),"first_line":seen[key.value]+1,"second_line":key.start_mark.line+1})
                seen[key.value]=key.start_mark.line
                walk(value,path+[key.value])
    walk(tree,[])
    return dup


def adam_scale():
    # Global norm clipping is a scalar on all gradients. Adam is approximately
    # invariant to a fixed scale, so clip factor != parameter-update factor.
    result=[]
    for k in (1.,.001):
        p=torch.nn.Parameter(torch.tensor([1.],dtype=torch.float64))
        opt=torch.optim.AdamW([p],lr=1e-4,weight_decay=0,eps=1e-8)
        p.grad=torch.tensor([k],dtype=torch.float64)
        opt.step()
        result.append(float(1-p.detach()))
    return {"update_at_gradient1":result[0],"update_at_gradient0001":result[1],
            "update_ratio":result[1]/result[0],"scope":"Constant scaling sanity check, not reproduction of historical drifting gradients."}


def hashes():
    names=["saccade/data.py","saccade/model.py","saccade/train.py","saccade/eval/s1.py","saccade/tests/test_m0.py","saccade/m0.yaml",
           "artifacts/saccade_m0/targets/change_teacher.txt","artifacts/saccade_m0/targets/change_targets.npy"]
    return {f:hashlib.sha256(Path(f).read_bytes()).hexdigest() for f in names}


def main():
    torch.set_num_threads(6)
    cfg=TR.load_cfg()
    r={"target_projection":target_projection(cfg),"AP_ties":ap_ties(),"C2ST_history":omitted_history(),
       "duplicate_yaml":duplicate_yaml(),"Adam_scale":adam_scale(),"sha256":hashes()}
    (OUT/"cpu_contract_results.json").write_text(json.dumps(r,indent=2)+'\n')
    print(json.dumps({k:v for k,v in r.items() if k!='sha256'},indent=2))


if __name__=='__main__': main()
