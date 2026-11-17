"""Rebase and update-one matching checks for the capped experiment."""
import os
os.environ.setdefault("JAX_PLATFORMS","cpu")
import json,importlib
import torch
from saccade import model as M,train as TR
from .train import P,fixture,atomic_write
@torch.no_grad()
def main():
    torch.set_num_threads(2)
    a,b=([json.loads(l) for l in (P/"smoke"/k/"log.jsonl").read_text().splitlines()] for k in ("endpoint","dense"))
    terms=("g1","change","tsp","rec","reward","cont","noise","endpoint_diag","dense_diag","horizon","valid_rollout")
    assert len(a)==len(b)==2 and all(a[0][k]==b[0][k] for k in terms)
    assert a[0]["multi"]==a[0]["endpoint_diag"] and b[0]["multi"]==b[0]["dense_diag"] and a[0]["multi"]!=b[0]["multi"]
    model,cfg=fixture();model.cuda().eval()
    bel=model.belief
    x=torch.randn(3,48,256,generator=torch.Generator().manual_seed(31)).cuda()
    seq=torch.zeros(3,48,device="cuda",dtype=torch.long)
    seq[0,8:]=1;seq[1,32:]=1;seq[2,6:]=1;seq[2,16:]=2
    with TR.autocast(x.device,cfg):
        full,_=bel(x,bel.initial(3,"cuda"),seq)
        _,c=bel(x[:,:25],bel.initial(3,"cuda"),seq[:,:25])
        bad,_=bel(x[:,25:],c,seq[:,25:])
        fix,_=bel(x[:,25:],c,seq[:,25:]-seq[:,24:25])
    err=lambda z:(z.float()-full[:,25:].float()).square().mean((1,2)).sqrt().tolist()
    assert max(err(fix))<.02 and err(bad)[0]>5*err(fix)[0] and err(bad)[2]>5*err(fix)[2]
    for arm in ("endpoint","dense"):
        z=torch.load(P/"smoke"/arm/"eval.pt",map_location="cpu",weights_only=False)
        M.M0(z["cfg"]).load_state_dict(z["model"],strict=True)
    out=dict(passed=True,first_update_other_terms_bitwise_equal=True,endpoint_selected=a[0]["multi"],
             dense_selected=b[0]["multi"],bad_fork_rms=err(bad),rebased_fork_rms=err(fix),
             smoke_peak_gib=max(r["peak_gib"] for r in a+b))
    atomic_write(P/"CHECKS.json",out);print(out,flush=True)
if __name__=="__main__":main()
