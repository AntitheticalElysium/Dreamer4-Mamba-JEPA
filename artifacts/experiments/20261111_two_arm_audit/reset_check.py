"""Exact stream replay and GPU continuation sham. No optimizer steps."""
import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")
import copy
import json
import sys
from pathlib import Path
import numpy as np
import torch
from saccade import data as D, model as M, train as TR

HERE = Path(__file__).resolve().parent
CK = HERE.parent / "20261111_p1_multistep_mamba/run/ckpt_10.pt"

def save(name, r):
    (HERE / (name + ".json")).write_text(json.dumps(r, indent=2))
    print(json.dumps(r, indent=2), flush=True)

def census():
    ck = torch.load(CK, map_location="cpu", mmap=True, weights_only=False)
    cfg = ck["cfg"]
    eps = D.load_split(cfg)["train"]
    d = cfg["data"]
    n = d["streams"]
    s = D.Streams(eps, n, d["context_prefix"], seed=1)
    sham = D.Streams(eps, n, d["context_prefix"], seed=1)
    gen = torch.Generator().manual_seed(1001)

    def draw(S):
        first = np.zeros((n,S), bool)
        for i in range(n):
            j = 0
            while j < S:
                k, uid = s.cur[i]
                t = s.t[i]
                z = min(S-j, eps[k].T + 1 - t)
                first[i,j] = t == 0
                D.fixation_prior(s.rng, (z,))
                j += z
                s.t[i] += z
                if s.t[i] == eps[k].T + 1:
                    s.cur[i], s.t[i] = s._pull(), 0
        return first

    result = {}
    frames = 0
    for u in range(ck["state"]["update"]):
        S = 64 if frames < cfg["p1"]["frames"]/2 else 256
        a = draw(S)
        if u < 4:
            b = sham.next(S)
            assert np.array_equal(a, b["first"][:,d["context_prefix"]:])
            for k in ("queue", "cur", "t", "uid", "rng"):
                assert s.state()[k] == sham.state()[k]
        micro = d["accumulate_at_64" if S==64 else "accumulate_at_256"]
        z = result.setdefault(str(S), dict(rows=0, split_after_reset=0, frames=0, suffix_until_reset=0,
                                         valid_student_rows=0, corrupted_teacher_endpoints=0))
        for rows in np.array_split(np.arange(n),micro):
            H = (2,4,8,16)[int(torch.randint(0,4,(),generator=gen))]
            t = int(torch.randint(0,S-H,(),generator=gen))
            q = a[rows]
            bad = q[:,:t+1].any(1)
            valid = ~q[:,t:t+H+1].any(1)
            z["rows"] += len(q)
            z["frames"] += len(q)*S
            z["split_after_reset"] += int(bad.sum())
            z["valid_student_rows"] += int(valid.sum())
            z["corrupted_teacher_endpoints"] += int((bad&valid).sum())
            z["suffix_until_reset"] += int((bad[:,None] & ~np.cumsum(q[:,t+1:],axis=1).astype(bool)).sum())
        frames += n*S
    saved = ck["streams"]
    for k in ("queue","cur","t","uid","rng"):
        assert saved[k] == s.state()[k], k
    assert torch.equal(gen.get_state(),ck["rng"]["gen"])
    assert frames == ck["state"]["frames"]
    save("EXPOSURE", {"exact_final_stream_and_horizon_rng": True, "frames": frames, "by_chunk": result})

@torch.no_grad()
def gpu():
    torch.set_num_threads(2)
    ck = torch.load(CK,map_location="cpu",mmap=True,weights_only=False)
    cfg = ck["cfg"]
    bel = M.Belief(cfg).cuda().eval()
    bel.load_state_dict({k.removeprefix("belief."):v for k,v in ck["model"].items() if k.startswith("belief.") and k!="belief.mask_token"})
    # Two rows, one with a real reset before the fork, one after. The prefix is real recorded input, not random carry.
    ep = D.load_split(cfg)["dev"][:2]
    frames = np.stack([D.RP.episode_fields(e.shard,e.slot)["observations"][:24].numpy() for e in ep])
    a = torch.tensor(np.stack([D.RP.episode_fields(e.shard,e.slot)["actions_taken"][:24].numpy() for e in ep]),device="cuda")
    from saccade.eval import s1
    model = M.M0(cfg).cuda().eval()
    model.load_state_dict({k:v for k,v in ck["model"].items() if k!="belief.mask_token"})
    initial = bel.initial(2,"cuda")
    q = torch.zeros((2,48),dtype=torch.long,device="cuda")
    q[0,8:] = 1; q[1,32:] = 1
    split = 25
    with TR.autocast(a.device,cfg):
        p=model.perception(torch.as_tensor(frames,device=a.device),torch.full((2,24,2),4,device=a.device,dtype=torch.long))
        x=bel.tokens(bel.percept_embed(p),torch.zeros(2,24,device=a.device),torch.zeros(2,24,device=a.device,dtype=torch.bool),a)
        all_y,all_c=bel(x,initial,q)
        prefix,c=bel(x[:,:split],initial,q[:,:split])
        y_bad,c_bad=bel(x[:,split:],c,q[:,split:])
        y_ok,c_ok=bel(x[:,split:],c,q[:,split:]-q[:,split-1:split])
        y_zero,_=bel(x[:1,split:],bel.initial(1,"cuda"),torch.zeros_like(q[:1,split:]))
    def diff(x,y):
        e=(x.float()-y.float()).flatten(1)
        return {"rms":e.square().mean(1).sqrt().tolist(),"max":e.abs().amax(1).tolist()}
    r={"split_token":split,"suffix_first_segments":q[:,split].tolist(),
       "as_written_vs_contiguous":diff(y_bad,all_y[:,split:]),"rebased_vs_contiguous":diff(y_ok,all_y[:,split:]),
       "affected_row_vs_zero_carry":diff(y_bad[:1],y_zero),
       "carry_rms_as_written_vs_rebased":[diff(b.ssm,c.ssm)["rms"] for b,c in zip(c_bad,c_ok)]}
    save("RESET_GPU",r)

if __name__=="__main__":
    {"census":census,"gpu":gpu}[sys.argv[1]]()
