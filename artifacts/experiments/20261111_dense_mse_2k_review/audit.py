"""Read-only CPU audit of the live 2k continuation. Writes only this review directory."""
import os
os.environ.setdefault("JAX_PLATFORMS","cpu")
os.environ.setdefault("OMP_NUM_THREADS","2")
import ast,hashlib,json
from pathlib import Path
import numpy as np
import torch
from saccade import data as D,chooser as C,train as TR
HERE=Path(__file__).resolve().parent
E=HERE.parent
BASE=E/"20261111_dense_mse_short"
EXT=E/"20261111_dense_mse_2k"
R=TR.ROOT

def load(p):
    return torch.load(p,map_location="cpu",mmap=True,weights_only=False)

def j(p):
    return json.loads(p.read_text())

def main():
    torch.set_num_threads(2)
    pins=j(EXT/"RUN_PINS.json")
    mismatches=[p for p,h in pins.items() if TR.file_sha256(R/p)!=h]
    assert not mismatches,mismatches
    assert all((BASE/f).read_bytes()==(EXT/f).read_bytes() for f in ("loss.py","readout.py"))
    def fn(path,name):
        return next(n for n in ast.parse(path.read_text()).body if isinstance(n,ast.FunctionDef) and n.name==name)
    # Same computation, apart from formatting and changed outer cap/resume admission.
    old,new=[fn(p/"train.py","run") for p in (BASE,EXT)]
    a,b=[next(n for n in f.body if isinstance(n,ast.For)) for f in (old,new)]
    assert ast.dump(a)==ast.dump(b)
    assert ast.dump(fn(BASE/"train.py","fixture"))==ast.dump(fn(EXT/"train.py","fixture"))
    base={x:load(BASE/"run"/x/"latest.pt") for x in ("endpoint","dense")}
    assert base["endpoint"]["cfg"]==base["dense"]["cfg"]
    assert base["endpoint"]["streams"]==base["dense"]["streams"]
    assert all(torch.equal(base["endpoint"][k],base["dense"][k]) for k in ("cpu","cuda","gen"))
    cfg=base["endpoint"]["cfg"]
    eps=D.load_split(cfg)["train"];offs=D.frame_offsets(eps)
    n=cfg["data"]["streams"];p=cfg["data"]["context_prefix"];T=64
    trace=np.memmap(E/"20261109_p1_hud_change/run/gaze.bin",dtype=C.TRACE,mode="r")
    assert len(trace)>=2000*n*T
    target=TR.file_sha256(D.targets_dir(cfg)/"change_targets.npy")
    out={"pins_verified":len(pins),"identical_loss_and_readout_bytes":True,"identical_training_loop_ast":True,
         "identical_model_constructor_ast":True,"arms":{}}
    for arm in ("endpoint","dense"):
        bp=BASE/"run"/arm;ep=EXT/"run"/arm
        b=base[arm];x=load(ep/"latest.pt")
        step=int(x["step"])
        assert b["step"]==384 and 384<=step<=2000 and x["cfg"]==b["cfg"]
        # atomic checkpoint rename means log can be ahead of this saved snapshot; compare checkpoint-owned prefix.
        raw=(ep/"log.jsonl").read_bytes()[:x["log_bytes"]]
        oldlog=(bp/"log.jsonl").read_bytes()
        assert len(oldlog)==b["log_bytes"] and raw.startswith(oldlog)
        logs=[json.loads(l) for l in raw.splitlines()]
        assert [v["update"] for v in logs]==list(range(1,step+1))
        same_source=TR.file_sha256(ep/"latest.pt")==TR.file_sha256(bp/"latest.pt") if step==384 else None
        if step==384:assert same_source
        if (ep/"identity.json").exists():
            i=j(ep/"identity.json");o=j(bp/"identity.json")
            assert i["updates"]==2000 and i["mode"]==arm and i["seed"]==o["seed"] and i["parent"]==o["parent"]
            assert i["data"]["targets_sha256"]==o["data"]["targets_sha256"]==target
            assert i["data"]["manifest_sha256"]==o["data"]["manifest_sha256"]
            assert i["gaze_prefix_sha256"]==hashlib.sha256(trace[:2000*n*T].tobytes()).hexdigest()
        s=D.Streams(eps,n,p,seed=1,targets_offsets=offs);s.load_state(b["streams"])
        g=torch.Generator();g.set_state(b["gen"])
        for u in range(384,step):
            rr=trace[u*n*T:(u+1)*n*T].reshape(n,T)
            for row in range(n):
                pos=0;hist=[]
                while pos<T:
                    e,uid=s.cur[row];t=s.t[row]
                    length=min(T-pos,eps[e].T+1-t)
                    f=D.fixation_prior(s.rng,(length,))
                    q=rr[row,pos:pos+length]
                    assert np.array_equal(q["uniform"],(f[:,0]-1)*7+f[:,1]-1)
                    ts=np.arange(t,t+length)
                    assert np.all(q["episode"]==uid) and np.array_equal(q["target"],np.where(ts>0,offs[e]+ts,-1))
                    win=q["window"].astype(int)
                    hist.extend((e,uid,t+i,1+v//7,1+v%7) for i,v in enumerate(win.tolist()))
                    pos+=length;s.t[row]+=length
                    if s.t[row]==eps[e].T+1:s.cur[row],s.t[row]=s._pull(),0
                s.hist[row]=hist[-p:]
            hs=[]
            for _ in range(2):
                h=(2,4,8,16)[int(torch.randint(0,4,(),generator=g))]
                torch.randint(0,T-h,(),generator=g)
                hs.append(h)
            assert logs[u]["horizon"]==sum(hs)/2
        assert s.state()==x["streams"] and torch.equal(g.get_state(),x["gen"])
        moment_steps=sorted({int(v["step"]) for v in x["opt"]["state"].values() if "step" in v})
        assert moment_steps==[step],moment_steps
        z={"checked_saved_update":step,"prefix_384_log_bitwise_equal":True,"checkpoint_equal_to_384_if_not_advanced":same_source,
           "full_stream_and_horizon_replay_equal":True,"adam_steps":moment_steps,
           "median_update_seconds_after_384":float(np.median([v["seconds"] for v in logs[384:]])) if step>384 else None}
        if (ep/"eval.pt").exists():
            y=load(ep/"eval.pt")
            assert y["state"]["update"]==2000 and step==2000
            assert y["training_sha256"]==TR.file_sha256(ep/"latest.pt")
            assert set(x["model"])-set(y["model"])=={"belief.mask_token"}
            assert all(torch.equal(v,x["model"][k]) for k,v in y["model"].items())
            z["inference_export_verified"]=True
        out["arms"][arm]=z
    (HERE/"CHECKS.json").write_text(json.dumps(out,indent=2))
    print(json.dumps(out,indent=2),flush=True)
if __name__=="__main__":main()
