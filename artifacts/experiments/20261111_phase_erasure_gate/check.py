"""CPU simulator replay and arithmetic objective gate; no world learning, no edits to original banks."""
import os
os.environ.setdefault("JAX_PLATFORMS","cpu")
import json,math
from pathlib import Path
import numpy as np
import torch
from saccade import data as D,model as M,train as TR

P=Path(__file__).resolve().parent
E=P.parent
H=(1,2,4,8,16)
FORK=24

def run():
    import jax,jax.numpy as jnp
    assert jax.default_backend()=="cpu"
    torch.set_num_threads(2)
    bankpath=E/"20261108_full_frame_change/phase_bank.pt"
    bank=torch.load(bankpath,map_location="cpu",weights_only=False)
    L=dict(np.load(D.ROOT/"artifacts/eda/s0/s0_dev_labels.npz"))
    _,_,_,step,render=D.RP.env_and_render()
    ckpath=E/"20261106_p1f4/run/ckpt_10.pt"
    ck=torch.load(ckpath,map_location="cpu",weights_only=False,mmap=True)
    encoder=M.Perception(ck["cfg"]).eval()
    encoder.load_state_dict({k.removeprefix("perception."):v for k,v in ck["model"].items() if k.startswith("perception.")})
    out=[]
    for rec in bank:
        i=rec["e"];t=rec["t0"]
        state=D.RP.advance_to(int(L["ep_shard"][i]),int(L["ep_slot"][i]),t)
        st=[state.replace(player_hunger=jnp.asarray(h,state.player_hunger.dtype)) for h in (20.,12.)]
        keys=jax.random.split(jax.random.PRNGKey(110812+i),50)
        frame=[[],[]];food=[[],[]];rw=[[],[]];term=[[],[]]
        for j in range(FORK+max(H)+1):
            for b in (0,1):
                z=st[b]
                frame[b].append(np.asarray(render(z)));food[b].append(int(z.player_food))
                _,st[b],r,d,_=step(keys[j],z,jnp.asarray(0,jnp.int32))
                rw[b].append(float(r));term[b].append(bool(d))
        f=np.asarray(frame);fd=np.asarray(food);rw=np.asarray(rw);term=np.asarray(term)
        # Original bank is the reproduction control. Later continuations use the same keys.
        n=rec["frames"].shape[1]
        assert np.array_equal(f[:,:n],rec["frames"])
        assert np.array_equal(fd[:,:n],rec["food"])
        assert np.array_equal(rw[:,:n-1],rec["reward"]) and np.array_equal(term[:,:n-1],rec["term"])
        assert np.array_equal(f[0,FORK-7:FORK+1],f[1,FORK-7:FORK+1])
        assert np.array_equal(rw[0,FORK-8:FORK],rw[1,FORK-8:FORK])
        if term[:,:FORK+max(H)].any():
            out.append({"e":i,"excluded":"termination before endpoint"});continue
        crop=f.reshape(2,len(f[0]),9,7,9,7,3).transpose(0,1,2,4,3,5,6)
        diff=np.any(crop[:,FORK+1:]!=crop[:,FORK:FORK+1],axis=(-1,-2,-3))
        row={"e":i,"current_food":fd[:,FORK].tolist(),
             "first_food_drop":(np.argmax(fd[:,FORK+1:]<fd[:,FORK:FORK+1],axis=1)+1).tolist(),
             "h":{}}
        for h in H:
            f1=f[:,FORK+h]
            fix=torch.full((2,2),4,dtype=torch.long)
            with torch.no_grad():
                p=encoder(torch.as_tensor(f1.copy()),fix)
            dp=(p.p[0]-p.p[1]).float()
            sq=sum((x[0]-x[1]).float().square().sum().item() for x in p.parts())
            n=sum(x[0].numel() for x in p.parts())
            different=diff[0,h-1]!=diff[1,h-1]
            row["h"][str(h)]=dict(food=fd[:,FORK+h].tolist(),
                 different_rgb_cells=int(np.any(crop[0,FORK+h]!=crop[1,FORK+h],axis=(-1,-2,-3)).sum()),
                 different_change_queries=int(different.sum()), latent_mse_erasure_gap=sq/(4*n),
                 max_local_periphery_mse_gap=float(dp.square().mean(-1).max()/4),
                 change_query_logscore_gap_nats=float(different.sum()*math.log(2)/81))
        # All screen cells are queried with the same generic first-visible-change target.
        onset=np.where(diff.any(1),diff.argmax(1)+1,17)
        row["first_change_different_queries"]=int((onset[0]!=onset[1]).sum())
        row["first_change_full_grid_gap_nats"]=row["first_change_different_queries"]*math.log(2)/81
        out.append(row)
        print(i,row["first_food_drop"],row["first_change_different_queries"],flush=True)
    report={"fork":FORK,"checked_against_original_bank":True,"bank_sha256":TR.file_sha256(bankpath),"encoder_checkpoint_sha256":TR.file_sha256(ckpath),"rows":out}
    (P/"ERASURE.json").write_text(json.dumps(report,indent=2))
if __name__=="__main__":run()
