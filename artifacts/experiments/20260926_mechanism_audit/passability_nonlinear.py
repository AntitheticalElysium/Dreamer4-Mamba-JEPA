"""Nonlinear capacity control for passability from the saved U root state.

55k train, 56k select training epoch, 55k+56k refit, 57k+58k test.
All roots are previously inspected; this only tests whether the linear U
passability readout understated what its 192 dimensions retain.
"""
import json
import sys
from pathlib import Path

import torch
from torch import nn

HERE = Path(__file__).parent
ROOT = HERE.parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "artifacts/experiments/20260926_diagnosis"), str(HERE)]
from choices import move_table
from decision import unpack, C, MOVES, LAVA
from transition_corrected import auc


def data(block, action):
    d = ROOT / "artifacts/eda/diagnosis_dump_v1"
    meta = torch.load(d / f"{block}_meta.pt", weights_only=False)
    root = torch.load(d / f"{block}_U.pt", weights_only=False)["root"].float()
    cat, _ = move_table(meta["visible"])
    lava = []
    for v in meta["visible"]:
        tiles, _, _, _ = unpack(v)
        dr, dc = MOVES[action]
        lava.append(int(tiles[C[0]+dr, C[1]+dc]) == LAVA)
    sel = (cat[:, action] <= 1) & ~torch.tensor(lava)
    return root[sel], (cat[sel, action] == 0).float()


def network(seed):
    torch.manual_seed(seed)
    return nn.Sequential(nn.Linear(192, 256), nn.LayerNorm(256), nn.GELU(),
                         nn.Linear(256, 128), nn.GELU(), nn.Linear(128, 1))


def fit(x, y, seed, epochs, *, dev=None):
    model = network(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    order = torch.Generator().manual_seed(seed + 100)
    best = (-1.0, 0)
    for ep in range(1, epochs + 1):
        model.train()
        for idx in torch.randperm(len(y), generator=order).split(128):
            logits = model(x[idx]).squeeze(-1)
            loss = nn.functional.binary_cross_entropy_with_logits(logits, y[idx])
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        if dev is not None and ep % 5 == 0:
            model.eval()
            with torch.no_grad():
                metric = auc(model(dev[0]).squeeze(-1), dev[1])
            if metric > best[0]:
                best = (metric, ep)
    return model.eval(), best


def main():
    torch.set_num_threads(4)
    out = {}
    for action in (1, 2, 3, 4):
        blocks = {b: data(b, action) for b in ("55k", "56k", "57k", "58k")}
        xtr,ytr=blocks["55k"]; xdv,ydv=blocks["56k"]
        xf=torch.cat([xtr,xdv]);yf=torch.cat([ytr,ydv])
        xt=torch.cat([blocks["57k"][0],blocks["58k"][0]])
        yt=torch.cat([blocks["57k"][1],blocks["58k"][1]])
        mu,sd=xtr.mean(0),xtr.std(0).clamp_min(1e-6)
        ztr,zdv=(xtr-mu)/sd,(xdv-mu)/sd
        fullmu,fullsd=xf.mean(0),xf.std(0).clamp_min(1e-6)
        zf,zt=(xf-fullmu)/fullsd,(xt-fullmu)/fullsd
        runs=[]
        for seed in (1,2,3):
            _, best = fit(ztr,ytr,seed,epochs=150,dev=(zdv,ydv))
            model,_=fit(zf,yf,seed,epochs=best[1])
            with torch.no_grad():
                test=auc(model(zt).squeeze(-1),yt)
            runs.append({"seed":seed,"selected_epoch":best[1],"dev_auc":best[0],"test_auc":test})
        out[str(action)]={"n_fit":len(yf),"n_test":len(yt),"runs":runs,
                          "mean_test_auc":sum(r["test_auc"] for r in runs)/len(runs)}
        print(action,out[str(action)],flush=True)
    out["mean_auc"]=sum(v["mean_test_auc"] for v in out.values())/4
    (HERE / "passability_nonlinear.json").write_text(json.dumps(out,indent=2)+"\n")
    print("overall",out["mean_auc"],flush=True)


if __name__=="__main__":
    main()
