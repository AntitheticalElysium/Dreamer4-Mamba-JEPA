"""S0 step 5 (2026-10-27): the pixel-probe reference for S1.1 (entity and terrain retention per cell).

Probe definition (fixed here; the token probe after P1 uses the SAME definition with the cell's token in place of its
pixels and the lighting latent v in place of the frame mean):
  input  = one map cell's pixels (7 x 7 x 3, / 255) + the frame's mean map RGB (/ 255);
  model  = MLP 150 -> 256 -> 256 (GELU) with two linear heads: kind (none / zombie / cow / skeleton / arrow) and
           terrain (17 block types);
  loss   = cross-entropy on kind (every map cell but the player's) + cross-entropy on terrain (cells with no entity),
           natural frequencies, no reweighting;
  fit    = AdamW (lr 1e-3, weight decay 1e-4), batch 4096, over the train-400 episodes minus 40 held out, until the
           held-out loss has not improved for 3 epochs (at most 30); the best epoch is kept; 2 seeds. (A fixed 4-epoch
           budget was the first rule; held-out loss was still falling and the seeds disagreed on night arrows by ~8 SE,
           so it was a harness defect, fixed before any token probe exists.)
Reading on ALL dev episodes, by light (day / night at 0.5) and ring (Chebyshev distance to the player, 1-4), with
episode-bootstrap SE: entity presence recall and precision, kind accuracy on entity cells, per-kind recall, terrain
accuracy and class-averaged recall (classes with >= 100 dev cells).
Usage (project venv, GPU): s0_probe.py TRAIN_LABELS_NPZ DEV_LABELS_NPZ OUT_JSON
"""
import json, sys
import numpy as np, torch, torch.nn as nn
sys.path.insert(0, "/home/antithetical/EPITA/PERSO/DynamicHorizons-Mamba-JEPA/artifacts/eda")
import replay as RP
import s0_gates as G

DEV = torch.device("cuda")
RING = np.maximum(np.abs(np.arange(7)[:, None] - 3), np.abs(np.arange(9)[None] - 4)).ravel()
KEEP = RING > 0                                                     # every map cell but the player's


def cells(L):
    """Per episode: cell pixels [n*62, 147] uint8, frame mean [n*62, 3], kind, terrain (-1 if covered), ring, night."""
    s_off = np.r_[0, np.cumsum(L["ep_T"] + 1)]
    for e, ep in enumerate(G.episode_grids(L)):
        obs = RP.episode_fields(int(L["ep_shard"][e]), int(L["ep_slot"][e]))["observations"].numpy()[:, :49]
        n = len(obs); assert n == len(ep["grid"])
        px = obs.reshape(n, 7, 7, 9, 7, 3).transpose(0, 1, 3, 2, 4, 5).reshape(n, 63, 147)[:, KEEP]
        mean = np.repeat(np.round(obs.reshape(n, -1, 3).mean(1)).astype(np.uint8)[:, None], KEEP.sum(), 1)
        kind = ep["grid"].reshape(n, 63)[:, KEEP]; terr = ep["view"].reshape(n, 63)[:, KEEP].astype(np.int8)
        terr = np.where(kind == 0, terr, -1).astype(np.int8)
        night = np.repeat(L["s_light"][s_off[e]:s_off[e + 1]] < 0.5, KEEP.sum())
        yield e, px.reshape(-1, 147), mean.reshape(-1, 3), kind.ravel(), terr.ravel(), np.tile(RING[KEEP].astype(np.int8), n), night


class Probe(nn.Module):
    def __init__(self):
        super().__init__()
        self.body = nn.Sequential(nn.Linear(150, 256), nn.GELU(), nn.Linear(256, 256), nn.GELU())
        self.kind, self.terr = nn.Linear(256, 5), nn.Linear(256, 17)

    def forward(self, px, mean):
        h = self.body(torch.cat([px.float() / 255, mean.float() / 255], 1))
        return self.kind(h), self.terr(h)


def loss_fn(model, px, mean, k, t):
    lk, lt = model(px, mean)
    return nn.functional.cross_entropy(lk, k) + nn.functional.cross_entropy(lt, t, ignore_index=-1)


def fit(train, val, seed):
    torch.manual_seed(seed); model = Probe().to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    best, best_state, log = 1e9, None, []
    g = torch.Generator(device=DEV).manual_seed(seed)
    for epoch in range(30):
        model.train()
        for idx in torch.randperm(len(train[0]), device=DEV, generator=g).split(4096):
            loss = loss_fn(model, *(x[idx] for x in train)); opt.zero_grad(); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            vl = float(np.mean([loss_fn(model, *(x[i] for x in val)).item() for i in torch.arange(len(val[0]), device=DEV).split(65536)]))
        log.append(vl)
        if vl < best:
            best, best_state, best_epoch = vl, {k: v.clone() for k, v in model.state_dict().items()}, epoch
        elif epoch - best_epoch >= 3:
            break
    model.load_state_dict(best_state)
    return model, log


def to_dev(parts):
    return [torch.as_tensor(np.concatenate(p)).to(DEV) for p in parts]


def main():
    Ltr, Ldev = dict(np.load(sys.argv[1])), dict(np.load(sys.argv[2]))
    tr, va = [[], [], [], []], [[], [], [], []]
    for e, px, mean, kind, terr, ring, night in cells(Ltr):
        for p, x in zip(va if e % 10 == 9 else tr, (px, mean, kind.astype(np.int64), terr.astype(np.int64))):
            p.append(x)
    train, val = to_dev(tr), to_dev(va); del tr, va
    print("train cells", len(train[0]), "held-out cells", len(val[0]), flush=True)
    out = {"n_train_cells": len(train[0]), "n_heldout_cells": len(val[0]), "seeds": {}}
    for seed in (0, 1):
        model, log = fit(train, val, seed)
        R = {"kind": [], "kp": [], "terr": [], "tp": [], "ring": [], "night": [], "ep": []}
        with torch.no_grad():
            for e, px, mean, kind, terr, ring, night in cells(Ldev):
                lk, lt = model(torch.as_tensor(px).to(DEV), torch.as_tensor(mean).to(DEV))
                for k, v in (("kind", kind), ("kp", lk.argmax(1).to(torch.int8).cpu().numpy()), ("terr", terr),
                             ("tp", lt.argmax(1).to(torch.int8).cpu().numpy()), ("ring", ring), ("night", night),
                             ("ep", np.full(len(kind), e, np.int32))):
                    R[k].append(v)
        R = {k: np.concatenate(v) for k, v in R.items()}
        pres, ppres, valid = R["kind"] > 0, R["kp"] > 0, R["terr"] >= 0
        res = {"heldout_loss_by_epoch": log}
        for lname, lm in (("day", ~R["night"]), ("night", R["night"])):
            for ring in (0, 1, 2, 3, 4):
                rm = lm & ((R["ring"] == ring) if ring else True)
                key = f"{lname}_ring{ring}" if ring else f"{lname}_all"
                res[key] = {"presence_recall": G.estimate(ppres, pres & rm, R["ep"]),
                            "presence_precision": G.estimate(pres, ppres & rm, R["ep"]),
                            "kind_accuracy_on_entities": G.estimate(R["kp"] == R["kind"], pres & rm, R["ep"]),
                            "terrain_accuracy": G.estimate(R["tp"] == R["terr"], valid & rm, R["ep"])}
            for k, m in enumerate(G.KINDS, start=1):
                res[f"{lname}_all"][f"recall_{m}"] = G.estimate(R["kp"] == k, (R["kind"] == k) & lm, R["ep"])
            cls = [c for c in range(17) if ((R["terr"] == c) & lm).sum() >= 100]
            res[f"{lname}_all"]["terrain_class_averaged_recall"] = float(np.mean([((R["tp"] == c) & (R["terr"] == c) & lm).sum() / ((R["terr"] == c) & lm).sum() for c in cls]))
            res[f"{lname}_all"]["terrain_classes"] = cls
        out["seeds"][seed] = res
        print(seed, json.dumps({k: v for k, v in res.items() if k.endswith("_all") or k == "heldout_loss_by_epoch"}), flush=True)
    json.dump(out, open(sys.argv[3], "w"), indent=1)


if __name__ == "__main__":
    main()
