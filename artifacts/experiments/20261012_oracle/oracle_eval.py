"""10-12 oracle-innovation test (NOTEBOOK "10-12 ORACLE-INNOVATION TEST", predeclared): evaluation on the DEV panel (build.py panel:
200 roots, frames t-15 .. t+16, index 15 = root). Worlds: oracle_world.py outputs (control / oracle) and the two parents.
  tf    teacher-forced depth k = 1..16: the 15 true frames ending at frame 14 + k (and their actions) predict frame 15 + k
  gen   self-fed 16 steps from the 15 true frames ending at the root; every input position j of an oracle world gets the TRUE layout
        of the frame after it (the true future, also under imagined frames)
Decoding: scoreboard.AllProbes (fitted on true TRAIN tokens of the diagnosis panel; same Raw encoder), map cells except the
player's (31). Metrics per depth group (1, 2-4, 5-8, 9-16), per mob type (zombie, skeleton, cow):
  census      decoded count / true count
  recall      decoded present among true-present cells;   precision   true present among decoded-present cells
  arrive      recall on cells a mob ARRIVED at (true at k, not at k - 1)            -- destination completion
  residue     decoded present on cells a mob LEFT (true at k - 1, not at k)        -- source residue (ghost copies)
For gen, also split by scroll alignment (the imagined cumulative scroll, scroll.estimate, equals the true one).
TRUE (name=TRUE): the real future tokens, i.e. the reader's ceiling on this panel.
Usage: oracle_eval.py <name=path.pt | name=TRUE> ...  -> artifacts/eda/oracle_v1/eval_<name>.json
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path("/home/antithetical/EPITA/PERSO/DynamicHorizons-Mamba-JEPA")
for d in ("20260927_levers", "20260926_diagnosis", "20260921_readout_ladder"):
    sys.path.insert(0, str(ROOT / "artifacts/experiments" / d))
sys.path.insert(0, str(ROOT / "artifacts/experiments/20261012_oracle"))
OUT = ROOT / "artifacts/eda/oracle_v1"
H, W = 16, 15
TYPES = ("zombie", "skeleton", "cow")
LAYOUT_CH = {"zombie": 0, "cow": 1, "skeleton": 2}                        # build.py channel order
PROBE_CH = {"zombie": 0, "skeleton": 1, "cow": 2}                         # AllProbes.mob order
MAPC = [c for c in range(63) if c != 31]


def load(path, device):
    import tworld as TW
    import oracle_world as OW
    st = torch.load(path, map_location="cpu", weights_only=False)
    world = TW.TWorld("corrt", None, "fmamba", "all", 0, False, 16)
    w = st["world"]
    if w["time"].shape[0] != world.time.shape[0]:
        w["time"] = w["time"][torch.arange(world.time.shape[0]) % w["time"].shape[0]]
    world.load_state_dict(w)
    oc = None
    if st.get("args", {}).get("arm") == "oracle":
        oc = OW.OracleCond(); oc.load_state_dict(st["oracle"]); oc = oc.to(device).eval()
    return world.to(device).eval(), oc


@torch.no_grad()
def step(world, oc, frames, actions, nxt_layout, device, config):
    from d4mj.train import autocast_context
    with autocast_context(config):
        world.delta = oc(nxt_layout.to(device), frames.shape[1]) if oc is not None else None
        out = world(frames.to(device).float(), actions.to(device))[0]
        world.delta = None
    return out[:, -1].float().cpu()


def main():
    import teval as T
    import scoreboard as SB
    from scroll import estimate
    from d4mj.config import config_from_dict
    import spatial as Sp
    device = torch.device("cuda")
    config = config_from_dict(torch.load(Sp.CHECKPOINT, map_location="cpu", weights_only=False)["config"])
    panel = torch.load(OUT / "panel.pt", weights_only=False)
    tok, lay, act = panel["tokens"], panel["layouts"], panel["actions"]                  # [R,32,81,192], [R,32,63,4], [R,31]
    R = len(tok)
    meta, train_roots, train_seeds = T.split()
    cache = T.build_cache("raw", torch.device("cpu"))
    probes = SB.AllProbes(T, cache, meta, train_roots, train_seeds)
    thr = torch.tensor(probes.thr)

    def decode(frames):                                                    # [n,81,192] -> present [n,63,3] (zombie, skeleton, cow)
        cells = frames[:, MAPC].float().flatten(0, 1)
        m = (probes.mob(cells) > thr).view(len(frames), 62, 4)
        out = torch.zeros(len(frames), 63, 3, dtype=torch.bool)
        for j, t in enumerate(TYPES):
            out[:, MAPC, j] = m[..., PROBE_CH[t]]
        return out
    truth = torch.stack([lay[..., LAYOUT_CH[t]] > 0 for t in TYPES], -1)                 # [R,32,63,3]
    truth[:, :, 31] = False
    true_shift = torch.stack([estimate(tok[:, i].float(), tok[:, i + 1].float()) for i in range(31)], 1)   # [R,31]
    for arg in sys.argv[1:]:
        name, path = arg.split("=", 1)
        bs = 8
        tf = torch.empty(R, H, 81, 192)
        gen = torch.empty(R, H, 81, 192)
        if path == "TRUE":                                                 # the reader's ceiling: the real future tokens
            tf[:] = tok[:, 16:32].float(); gen[:] = tok[:, 16:32].float()
            world = None
        else:
            world, oc = load(path, device)
        for i in range(0, R if world is not None else 0, bs):
            sl = slice(i, min(i + bs, R))
            for k in range(1, H + 1):                                      # teacher-forced
                fr = tok[sl, k:k + W]; ac = act[sl, k:k + W]; nl = lay[sl, k + 1:k + 1 + W]
                tf[sl, k - 1] = step(world, oc, fr, ac, nl, device, config)
            frames = [tok[sl, j].float() for j in range(1, 16)]            # self-fed from the 15 true frames ending at the root
            for k in range(1, H + 1):
                fr = torch.stack(frames[-W:], 1)
                ac = act[sl, k:k + W]                                      # actions after frames k .. k + 14 (time indices)
                nl = lay[sl, k + 1:k + 1 + W]
                g = step(world, oc, fr, ac, nl, device, config)
                gen[sl, k - 1] = g; frames.append(g)
        res = {"world": name}
        prev_gen = torch.cat([tok[:, 15:16].float(), gen[:, :-1]], 1)
        gen_shift = torch.stack([estimate(prev_gen[:, k], gen[:, k]) for k in range(H)], 1)
        import driftanat as DA
        aligned = (DA.offsets(gen_shift) == DA.offsets(true_shift[:, 15:31])).all(-1)      # [R,16]
        res["gen_aligned_share"] = [round(float(aligned[:, k].float().mean()), 3) for k in range(H)]
        for mode, pred in (("tf", tf), ("gen", gen)):
            dec = decode(pred.flatten(0, 1)).view(R, H, 63, 3)
            tru = truth[:, 16:32]; prv = truth[:, 15:31]
            groups = {"1": [0], "2-4": [1, 2, 3], "5-8": [4, 5, 6, 7], "9-16": list(range(8, 16))}
            masks = {"all": torch.ones(R, H, dtype=torch.bool)}
            if mode == "gen":
                masks["aligned"] = aligned
            for mk, mask in masks.items():
                for g, ks in groups.items():
                    row = {}
                    for j, t in enumerate(TYPES):
                        sel = mask[:, ks]
                        d = dec[:, ks][..., j][sel]; tr = tru[:, ks][..., j][sel]; pv = prv[:, ks][..., j][sel]
                        arrive, left = tr & ~pv, pv & ~tr
                        row[t] = {"true_per_frame": round(float(tr.sum(-1).float().mean()), 4),
                                  "census": round(float(d.sum() / max(int(tr.sum()), 1)), 3),
                                  "recall": round(float((d & tr).sum() / max(int(tr.sum()), 1)), 3),
                                  "precision": round(float((d & tr).sum() / max(int(d.sum()), 1)), 3),
                                  "arrive_recall": round(float((d & arrive).sum() / max(int(arrive.sum()), 1)), 3) if arrive.any() else None,
                                  "residue": round(float((d & left).sum() / max(int(left.sum()), 1)), 3) if left.any() else None,
                                  "n_arrive": int(arrive.sum()), "n_left": int(left.sum())}
                    res.setdefault(mode, {}).setdefault(mk, {})[g] = row
        (OUT / f"eval_{name}.json").write_text(json.dumps(res, indent=1))
        torch.save({"tf": tf.half(), "gen": gen.half(), "aligned": aligned}, OUT / f"pred_{name}.pt")      # for recursion.py
        print(json.dumps({name: {m: {g: {t: [v[t]["census"], v[t]["recall"], v[t]["arrive_recall"], v[t]["residue"]] for t in ("cow", "zombie")}
                                     for g, v in res[m]["all"].items()} for m in ("tf", "gen")}}), flush=True)
        del world; torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
