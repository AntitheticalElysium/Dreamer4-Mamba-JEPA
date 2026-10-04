"""E5c. Why does the neighbour-copy world still scroll on blocked moves? Is the move decision learned, and is it applied?

On the diagnostic futures' one-step fans (key 0), move actions only, split by the simulator into moved / blocked
(onestep.classify; lava entries count as moved). For the corr and corrg worlds (seed 7, Raw tokens):
  frame_logit   corrg's per-frame "moved" logit: its distribution for moved vs blocked, and AUC
  copy_mass     mean over the 81 tiles of the softmax mass on the 4 neighbour candidates (a scroll puts ~1 on
                the neighbour opposite to the move): moved vs blocked, and AUC
  in-train      the same two AUCs on 2,048 TRAIN windows of spatial_pool_v1 (moves only; moved/blocked from the
                pixel-free token test: the next frame's tokens closer to the rolled root than to the root itself)
Reads the decision out of the model rather than inferring it from errors.
"""
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

HERE = Path(__file__).parent
ROOT = Path("/home/antithetical/EPITA/PERSO/DynamicHorizons-Mamba-JEPA")
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "artifacts/experiments/20260926_diagnosis"))
import teval as T  # noqa: E402
from compound import auc  # noqa: E402
from onestep import classify  # noqa: E402
from tworld import NEIGHBOURS, TWorld  # noqa: E402

W = ROOT / "artifacts/eda/levers_tworlds_v1"
N = 17


@torch.no_grad()
def readout(world, s, a):
    """s [B,T,81,192], a [B,T] -> (copy_mass [B], frame_logit [B] or None) at the last position."""
    if world.head == "corrg":
        h, ha = world.backbone_full(s, a)
    else:
        h, ha = world.backbone(s, a), None
    logits = world.choose(h[:, -1]).float()                                        # [B,81,6]
    frame = None
    if ha is not None:
        frame = world.frame(ha[:, -1]).float()[:, 0]
        logits = logits + frame[:, None, None] * logits.new_tensor([0, 1, 1, 1, 1, 0])
    w = torch.softmax(logits, -1)
    return w[..., 1:5].sum(-1).mean(-1), frame


def main():
    from d4mj.config import config_from_dict
    from d4mj.train import autocast_context
    import spatial as S
    device = torch.device("cuda")
    config = config_from_dict(torch.load(S.CHECKPOINT, map_location="cpu", weights_only=False)["config"])
    meta, _, _ = T.split()
    cls, _ = classify(meta)
    cache = T.build_cache("raw", device)
    ctx, ca = cache["ctx"].float(), cache["ctx_a"]
    rows = [(i, a) for i in range(len(ctx)) for a in (1, 2, 3, 4) if cls[i, a] in (0, 1)]
    label = torch.tensor([bool(cls[i, a] == 0) for i, a in rows])
    # training windows: moves only, moved/blocked from tokens
    pool = torch.load(ROOT / "artifacts/eda/spatial_pool_v1/pool.pt", weights_only=False, mmap=True)
    main_rows = torch.where(~pool["terminal"])[0]
    pick = main_rows[torch.randperm(len(main_rows), generator=torch.Generator().manual_seed(5))[:4096]]
    toks = pool["tokens"][pick][:, :5].float()                                     # frames 0..4
    acts = pool["actions"][pick][:, :4]                                            # a_0..a_3 (a_3: frame 3 -> 4)
    move = (acts[:, 3] >= 1) & (acts[:, 3] <= 4)
    toks, acts = toks[move][:2048], acts[move][:2048]
    root, nxt = toks[:, 3], toks[:, 4]
    shift = {1: (0, 1), 2: (0, -1), 3: (1, 0), 4: (-1, 0)}                          # view scroll opposite to the move
    rolled = torch.stack([torch.roll(root[j].view(9, 9, -1), shift[int(acts[j, 3])], (0, 1)).view(81, -1)
                          for j in range(len(root))])
    map_idx = torch.tensor(T.MAP)
    d_still = (nxt[:, map_idx] - root[:, map_idx]).norm(dim=-1).mean(-1)
    d_roll = (nxt[:, map_idx] - rolled[:, map_idx]).norm(dim=-1).mean(-1)
    train_label = d_roll < d_still
    result = {"futures": {"moves": len(rows), "moved_share": float(label.float().mean())},
              "train_windows": {"moves": len(root), "moved_share": float(train_label.float().mean())}}
    for name in ("corr_raw_suffix_s7", "corrg_raw_suffix_s7"):
        world, _ = T.load_world(W / f"{name}.pt", device)
        mass, frame = [], []
        with autocast_context(config):
            for k in range(0, len(rows), 256):
                chunk = rows[k:k + 256]
                s = torch.stack([ctx[i] for i, _ in chunk]).to(device)
                a = torch.stack([torch.cat([ca[i], torch.tensor([act])]) for i, act in chunk]).to(device)
                m, f = readout(world, s, a)
                mass.append(m.cpu()); frame.append(f.cpu() if f is not None else None)
            tm, tf = [], []
            for k in range(0, len(toks), 256):
                m, f = readout(world, toks[k:k + 256, :4].to(device), acts[k:k + 256].to(device))
                tm.append(m.cpu()); tf.append(f.cpu() if f is not None else None)
        mass, tm = torch.cat(mass), torch.cat(tm)
        r = {"copy_mass_moved": float(mass[label].mean()), "copy_mass_blocked": float(mass[~label].mean()),
             "copy_mass_auc": auc(mass, label), "train_copy_mass_auc": auc(tm, train_label),
             "train_copy_mass_moved": float(tm[train_label].mean()), "train_copy_mass_blocked": float(tm[~train_label].mean())}
        if frame[0] is not None:
            frame, tf = torch.cat(frame), torch.cat(tf)
            r.update({"frame_logit_moved": float(frame[label].mean()), "frame_logit_blocked": float(frame[~label].mean()),
                      "frame_logit_auc": auc(frame, label), "train_frame_logit_auc": auc(tf, train_label)})
        result[name] = r
        print(name, json.dumps({k: round(v, 3) for k, v in r.items()}), flush=True)
    (HERE / "corrg_diag.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k in ("futures", "train_windows")}))


if __name__ == "__main__":
    main()
