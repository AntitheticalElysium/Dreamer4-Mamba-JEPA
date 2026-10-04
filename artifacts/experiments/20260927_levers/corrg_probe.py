"""E5d. Where is the move decision lost in the corrg world: in the backbone, or in the head that reads it?

Futures one-step fans, move actions, moved vs blocked (simulator truth). Ridge probes (train seeds -> test seeds,
diagnosis split), AUC, from:
  input_target   the INPUT token of the tile the player moves into (context frame t)
  h_target       the backbone output at that tile, last position
  h_player       the backbone output at the player tile (31)
  h_action       the backbone output of the action token (what corrg's frame gate reads)
  frame_logit    corrg's learned gate itself (no probe)
If h_action is probe-decodable near 1.0 while frame_logit sits at 0.78, the head is under-fit; if h_action itself is
~0.78, the backbone does not route the target tile's passability to the action token.
"""
import json
import sys
from pathlib import Path

import torch

HERE = Path(__file__).parent
ROOT = Path("/home/antithetical/EPITA/PERSO/DynamicHorizons-Mamba-JEPA")
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "artifacts/experiments/20260926_diagnosis"))
import teval as T  # noqa: E402
from compound import auc  # noqa: E402
from onestep import classify  # noqa: E402

TARGET = {1: 30, 2: 32, 3: 22, 4: 40}


def probe_auc(x, y, tr, te, lams=(1e-3, 1e-2, 1e-1, 1, 10)):
    mu, sd = x[tr].mean(0), x[tr].std(0).clamp_min(1e-6)
    X = torch.cat([((x - mu) / sd).double(), torch.ones(len(x), 1, dtype=torch.float64)], 1)
    t = y.double() * 2 - 1
    idx = torch.where(tr)[0]; a, b = idx[: len(idx) // 2], idx[len(idx) // 2:]
    best = None
    for lam in lams:
        w = torch.linalg.solve(X[a].T @ X[a] + lam * len(a) * torch.eye(X.shape[1], dtype=X.dtype), X[a].T @ t[a])
        s = auc(X[b] @ w, y[b])
        if best is None or s > best[0]:
            best = (s, lam)
    w = torch.linalg.solve(X[tr].T @ X[tr] + best[1] * int(tr.sum()) * torch.eye(X.shape[1], dtype=X.dtype), X[tr].T @ t[tr])
    return auc(X[te] @ w, y[te])


@torch.no_grad()
def main():
    from d4mj.config import config_from_dict
    from d4mj.train import autocast_context
    import spatial as S
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "artifacts/eda/levers_tworlds_v1/corrg_raw_suffix_s7.pt"
    config = config_from_dict(torch.load(S.CHECKPOINT, map_location="cpu", weights_only=False)["config"])
    meta, train_roots, _ = T.split()
    cls, _ = classify(meta)
    cache = T.build_cache("raw", device)
    ctx, ca = cache["ctx"].float(), cache["ctx_a"]
    rows = [(i, a) for i in range(len(ctx)) for a in (1, 2, 3, 4) if cls[i, a] in (0, 1)]
    y = torch.tensor([bool(cls[i, a] == 0) for i, a in rows])
    tr = torch.tensor([bool(train_roots[i]) for i, _ in rows]); te = ~tr
    world, st = T.load_world(path, device)
    feats = {k: [] for k in ("input_target", "h_target", "h_player", "h_action", "frame_logit", "decision_logit")}
    with autocast_context(config):
        for k in range(0, len(rows), 256):
            chunk = rows[k:k + 256]
            s = torch.stack([ctx[i] for i, _ in chunk]).to(device)
            a = torch.stack([torch.cat([ca[i], torch.tensor([act])]) for i, act in chunk]).to(device)
            h, ha = world.backbone_full(s, a)
            tgt = torch.tensor([TARGET[act] for _, act in chunk], device=device)
            ar = torch.arange(len(chunk), device=device)
            feats["input_target"].append(s[ar, -1, tgt].float().cpu())
            feats["h_target"].append(h[ar, -1, tgt].float().cpu())
            feats["h_player"].append(h[:, -1, 31].float().cpu())
            feats["h_action"].append(ha[:, -1].float().cpu())
            frame = world.frame(ha[:, -1]).float()[:, 0]
            feats["frame_logit"].append(frame.cpu())
            gate = world.target_gate(h[ar, -1, tgt]).float()[:, 0] if hasattr(world, "target_gate") else 0 * frame
            feats["decision_logit"].append((frame + gate).cpu())            # corrt's move logit (corrg: = frame)
    feats = {k: torch.cat(v) for k, v in feats.items()}
    res = {k: probe_auc(v, y, tr, te) for k, v in feats.items() if k not in ("frame_logit", "decision_logit")}
    res["frame_logit_itself"] = auc(feats["frame_logit"][te], y[te])
    res["decision_logit_itself"] = auc(feats["decision_logit"][te], y[te])
    res["decision_logit_mean_moved_blocked"] = [float(feats["decision_logit"][y].mean()), float(feats["decision_logit"][~y].mean())]
    res["n_test"] = int(te.sum()); res["moved_share"] = float(y.float().mean())
    out = HERE / ("corrg_probe.json" if len(sys.argv) == 1 else f"probe_{st['name']}.json")
    out.write_text(json.dumps(res, indent=2) + "\n")
    print(json.dumps({k: round(v, 3) if isinstance(v, float) else v for k, v in res.items()}))


if __name__ == "__main__":
    main()
