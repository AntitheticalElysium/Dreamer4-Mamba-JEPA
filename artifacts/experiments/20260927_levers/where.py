"""Where a per-tile world's error lives: by token group, transition class and depth; and what its copy head chose.

Groups of the 81 tokens: player (31), near (the 4 cells around it), interior (other non-border map cells), edge
(map border: where entering rows/columns land), hud (rows 7-8). Futures roots, as teval:
  onestep   all 17 actions from the 4 context frames: per class (onestep.classify) and group, squared error x copy
            and the group's share of the class's error; corr-family heads: mean mixture weight per group on self, the
            neighbour a successful move of that action scrolls in ("scroll"), the other 3 neighbours, generate
  rollout   the factual 16 steps, teval's windows: per depth and group, imagined squared error / the group's own
            variance, and the group's share of the depth's error
Usage: where.py <world.pt> [...]  -> where_<name>.json. CPU is enough for the `full` backbone.
"""
import json
import sys
from pathlib import Path

import torch

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent / "20260926_diagnosis"))
import teval as T  # noqa: E402

GROUPS = {"player": [31], "near": list(T.NEAR), "interior": T.INTERIOR, "edge": T.EDGE, "hud": list(range(63, 81))}
SCROLL = {1: 3, 2: 4, 3: 1, 4: 2}               # action -> index of its scroll source in [self, up, down, left, right, gen]


@torch.no_grad()
def main():
    from d4mj.config import config_from_dict
    import spatial as S
    from onestep import CLASSES, classify
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = config_from_dict(torch.load(S.CHECKPOINT, map_location="cpu", weights_only=False)["config"])
    meta, _, _ = T.split()
    cls, _ = classify(meta)
    for path in map(Path, sys.argv[1:]):
        world, st = T.load_world(path, device)
        cache = T.build_cache(st["args"]["pool"], device)
        R = len(meta["seed"])
        err = torch.empty(R, T.N, 81)
        weights = torch.zeros(R, T.N, 81, 6)
        gen_err = torch.full((R, T.N, 81), float("nan"))                   # the generate candidate alone, LN'd
        for i in range(0, R, 16):
            b = min(16, R - i)
            fan = cache["ctx"][i:i + b].float().repeat_interleave(T.N, 0)
            acts = torch.cat([cache["ctx_a"][i:i + b].repeat_interleave(T.N, 0), torch.arange(T.N).repeat(b)[:, None]], 1)
            pred = T.step(world, fan, acts, device, config).view(b, T.N, 81, 192)
            err[i:i + b] = ((pred - cache["one"][i:i + b].float()) ** 2).sum(-1)
            if hasattr(world, "last_weights"):
                weights[i:i + b] = world.last_weights[:, -1].float().cpu().view(b, T.N, 81, 6)
                alone = torch.nn.functional.layer_norm(world.last_generated[:, -1].float().cpu(), (192,))
                gen_err[i:i + b] = ((alone.view(b, T.N, 81, 192) - cache["one"][i:i + b].float()) ** 2).sum(-1)
        root = cache["ctx"][:, -1].float()
        cp = ((root[:, None] - cache["one"].float()) ** 2).sum(-1)
        out = {"onestep": {}}
        for ci, c in enumerate(CLASSES):
            m = cls == ci
            row = {}
            for g, tok in GROUPS.items():
                e, q = err[m][:, tok].sum(), cp[m][:, tok].sum()
                row[g] = {"x_copy": float(e / q.clamp_min(1e-9)), "share_of_error": float(e / err[m].sum()),
                          "share_of_copy_error": float(q / cp[m].sum())}
                if hasattr(world, "last_weights"):
                    w = weights[m][:, tok]                                                  # [n,|g|,6]
                    acts = torch.arange(T.N)[None].expand(R, -1)[m]
                    scroll = torch.tensor([SCROLL.get(int(a), 0) for a in acts])
                    on_scroll = w.gather(2, scroll[:, None, None].expand(-1, len(tok), 1))[..., 0]
                    is_move = torch.tensor([int(a) in SCROLL for a in acts])[:, None].expand_as(on_scroll)
                    row[g]["w_self"] = float(w[..., 0].mean())
                    row[g]["w_generate"] = float(w[..., 5].mean())
                    row[g]["w_scroll_source"] = float(on_scroll[is_move].mean()) if is_move.any() else None
                    row[g]["w_all_neighbours"] = float(w[..., 1:5].sum(-1).mean())
            out["onestep"][c] = row
            if hasattr(world, "last_weights"):
                for g, tok in GROUPS.items():
                    row[g]["generator_alone_x_copy"] = float(gen_err[m][:, tok].sum() / cp[m][:, tok].sum().clamp_min(1e-9))
        # moves only: the row / column a move scrolls in (its scroll source is off-grid) vs the rest of the border
        entering_of = {1: [r * 9 for r in range(7)], 2: [r * 9 + 8 for r in range(7)], 3: list(range(9)),
                       4: [54 + c for c in range(9)]}
        for ci, c in ((0, "moved"), (1, "blocked")):
            rows = {"entering": [], "edge_other": []}
            for a, ent in entering_of.items():
                m = cls[:, a] == ci
                other = [e for e in T.EDGE if e not in ent]
                for key, tok in (("entering", ent), ("edge_other", other)):
                    rec = {"err": err[m, a][:, tok].sum(), "copy": cp[m, a][:, tok].sum(), "n": int(m.sum()) * len(tok)}
                    if hasattr(world, "last_weights"):
                        w = weights[m, a][:, tok]
                        rec |= {"gen_err": gen_err[m, a][:, tok].sum(), "w_self": w[..., 0].sum(),
                                "w_scroll": w[..., SCROLL[a]].sum(), "w_generate": w[..., 5].sum()}
                    rows[key].append(rec)
            for key, recs in rows.items():
                tot = lambda f: sum(float(r[f]) for r in recs)
                n = sum(r["n"] for r in recs)
                res = {"x_copy": tot("err") / tot("copy"), "share_of_error": tot("err") / float(err[cls == ci].sum())}
                if hasattr(world, "last_weights"):
                    res |= {"generator_alone_x_copy": tot("gen_err") / tot("copy"), "w_self": tot("w_self") / n,
                            "w_scroll_source": tot("w_scroll") / n, "w_generate": tot("w_generate") / n}
                out["onestep"][c][key] = res
        # rollout, teval's windows
        fut, ca, fa = cache["fut"], cache["ctx_a"], cache["fut_a"]
        alive = ~meta["future_dead"][:, 0].cumsum(1).bool()
        gerr = torch.empty(R, T.H, 81)
        for i in range(0, R, 16):
            ctx = cache["ctx"][i:i + 16].float()
            frames, hist = [ctx[:, j] for j in range(4)], [ca[i:i + 16, j] for j in range(3)]
            for k in range(T.H):
                w_ = 4 if k == 0 else 5
                a = torch.stack(hist[-(w_ - 1):] + [fa[i:i + 16, k]], 1)
                g = T.step(world, torch.stack(frames[-w_:], 1), a, device, config)
                gerr[i:i + 16, k] = ((g - fut[i:i + 16, k].float()) ** 2).sum(-1)
                frames.append(g); hist.append(fa[i:i + 16, k])
        var = ((fut.float() - fut.float().flatten(0, 1).mean(0)) ** 2).sum(-1).mean((0, 1))     # [81] per-token variance
        out["rollout"] = {}
        for k in (1, 2, 4, 8, 16):
            m = alive[:, k - 1]
            e = gerr[m, k - 1]
            out["rollout"][k] = {g: {"over_group_V": float(e[:, tok].sum(-1).mean() / var[tok].sum()),
                                     "share_of_error": float(e[:, tok].sum() / e.sum())} for g, tok in GROUPS.items()}
        name = st["name"]
        (HERE / f"where_{name}.json").write_text(json.dumps(out, indent=2) + "\n")
        print(name, json.dumps({c: {g: {k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()}
                                    for g, r in row.items()} for c, row in out["onestep"].items()}), flush=True)
        print(name, "rollout", json.dumps({k: {g: round(v["over_group_V"], 3) for g, v in r.items()}
                                           for k, r in out["rollout"].items()}), flush=True)


if __name__ == "__main__":
    main()
