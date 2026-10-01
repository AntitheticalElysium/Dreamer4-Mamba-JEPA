"""E10 stage 3. The H16 decision panel for per-tile worlds: action-dependent death to depth 16 under a fixed continuation.

Data: deeppanel.py stage-2 rows. fit = FIT-train seeds (replayed), dev = FIT-dev seeds (replayed; selection only),
judge = the fresh block 62,000-62,399 (collect mode). Kept: roots whose P varies over the 17 first actions at some
judged depth.
Continuation: `recorded` (the BC policy's next 15 actions, open loop, identical across the 17 branches).
Labels: P(dead by k | first action) over 32 key sequences, k in DEPTHS = (1, 4, 16).
Tokens: layer-normed patch tokens of the frozen Raw encoder (dpanel.py's), for the root's last 4 frames and the real
depth-k frames of every branch (key sequence 0: ONE draw, while the labels average 32).
Imagination (teval's convention): step 1 from the 4 context frames and the first action; steps 2..16 from the last
`window` (5) frames and the continuation's actions; imagined states kept at DEPTHS.
Heads: dpanel.fit_head (spatial_why4's per-branch attention probe, soft_rank, 3,000 updates, dev selection), 3 seeds:
  gen_k        fitted and read on the world's imagined depth-k states
  transfer_k   the real_k head read on imagined depth-k states (what imagination training with real-fitted heads sees)
References (world-independent): real_k (real depth-k tiles, hindsight), root_rank_k (root tiles + first action; it does
NOT see the continuation), prior_k (the FIT action prior).
Fidelity: squared error of imagined vs real depth-k tokens over copying the root, per depth (the real frame is one
draw, so stochastic mobs set a floor above zero).
Judged: expected safe = 1 - P(dead by k | argmin risk) on judge roots with opportunity at k, per root averaged over head
seeds; paired episode-seed-clustered 95% intervals (1,000 draws); zombie / lava / night strata.
Usage: deepeval.py <world.pt> ... [--window 5] -> evals/deep_<name>.json;  deepeval.py --compare A:B ...
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE))
import dpanel as D  # noqa: E402  (sys.path for teval / the readout ladder is set there)
T = D.T

CACHE = ROOT / "artifacts/eda/deepeval_v1"
STORES = {"fit": ROOT / "artifacts/eda/deeppanel_fit_v1", "dev": ROOT / "artifacts/eda/deeppanel_dev_v1",
          "judge": ROOT / "artifacts/eda/deeppanel_judge_v1"}
N, SEEDS, DEPTHS, STORED = 17, 3, (1, 4, 16), (1, 2, 4, 8, 16)


def token_cache(device, log):
    """Per split: ctx [R,4,81,192] and real_k [R,17,81,192] fp16 memmaps; meta (acts, continuation, P_k, seed, visible)."""
    import spatial as S
    CACHE.mkdir(parents=True, exist_ok=True)
    encoder, out = None, {}
    for split, store in STORES.items():
        meta_path = CACHE / f"{split}_meta.pt"
        if not meta_path.exists():
            files = sorted(store.glob("seed-*.pt"))
            judged = lambda r: r["p_dead_by"]["recorded"][:, [k - 1 for k in DEPTHS]]                   # [17, 3]
            kept = lambda rows: [r for r in rows if bool((judged(r).amax(0) > judged(r).amin(0)).any())]
            R = sum(len(kept(torch.load(f, weights_only=False))) for f in files)      # pass 1: count (frames do not fit in RAM)
            encoder = encoder or S.bridge()[0]
            ctx = D.memmap(CACHE / f"{split}_ctx.f16", (R, 4, 81, 192), "w+")
            real = {k: D.memmap(CACHE / f"{split}_real{k}.f16", (R, N, 81, 192), "w+") for k in DEPTHS}
            meta, i = {k: [] for k in ("seed", "acts", "cont", "visible", "p")}, 0
            for f in files:                                                           # pass 2: encode file by file
                rows = kept(torch.load(f, weights_only=False))
                for j in range(0, len(rows), 8):
                    chunk = rows[j:j + 8]
                    b = len(chunk)
                    ctx[i:i + b] = S.encode(encoder, torch.stack([r["frames"][-4:] for r in chunk]), device)[1].numpy()
                    for k in DEPTHS:
                        frames = torch.stack([r["depth_frames"][:, STORED.index(k)] for r in chunk])  # [b,17,63,63,3]
                        real[k][i:i + b] = S.encode(encoder, frames, device)[1].numpy()
                    i += b
                for r in rows:
                    meta["seed"].append(r["seed"]); meta["acts"].append(r["led_to_action"][-3:])
                    meta["cont"].append(r["recorded"]); meta["visible"].append(r["visible"]); meta["p"].append(judged(r))
            if i != R:
                raise SystemExit(f"{split}: encoded {i} of {R} kept roots")
            ctx.flush(); [m.flush() for m in real.values()]
            p = torch.stack(meta.pop("p"))
            meta = {"seed": torch.tensor(meta["seed"]), **{k: torch.stack(v) for k, v in meta.items() if k != "seed"},
                    **{f"p{k}": p[:, :, j] for j, k in enumerate(DEPTHS)}}
            torch.save(meta, meta_path.with_suffix(".tmp"))
            meta_path.with_suffix(".tmp").replace(meta_path)
            log(stage="tokens", split=split, roots=R)
        meta = torch.load(meta_path)
        R = len(meta["seed"])
        out[split] = meta | {"ctx": torch.from_numpy(D.memmap(CACHE / f"{split}_ctx.f16", (R, 4, 81, 192)))} | {
            f"real{k}": torch.from_numpy(D.memmap(CACHE / f"{split}_real{k}.f16", (R, N, 81, 192))) for k in DEPTHS}
    return out


@torch.no_grad()
def imagine(world, config, data, window, path, device, batch=None):
    """Imagined states at DEPTHS for every root and first action, continuation actions after step 1 -> memmaps."""
    R = len(data["seed"])
    batch = batch or (16 if world.backbone_kind == "full" else 4)      # per-token SSM scans: where.py's OOM rule
    mms = {k: D.memmap(f"{path}_gen{k}.f16", (R, N, 81, 192), "w+") for k in DEPTHS}
    for i in range(0, R, batch):
        frames = data["ctx"][i:i + batch].float().repeat_interleave(N, 0)
        b = len(frames) // N
        acts = torch.cat([data["acts"][i:i + batch].repeat_interleave(N, 0), torch.arange(N).repeat(b)[:, None]], 1)
        cont = data["cont"][i:i + batch].repeat_interleave(N, 0)
        for k in range(1, 17):
            nxt = T.step(world, frames[:, -window:] if k > 1 else frames, acts[:, -(window if k > 1 else 4):], device, config)
            if k in DEPTHS:
                mms[k][i:i + b] = nxt.view(b, N, 81, 192).half().numpy()
            if k < 16:
                frames = torch.cat([frames, nxt[:, None]], 1)[:, -window:]
                acts = torch.cat([acts, cont[:, k - 1:k]], 1)[:, -window:]
    for m in mms.values():
        m.flush()
    return {k: torch.from_numpy(m) for k, m in mms.items()}


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("worlds", nargs="*", type=Path)
    parser.add_argument("--window", type=int, default=5)
    parser.add_argument("--compare", nargs="*", default=[])
    args = parser.parse_args(argv)
    started = time.time()
    log = lambda **kw: print(json.dumps({**kw, "seconds": round(time.time() - started, 1)}), flush=True)
    out_dir = HERE / "evals"
    from ladder import paired
    if args.compare:
        res = {}
        for pair in args.compare:
            a, b = pair.split(":")
            ra, rb = (torch.load(out_dir / f"deep_{x}_rows.pt") for x in (a, b))
            if not torch.equal(ra["seed"], rb["seed"]):
                raise SystemExit("the two worlds were judged on different roots")
            for arm in [f"{x}{k}" for k in DEPTHS for x in ("gen", "transfer")]:
                k = int(arm.lstrip("gentransfer"))
                for stratum, m in (("overall", ra[f"opp{k}"]), ("zombie_adjacent", ra[f"opp{k}"] & ra["zombie"])):
                    res[f"{pair}:{arm}@{stratum}"] = paired(ra[arm][m], rb[arm][m], ra["seed"][m], draws=1000, seed=20261001)
                    log(pair=pair, arm=arm, stratum=stratum, **res[f"{pair}:{arm}@{stratum}"])
        (out_dir / "deep_compare.json").write_text(json.dumps(res, indent=2) + "\n")
        return
    from d4mj.config import config_from_dict
    from frozen_ladder import strata
    from observability import expected_safe
    import spatial as S
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = token_cache(device, log)
    fit, dev, judge = data["fit"], data["dev"], data["judge"]
    log(stage="data", **{s: len(d["seed"]) for s, d in data.items()})
    rows_j = torch.arange(len(judge["seed"]))
    opp = {k: judge[f"p{k}"].amax(1) > judge[f"p{k}"].amin(1) for k in DEPTHS}
    (CACHE / "heads").mkdir(exist_ok=True)
    safe, per_seed, real_heads = {}, {}, {}
    for k in DEPTHS:                                             # world-independent references
        pf, pd, pj = fit[f"p{k}"], dev[f"p{k}"], judge[f"p{k}"]
        prior = int(pf[pf.amax(1) > pf.amin(1)].mean(0).argmin())
        safe[f"prior{k}"] = expected_safe(-F.one_hot(torch.full((len(pj),), prior), N).float(), pj)[0]
        per_seed[f"prior{k}"] = [float(safe[f"prior{k}"][opp[k]].mean())]
        for name, key in (("root_rank", None), ("real", f"real{k}")):
            xs = [s["ctx"][:, -1] if key is None else s[key] for s in (fit, dev, judge)]
            runs = []
            for seed in range(SEEDS):
                model, _ = D.fit_head(xs[0], pf, xs[1], pd, device, seed, CACHE / "heads" / f"{name}{k}_s{seed}.pt")
                runs.append(expected_safe(D.judge_risk(model, xs[2], rows_j, device), pj)[0])
                if name == "real":
                    real_heads.setdefault(k, []).append(model)
            safe[f"{name}{k}"], per_seed[f"{name}{k}"] = torch.stack(runs).mean(0), [float(r[opp[k]].mean()) for r in runs]
            log(ref=f"{name}{k}", safe=[round(v, 4) for v in per_seed[f"{name}{k}"]])
    config = config_from_dict(torch.load(S.CHECKPOINT, map_location="cpu", weights_only=False)["config"])
    strat = strata(judge["visible"])
    seeds = judge["seed"]
    for path in args.worlds:
        world, st = T.load_world(path, device)
        name = st["name"] + ("" if args.window == 5 else f"__w{args.window}")
        target = out_dir / f"deep_{name}.json"
        if target.exists():
            log(world=name, status="exists")
            continue
        tag = CACHE / f"world_{name}"
        gen = {split: imagine(world, config, data[split], args.window, f"{tag}_{split}", device) for split in data}
        fidelity = {}
        for k in DEPTHS:
            e = c = 0.0
            for i in range(0, len(rows_j), 64):
                real, root = judge[f"real{k}"][i:i + 64].float(), judge["ctx"][i:i + 64, -1:].float()
                e += float(((gen["judge"][k][i:i + 64].float() - real) ** 2).sum((-1, -2)).sum())
                c += float(((root - real) ** 2).sum((-1, -2)).sum())
            fidelity[f"err_over_copy{k}"] = e / c
        log(world=name, stage="imagined", **fidelity)
        w_safe, w_seed, traces = dict(safe), dict(per_seed), {}
        for k in DEPTHS:
            runs = []
            for seed in range(SEEDS):
                model, trace = D.fit_head(gen["fit"][k], fit[f"p{k}"], gen["dev"][k], dev[f"p{k}"], device, seed,
                                          Path(f"{tag}_gen{k}_s{seed}.pt"))
                runs.append(expected_safe(D.judge_risk(model, gen["judge"][k], rows_j, device), judge[f"p{k}"])[0])
                traces.setdefault(f"gen{k}", []).append(trace)
            w_safe[f"gen{k}"], w_seed[f"gen{k}"] = torch.stack(runs).mean(0), [float(r[opp[k]].mean()) for r in runs]
            runs = [expected_safe(D.judge_risk(m, gen["judge"][k], rows_j, device), judge[f"p{k}"])[0] for m in real_heads[k]]
            w_safe[f"transfer{k}"], w_seed[f"transfer{k}"] = torch.stack(runs).mean(0), [float(r[opp[k]].mean()) for r in runs]
            log(world=name, depth=k, gen=[round(v, 4) for v in w_seed[f"gen{k}"]], transfer=[round(v, 4) for v in w_seed[f"transfer{k}"]])
        res = {"world": name, "world_path": str(path), "window": args.window, "fidelity": fidelity, "traces": traces,
               "roots": {s: len(d["seed"]) for s, d in data.items()}, "per_seed": w_seed, "expected_safe": {}, "contrasts": {}}
        for k in DEPTHS:
            masks = {"overall": opp[k]} | {s: opp[k] & strat[s] for s in ("zombie_adjacent", "lava_adjacent", "night", "day")}
            res[f"roots_k{k}"] = {s: int(m.sum()) for s, m in masks.items()}
            for arm in (f"gen{k}", f"transfer{k}", f"real{k}", f"root_rank{k}", f"prior{k}"):
                res["expected_safe"][arm] = {s: float(w_safe[arm][m].mean()) for s, m in masks.items() if m.any()}
            for a, b in ((f"gen{k}", f"prior{k}"), (f"gen{k}", f"root_rank{k}"), (f"gen{k}", f"real{k}"),
                         (f"transfer{k}", f"prior{k}"), (f"gen{k}", f"transfer{k}")):
                for s in ("overall", "zombie_adjacent"):
                    m = masks[s]
                    res["contrasts"][f"{a}-{b}@{s}"] = paired(w_safe[a][m], w_safe[b][m], seeds[m], draws=1000, seed=20261001)
        torch.save({"seed": seeds, "zombie": strat["zombie_adjacent"], **{f"opp{k}": opp[k] for k in DEPTHS}, **w_safe},
                   out_dir / f"deep_{name}_rows.pt")
        target.write_text(json.dumps(res, indent=2) + "\n")
        for f in CACHE.glob(f"world_{name}_*"):
            f.unlink()
        log(world=name, status="done")
        del world
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
