"""E9. Decision panel: do the per-tile worlds' imagined successors carry action-dependent death at H1 and H2?

Why: the diagnosis futures (teval's 1,002 roots) hold 1 opportunity root in TRAIN and 0 in TEST, so no per-tile world
has been judged on a decision. The readout ladder's blocks carry them (observe.py: damaging-choice retention, 32-frame
history, all 17 first actions, the stored NOOP second step, P(death1) and P(death2) = P(death within two steps) over
32 key pairs).

Data (the readout-ladder partition, split as frozen_heads.py):
  fit    FIT-train seeds (700): fork-store pixels joined to observe_fit_v1's 32-key P by (seed, step)
  dev    FIT-dev seeds (350): fork-store rows, realized single-key death1 = terminated, death2 = terminated |
         second_terminated; head selection only
  judge  --blocks, default observe_fresh_v6 + v7 (seeds 55,000-56,423): OPENED blocks (read by earlier tests), so
         everything here is EXPLORATORY; a sealed confirmation needs a new block
  Kept: roots whose P (dev: realized death) varies over the 17 actions at H1 or H2 -- soft_rank and expected safe read
  nothing else.
Tokens: the frozen Raw encoder's layer-normed patch tokens (raw bridge step 2,000 encoder = joint step 10,000's, bit
for bit: the levers Raw token space), fp16, cached once under artifacts/eda/dpanel_v1.
Imagination (teval's convention): step 1 from the 4 context frames and the branch action; step 2 from the last
`--window` frames (default 5: the 4 context frames and step 1) and NOOP, the collector's second step.
Heads: frozen_heads.train_head's ranking arms (spatial_why4's attention probe per branch, soft_rank over 16 roots per
update, 3,000 updates, AdamW 1e-3 / 1e-4, selection every 100 on dev expected safe), three head seeds:
  gen1, gen2    fitted and read on the world's imagined step-1 / step-2 states
  gen1_h2       the imagined step-1 state fitted on P(death2): what the second imagined step adds
  transfer1/2   the REAL-successor head (real1 / real2) read on imagined states: what an actor trained in imagination
                with heads fitted on real data would see
World-independent references: real1 (real successor tiles), real2 (real step-2 tiles; the step-1 frame where step 1
ends the episode), root_rank (root tiles + action embedding: no transition), root_tokens (frozen_ladder tokens_attn on
root tiles -> 17), actions_only (last 4 actions -> 17), prior (the FIT action prior: one fixed action).
Judged: expected safe = 1 - P(death_h | argmin risk) on judge opportunity roots at h, per root averaged over head seeds;
paired episode-seed-clustered 95% intervals (1,000 draws); zombie / lava / night strata. Reported, not ruled.
Usage: dpanel.py <world.pt> ... [--window 5] [--blocks observe_fresh_v6 observe_fresh_v7] -> evals/dpanel_<name>.json
       dpanel.py --compare A:B ...   paired contrasts between two worlds' judged rows
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
LADDER = ROOT / "artifacts/experiments/20260921_readout_ladder"
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(HERE)); sys.path.insert(0, str(LADDER))
import teval as T  # noqa: E402

CACHE = ROOT / "artifacts/eda/dpanel_v1"
N, SEEDS, STEPS = 17, 3, 3000
HEADS = ("gen1", "gen2", "gen1_h2", "transfer1", "transfer2")
REFS = ("prior", "actions_only", "root_tokens", "root_rank", "real")


def load_rows(split, seeds, stores):
    """Opportunity roots of one split: 4 context frames, 17 successors, 17 step-2 frames, actions, P, strata state."""
    from d4mj.lewm_diagnostics import FORK_STORE
    from observability import FIT_STATE
    pixels = {int(p.stem.split("-")[1]): p for p in FORK_STORE.glob("seed-*.pt")}
    if split == "judge":
        files = [(f, None) for s in stores for f in sorted((ROOT / "artifacts/eda" / s).glob("seed-*.pt"))]
    else:
        state = {int(p.stem.split("-")[1]): p for p in FIT_STATE.glob("seed-*.pt")}
        files = [(pixels[s], state[s] if split == "fit" else None) for s in sorted(seeds)]
    out = {k: [] for k in ("seed", "frames", "acts", "succ", "second", "p1", "p2", "visible")}
    for pixel_file, state_file in files:
        rows = torch.load(pixel_file, weights_only=False)
        labels = {int(r["step"]): r for r in torch.load(state_file, weights_only=False)} if state_file else None
        for r in rows:
            if len(r["frames"]) < 4 or bool((r["led_to_action"][-4:] >= N).any()):
                continue
            if split == "judge" or labels is not None:
                lab = r if labels is None else labels.get(int(r["step"]))
                if lab is None:
                    continue
                p1, p2, visible = lab["p_death1"].float(), lab["p_death2"].float(), lab["visible"].float()
            else:
                p1 = r["terminated"].float()
                p2 = (r["terminated"] | r["second_terminated"]).float()
                visible = torch.zeros(0)
            if not (p1.max() > p1.min() or p2.max() > p2.min()):
                continue
            second = torch.where(r["second_valid"][:, None, None, None], r["second"], r["successors"])
            for k, v in (("seed", int(r["seed"])), ("frames", r["frames"][-4:]), ("acts", r["led_to_action"][-4:]),
                         ("succ", r["successors"]), ("second", second), ("p1", p1), ("p2", p2), ("visible", visible)):
                out[k].append(v)
    return {k: torch.tensor(v) if k == "seed" else torch.stack(v) for k, v in out.items()}


def memmap(path, shape, mode="c"):
    """fp16 file-backed array (default copy-on-write: readable by torch, never written back); "w+" writes, flush."""
    return np.memmap(path, dtype=np.float16, mode=mode, shape=tuple(shape))


def token_cache(splits, device, log):
    """Per split: ctx [R,4,81,192], real1 / real2 [R,17,81,192] (fp16 memmaps) and meta (actions, P, seeds)."""
    import spatial as S
    CACHE.mkdir(parents=True, exist_ok=True)
    encoder = None
    out = {}
    for split, (seeds, stores) in splits.items():
        meta_path = CACHE / f"{split}_meta.pt"
        if not meta_path.exists():
            data = load_rows(split, seeds, stores)
            encoder = encoder or S.bridge()[0]
            R = len(data["seed"])
            for key, src in (("ctx", "frames"), ("real1", "succ"), ("real2", "second")):
                t = data[src].shape[1]
                mm = memmap(CACHE / f"{split}_{key}.f16", (R, t, 81, 192), "w+")
                for i in range(0, R, 8):
                    mm[i:i + 8] = S.encode(encoder, data[src][i:i + 8], device)[1].numpy()
                mm.flush()
                del mm
            meta = {k: data[k] for k in ("seed", "acts", "p1", "p2", "visible")}
            torch.save(meta, meta_path.with_suffix(".tmp"))
            meta_path.with_suffix(".tmp").replace(meta_path)
            log(stage="tokens", split=split, roots=R)
        meta = torch.load(meta_path)
        R = len(meta["seed"])
        out[split] = meta | {k: torch.from_numpy(memmap(CACHE / f"{split}_{k}.f16", (R, t, 81, 192)))
                             for k, t in (("ctx", 4), ("real1", N), ("real2", N))}
    return out


@torch.no_grad()
def imagine(world, config, data, window, path, device, batch=16):
    """The world's step-1 (branch action) and step-2 (NOOP) states for every root and branch -> fp16 memmaps."""
    R = len(data["seed"])
    m1, m2 = memmap(f"{path}_gen1.f16", (R, N, 81, 192), "w+"), memmap(f"{path}_gen2.f16", (R, N, 81, 192), "w+")
    g1, g2 = torch.from_numpy(m1), torch.from_numpy(m2)
    for i in range(0, R, batch):
        fan = data["ctx"][i:i + batch].float().repeat_interleave(N, 0)
        b = len(fan) // N
        past = data["acts"][i:i + batch, 1:].repeat_interleave(N, 0)
        a = torch.arange(N).repeat(b)[:, None]
        one = T.step(world, fan, torch.cat([past, a], 1), device, config)
        frames = torch.cat([fan, one[:, None]], 1)[:, -window:]
        two = T.step(world, frames, torch.cat([past, a, torch.zeros_like(a)], 1)[:, -window:], device, config)
        g1[i:i + b], g2[i:i + b] = one.view(b, N, 81, 192).half(), two.view(b, N, 81, 192).half()
    m1.flush(); m2.flush()
    return g1, g2


def risk(model, x, rows, device):
    """[len(rows), 17] risk: x [R,17,81,192] per-branch states, or [R,81,192] root tiles + the action embedding."""
    b = len(rows)
    if x.dim() == 3:
        tiles = x[rows].float().to(device).repeat_interleave(N, 0)
        return model(tiles, torch.arange(N, device=device).repeat(b)).view(b, N)
    return model(x[rows].flatten(0, 1).float().to(device)).view(b, N)


@torch.no_grad()
def judge_risk(model, x, rows, device):
    return torch.cat([risk(model, x, rows[i:i + 64], device) for i in range(0, len(rows), 64)]).cpu()


def fit_head(x, p, dev_x, p_dev, device, seed, path):
    """frozen_heads.train_head's ranking arms, returning the selected model (saved to `path`, reused if present)."""
    from observability import expected_safe, soft_rank
    from spatial_why4 import Probe
    torch.manual_seed(seed)
    model = Probe(192, x.dim() == 3).to(device)
    if path.exists():
        saved = torch.load(path)
        model.load_state_dict(saved["state"])
        return model.eval(), saved["trace"]
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    gen = torch.Generator().manual_seed(seed)
    dev_rows = torch.where(p_dev.amax(1) > p_dev.amin(1))[0]
    best, state, curve = -1.0, None, []
    for step in range(STEPS):
        rows = torch.randint(len(p), (16,), generator=gen)
        loss = soft_rank(risk(model, x, rows, device), p[rows].to(device))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if (step + 1) % 100 == 0 or step + 1 == STEPS:
            model.eval()
            value = float(expected_safe(judge_risk(model, dev_x, dev_rows, device), p_dev[dev_rows])[0].mean())
            model.train()
            curve.append(round(value, 4))
            if value > best or state is None:
                best, state, chosen = value, {k: v.detach().clone() for k, v in model.state_dict().items()}, step + 1
    model.load_state_dict(state)
    trace = {"selected_step": chosen, "dev_safe": best, "curve": curve}
    torch.save({"state": state, "trace": trace}, path)
    return model.eval(), trace


def references(data, device, log):
    """World-independent arms per horizon: per-root judged safe (seed mean), per-seed means, the real heads."""
    from frozen_ladder import scores, standardize, train as ladder_train
    from observability import expected_safe
    fit, dev, judge = data["fit"], data["dev"], data["judge"]
    (CACHE / "heads").mkdir(exist_ok=True)
    safe, per_seed, real_heads, chosen = {}, {}, {}, {}
    rows_j = torch.arange(len(judge["seed"]))
    for h in (1, 2):
        pf, pd, pj = fit[f"p{h}"], dev[f"p{h}"], judge[f"p{h}"]
        opp = pj.amax(1) > pj.amin(1)
        both = torch.cat([pf, pd])
        train_rows, hold_rows = torch.arange(len(pf)), torch.arange(len(pf), len(both))
        prior = int(pf[pf.amax(1) > pf.amin(1)].mean(0).argmin())
        safe[f"prior{h}"] = expected_safe(-F.one_hot(torch.full((len(pj),), prior), N).float(), pj)[0]
        per_seed[f"prior{h}"], chosen[f"prior{h}"] = [float(safe[f"prior{h}"][opp].mean())], prior
        tiles = lambda s: s["ctx"][:, -1].float()
        for name, kind, xf, xj in (
                ("actions_only", "vector", F.one_hot(torch.cat([fit["acts"], dev["acts"]]), N).flatten(1).float(),
                 F.one_hot(judge["acts"], N).flatten(1).float()),
                ("root_tokens", "tokens_attn", torch.cat([tiles(fit), tiles(dev)]), tiles(judge))):
            mean, scale = standardize(xf, train_rows)
            xf, xj = (xf - mean) / scale, (xj - mean) / scale
            runs = []
            for seed in range(SEEDS):
                model, _ = ladder_train(kind, xf.shape[1:], xf, both, train_rows, hold_rows, seed=seed,
                                        device=device, steps=STEPS)
                runs.append(expected_safe(scores(model, xj, rows_j, device), pj)[0])
            safe[f"{name}{h}"] = torch.stack(runs).mean(0)
            per_seed[f"{name}{h}"] = [float(r[opp].mean()) for r in runs]
            log(ref=f"{name}{h}", safe=round(per_seed[f"{name}{h}"][0], 4))
        for name, key in (("root_rank", None), ("real", f"real{h}")):
            xf = fit["ctx"][:, -1] if key is None else fit[key]
            xd = dev["ctx"][:, -1] if key is None else dev[key]
            xj = judge["ctx"][:, -1] if key is None else judge[key]
            runs = []
            for seed in range(SEEDS):
                model, _ = fit_head(xf, pf, xd, pd, device, seed, CACHE / "heads" / f"{name}{h}_s{seed}.pt")
                runs.append(expected_safe(judge_risk(model, xj, rows_j, device), pj)[0])
                if name == "real":
                    real_heads.setdefault(h, []).append(model)
            safe[f"{name}{h}"] = torch.stack(runs).mean(0)
            per_seed[f"{name}{h}"] = [float(r[opp].mean()) for r in runs]
            log(ref=f"{name}{h}", safe=[round(v, 4) for v in per_seed[f"{name}{h}"]])
    return safe, per_seed, real_heads, chosen


def summarize(safe, per_seed, judge, contrasts):
    from frozen_ladder import strata
    from ladder import paired
    strat = strata(judge["visible"])
    seeds = judge["seed"]
    out = {"expected_safe": {}, "per_seed": per_seed, "contrasts": {}}
    for h in (1, 2):
        pj = judge[f"p{h}"]
        opp = pj.amax(1) > pj.amin(1)
        masks = {"overall": opp} | {k: opp & strat[k] for k in ("zombie_adjacent", "lava_adjacent", "night", "day")}
        out["roots_h%d" % h] = {k: int(m.sum()) for k, m in masks.items()}
        for arm, v in safe.items():
            if arm.endswith(str(h)):
                out["expected_safe"][arm] = {k: float(v[m].mean()) for k, m in masks.items() if m.any()}
        for a, b in contrasts[h]:
            for stratum in ("overall", "zombie_adjacent"):
                m = masks[stratum]
                out["contrasts"][f"{a}-{b}@{stratum}"] = paired(safe[a][m], safe[b][m], seeds[m], draws=1000, seed=20261001)
    return out


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("worlds", nargs="*", type=Path)
    parser.add_argument("--window", type=int, default=5)
    parser.add_argument("--blocks", nargs="+", default=["observe_fresh_v6", "observe_fresh_v7"])
    parser.add_argument("--compare", nargs="*", default=[])
    args = parser.parse_args(argv)
    started = time.time()
    log = lambda **kw: print(json.dumps({**kw, "seconds": round(time.time() - started, 1)}), flush=True)
    out_dir = HERE / "evals"
    if args.compare:
        from ladder import paired
        res = {}
        for pair in args.compare:
            a, b = pair.split(":")
            ra, rb = (torch.load(out_dir / f"dpanel_{x}_rows.pt") for x in (a, b))
            if not torch.equal(ra["seed"], rb["seed"]):
                raise SystemExit("the two worlds were judged on different roots")
            for arm in ("gen1", "gen2", "transfer1", "transfer2"):
                h = int(arm[-1])
                for stratum, m in (("overall", ra[f"opp{h}"]), ("zombie_adjacent", ra[f"opp{h}"] & ra["zombie"])):
                    res[f"{pair}:{arm}@{stratum}"] = paired(ra[arm][m], rb[arm][m], ra["seed"][m], draws=1000,
                                                            seed=20261001)
                    log(pair=pair, arm=arm, stratum=stratum, **{k: v for k, v in res[f"{pair}:{arm}@{stratum}"].items()})
        (out_dir / "dpanel_compare.json").write_text(json.dumps(res, indent=2) + "\n")
        return
    from d4mj.config import config_from_dict
    from confirm import seeds_for
    from d4mj.lewm_diagnostics import FORK_STORE
    from observability import expected_safe
    from frozen_ladder import strata
    import spatial as S
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    partition = json.loads((LADDER / "evidence/root_partition.json").read_text())
    fit_seeds, unallocated = seeds_for(partition, FORK_STORE)
    dev_seeds = sorted(partition["fit_dev"]["seeds"])
    if (set(partition["reserved_for_gate"]["seeds"]) | set(unallocated)) & (set(fit_seeds) | set(dev_seeds)):
        raise SystemExit("partition overlap")
    blocks_tag = "_".join(Path(b).name.replace("observe_fresh_", "") for b in args.blocks)
    if blocks_tag != "v6_v7":
        global CACHE
        CACHE = CACHE.with_name(f"{CACHE.name}_{blocks_tag}")
    data = token_cache({"fit": (fit_seeds, None), "dev": (dev_seeds, None), "judge": (None, args.blocks)}, device, log)
    log(stage="data", **{s: len(d["seed"]) for s, d in data.items()})
    safe_ref, seed_ref, real_heads, prior = references(data, device, log)
    config = config_from_dict(torch.load(S.CHECKPOINT, map_location="cpu", weights_only=False)["config"])
    judge = data["judge"]
    rows_j = torch.arange(len(judge["seed"]))
    for path in args.worlds:
        world, st = T.load_world(path, device)
        name = st["name"] + ("" if args.window == 5 else f"__w{args.window}")
        target = out_dir / f"dpanel_{name}.json"
        if target.exists():
            log(world=name, status="exists")
            continue
        tag = CACHE / f"world_{name}"
        gen = {split: imagine(world, config, data[split], args.window, f"{tag}_{split}", device)
               for split in ("fit", "dev", "judge")}
        fidelity = {}
        for h in (1, 2):                      # token check: imagined vs real successor, over copying the root
            e = c = 0.0
            for i in range(0, len(rows_j), 64):
                real, root = judge[f"real{h}"][i:i + 64].float(), judge["ctx"][i:i + 64, -1:].float()
                e += float(((gen["judge"][h - 1][i:i + 64].float() - real) ** 2).sum((-1, -2)).sum())
                c += float(((root - real) ** 2).sum((-1, -2)).sum())
            fidelity[f"err_over_copy{h}"] = e / c
        log(world=name, stage="imagined", **fidelity)
        safe, per_seed = dict(safe_ref), dict(seed_ref)
        traces, chosen = {}, {}
        opp = {h: judge[f"p{h}"].amax(1) > judge[f"p{h}"].amin(1) for h in (1, 2)}
        for arm, x, h in (("gen1", 0, 1), ("gen2", 1, 2), ("gen1_h2", 0, 2)):
            runs = []
            for seed in range(SEEDS):
                model, trace = fit_head(gen["fit"][x], data["fit"][f"p{h}"], gen["dev"][x], data["dev"][f"p{h}"],
                                        device, seed, Path(f"{tag}_{arm}_s{seed}.pt"))
                r = judge_risk(model, gen["judge"][x], rows_j, device)
                runs.append(expected_safe(r, judge[f"p{h}"])[0])
                chosen[arm] = chosen.get(arm, 0) + np.bincount(r.argmin(1)[opp[h]].numpy(), minlength=N)
                traces.setdefault(arm, []).append(trace)
            safe[arm], per_seed[arm] = torch.stack(runs).mean(0), [float(r[opp[h]].mean()) for r in runs]
            log(world=name, arm=arm, safe=[round(v, 4) for v in per_seed[arm]])
        for h in (1, 2):
            runs = [expected_safe(judge_risk(m, gen["judge"][h - 1], rows_j, device), judge[f"p{h}"])[0]
                    for m in real_heads[h]]
            safe[f"transfer{h}"], per_seed[f"transfer{h}"] = torch.stack(runs).mean(0), [float(r[opp[h]].mean()) for r in runs]
            log(world=name, arm=f"transfer{h}", safe=[round(v, 4) for v in per_seed[f"transfer{h}"]])
        contrasts = {h: [(f"gen{h}", f"{r}{h}") for r in REFS] + [(f"transfer{h}", f"prior{h}"),
                     (f"transfer{h}", f"real{h}"), (f"gen{h}", f"transfer{h}")] for h in (1, 2)}
        contrasts[2].append(("gen2", "gen1_h2"))
        res = {"world": name, "world_path": str(path), "window": args.window, "blocks": args.blocks,
               "roots": {s: len(d["seed"]) for s, d in data.items()}, "prior_action": prior, "fidelity": fidelity,
               "chosen_histogram_opportunity": {a: c.tolist() for a, c in chosen.items()},
               "traces": traces} | summarize(safe, per_seed, judge, contrasts)
        strat = strata(judge["visible"])
        torch.save({"seed": judge["seed"], "opp1": judge["p1"].amax(1) > judge["p1"].amin(1),
                    "opp2": judge["p2"].amax(1) > judge["p2"].amin(1), "zombie": strat["zombie_adjacent"],
                    **{k: v for k, v in safe.items()}}, out_dir / f"dpanel_{name}_rows.pt")
        target.write_text(json.dumps(res, indent=2) + "\n")
        for f in CACHE.glob(f"world_{name}_*"):
            f.unlink()
        log(world=name, status="done")
        del world
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
