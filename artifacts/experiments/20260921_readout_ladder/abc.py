"""A / B / C: does repaired terminal exposure, then a reweighted loss, recover useful generated states?

DIAGNOSE.md: the old u->u world (A) predicts 95.5% of the within-root action effect but not the fatal
direction, and was never given a death as a target (its 13-frame window cannot reach the last 9
transitions of an episode). Three arms, identical in the frozen TC-consecutive encoder, its persisted
PCA, the world architecture, init seed 7, batch seed 11, AdamW from the old recipe, 10k updates,
batch 128 and a fixed 25,600-window pool -- A's own recipe (`state_transition.train_arm`):

  A  the existing world: old 13-frame layout, no terminal target, plain MSE        (reference)
  B  a terminal-reaching pool: 95% uniform 4-frame windows from the same corpus's TRAIN split,
     5% death-ending windows each from a DISTINCT terminal episode; plain MSE       (exposure)
  C  exactly B's pool and batch order; weighted MSE with per-component weights
     proportional to TRAIN-variance^(-1/2), normalized to mean one                  (loss)

A versus B tests a combined terminal-access / sampling repair; B versus C tests the loss cleanly.
C's weights are bounded by construction: inverse square root, not inverse variance, whose 13,600x
dynamic range would all but delete the leading components that carry ordinary movement.

Declared rules, committed before any arm is trained:

  Stage 1 -- exploratory panel (the observability roots): train B, score A and B.
    B largely closes the gap  :=  B - A generated-fitted safe choice >= half of (root u - A), with
                                  the paired seed-clustered interval above zero
    -> if so, C is not trained; otherwise C is trained on B's exact pool and batches.

  Stage 2 -- sealed panel: seeds 52,000+ (observe.py, stop at 800 one-step opportunity roots),
  collected only after this commit and scored once, on the finalized arms. For X in {B, C}:
    S1  X - A generated-fitted within-root safe choice > 0, paired interval above zero
    S2  the same on zombie-adjacent roots
    S3  retention: X's lava-stratum safe choice not resolved below A's, and X's within-root
        normalized action-effect error at most twice A's (ordinary movement is kept)
    X succeeds iff S1 and S2 and S3.
    B succeeds                          -> exposure_repair_helps
    C succeeds and C - B resolved > 0   -> loss_adds_beyond_exposure
    neither succeeds                    -> stop loss tweaking; spatial dynamics or uncertainty next
  Reported beside: root u, the action prior, the real successor, chosen-action histograms, SLEEP
  choices, and the fatal-direction alignment of each arm's generated u.
"""

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

ROOT = Path("/home/antithetical/EPITA/PERSO/DynamicHorizons-Mamba-JEPA")
HERE = Path(__file__).parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

from d4mj.data import _sha256
from d4mj.lewm_diagnostics import FORK_STORE

from compactness import OLD, OLD_CHECKPOINT, apply_pca, old_encoder  # noqa: E402

EVIDENCE = HERE / "evidence"
SUPPORT = ROOT / "artifacts/craftax_support_v2"
RECIPE = ROOT / "artifacts/lewm_gates_20260916/paired_window/raw/resolved_recipe.json"
A_WORLD = ROOT / "artifacts/experiments/20260918_matched_10k/evidence/world_u_u.pt"
SEALED = ROOT / "artifacts/eda/observe_fresh_v3"
POOL, TERMINAL_SHARE, POOL_SEED = 25_600, 0.05, 20260925
N = 17


def world_config():
    stored = torch.load(OLD_CHECKPOINT, map_location="cpu", weights_only=False)
    from d4mj.config import config_from_dict, load_recipe
    return config_from_dict(stored["config"]), load_recipe(RECIPE)


# ------------------------------------------------------------------------------------------ pool
@torch.no_grad()
def build_pool(device):
    from d4mj.data import load_joint_corpus
    _, recipe = world_config()
    episodes, _ = load_joint_corpus(SUPPORT, recipe)
    eligible = [e for e in episodes if e.split == "train" and e.uniform_eligible and len(e) + 1 >= 4]
    terminal = [e for e in eligible if bool(e.terminated[-1])]
    gen = torch.Generator().manual_seed(POOL_SEED)
    n_term = int(round(TERMINAL_SHARE * POOL))
    order = torch.randperm(len(terminal), generator=gen)[:n_term].tolist()      # each used once
    counts = torch.tensor([len(e) - 2 for e in eligible], dtype=torch.float64)  # JointSampler, span 4
    windows = []
    for i in torch.multinomial(counts, POOL - n_term, replacement=True, generator=gen).tolist():
        start = int(torch.randint(int(counts[i]), (), generator=gen))
        windows.append((eligible[i], start))
    windows += [(terminal[i], len(terminal[i]) - 3) for i in order]
    encoder = old_encoder(device)
    pca = torch.load(OLD / "evidence/state_cache.pt", weights_only=False, map_location="cpu")["pca"]
    u, actions, target_terminal, ids = [], [], [], []
    for b in range(0, len(windows), 64):
        chunk = windows[b:b + 64]
        frames = torch.stack([torch.as_tensor(np.asarray(e.observations[s:s + 4])) for e, s in chunk]).to(device)
        _, _, grid = encoder.export(frames, grid=4)
        u.append(apply_pca(pca, grid.flatten(2).cpu()))
        actions.append(torch.stack([torch.as_tensor(np.asarray(e.actions_taken[s:s + 3])).long() for e, s in chunk]))
        target_terminal.append(torch.tensor([[bool(e.terminated[s + k]) for k in range(3)] for e, s in chunk]))
        ids += [(e.episode_id, s) for e, s in chunk]
    u, actions, target_terminal = torch.cat(u), torch.cat(actions), torch.cat(target_terminal)
    nxt = u[:, 1:].reshape(-1, u.shape[-1])
    var = nxt.var(0)
    weights = var.rsqrt()
    weights = weights / weights.mean()
    w = torch.load(EVIDENCE / "critical_direction.pt", weights_only=False)["w"]
    share = lambda lam: float((lam * w.square() * var).sum() / (lam * var).sum())
    loss_share = {"formula": "share(lam) = sum_i lam_i w_i^2 var_i / sum_i lam_i var_i: the fatal direction's "
                             "expected share of the loss on a target with per-component variance var",
                  "plain": share(torch.ones_like(var)), "inverse_sqrt": share(weights),
                  "inverse_variance": share(1 / var)}
    meta = {"windows": len(ids), "terminal_windows_stratum": n_term, "terminal_episodes_available": len(terminal),
            "unique_terminal_targets": int(target_terminal.any(1).sum()),
            "natural_terminal_windows": int(target_terminal[:POOL - n_term].any(1).sum()),
            "unique_windows": len(set(ids)), "weight_range": float(weights.max() / weights.min()),
            "variance_range": float(var.max() / var.min()), "fatal_direction_loss_share": loss_share}
    torch.save({"u": u, "actions": actions, "target_terminal": target_terminal, "ids": ids,
                "weights": weights, "variance": var, "meta": meta}, EVIDENCE / "abc_pool.pt")
    return meta


# ----------------------------------------------------------------------------------------- train
def train_world(cache, *, steps, weights=None, device, init_seed=7, batch_seed=11):
    """`state_transition.train_arm`, verbatim but for optional per-component loss weights."""
    from d4mj.lewm import LeWMWorld
    _, recipe = world_config()
    # `train_arm` built the world from the resolved recipe, not the checkpoint config; mirror it.
    with torch.random.fork_rng(devices=list(range(torch.cuda.device_count()))):
        torch.manual_seed(init_seed)
        torch.cuda.manual_seed_all(init_seed)
        world = LeWMWorld(recipe).to(device)
    world.train()
    parameters = [p for p in world.parameters() if p.requires_grad]
    j = recipe.joint
    optimizer = torch.optim.AdamW(parameters, lr=j.learning_rate, betas=tuple(j.betas), eps=j.optimizer_eps,
                                  weight_decay=j.weight_decay)
    order = torch.Generator().manual_seed(batch_seed)
    n, batch = len(cache["u"]), j.batch
    lam = None if weights is None else weights.to(device)
    history, presented = [], 0
    for step in range(steps):
        index = torch.randint(n, (batch,), generator=order)
        if "target_terminal" in cache:
            presented += int(cache["target_terminal"][index].sum())
        src = cache["u"][index].to(device).unsqueeze(2)
        predicted = world.teacher(src, cache["actions"][index].to(device)).predicted
        err = (predicted.float() - src[:, 1:].float()).square()
        loss = err.mean() if lam is None else (err * lam).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters, j.grad_clip)
        optimizer.step()
        if (step + 1) % max(1, steps // 20) == 0:
            history.append({"step": step + 1, "loss": float(loss)})
            print(json.dumps({"stage": "train", "step": step + 1, "loss": round(float(loss), 6)}), flush=True)
    return world.eval(), history, presented


# ----------------------------------------------------------------------------------------- score
def sealed_rows():
    files = sorted(SEALED.glob("seed-*.pt"))
    rows = [r for f in files for r in torch.load(f, weights_only=False)]
    stack = lambda k: torch.stack([r[k] for r in rows])
    manifest = hashlib.sha256("".join(f"{f.name}:{_sha256(f)}" for f in files).encode()).hexdigest()
    return {"seed": torch.tensor([int(r["seed"]) for r in rows]), "frames": stack("frames"),
            "actions": nn.functional.one_hot(stack("led_to_action").clamp(max=N), N + 1).float(),
            "visible": stack("visible").float(), "p_death1": stack("p_death1").float(),
            "p_death2": stack("p_death2").float(), "successors": stack("successors")}, manifest


def score(panel, arms, device, *, steps=3000, probe_seeds=3, draws=1000, seed=20260924):
    from confirm import seeds_for
    from diagnose import within_auc
    from frozen_ladder import scores, standardize, strata, train
    from ladder import paired
    from observability import FRESH, expected_safe, load
    from u_world import successors, u_world_features
    from d4mj.lewm import LeWMWorld
    fit_seeds, _ = seeds_for(json.loads((HERE / "evidence/root_partition.json").read_text()), FORK_STORE)
    data = load(fit_seeds)
    recorded = json.loads((HERE / "evidence/observability.json").read_text())["identity"]
    if {s: data[s]["identity"] for s in data} != recorded:
        raise SystemExit("FIT / exploratory rows differ from what the observability test measured")
    succ = successors(fit_seeds)
    fit = dict(data["fit"], successors=succ["fit"][0])
    provenance = {"fit_identity": recorded["fit"]}
    if panel == "explore":
        judge = dict(data["judge"], successors=succ["judge"][0])
        provenance["judge_identity"] = recorded["judge"]
    else:
        judge, manifest = sealed_rows()
        used = {int(f.stem.split("-")[1]) for d in (FRESH, ROOT / "artifacts/eda/observe_fresh_v2") for f in d.glob("seed-*.pt")}
        new = set(judge["seed"].unique().tolist())
        if min(new) < 52_000 or new & used or new & set(fit_seeds):
            raise SystemExit("sealed seeds are not a new, untouched block")
        provenance.update(judge_manifest=manifest, judge_seeds=sorted(new))
    del data, succ
    config, _ = world_config()
    encoder = old_encoder(device)
    pca = torch.load(OLD / "evidence/state_cache.pt", weights_only=False, map_location="cpu")["pca"]
    worlds = {"A": A_WORLD, "B": EVIDENCE / "abc_world_B.pt", "C": EVIDENCE / "abc_world_C.pt"}
    feats = {}
    for arm in arms:
        world = LeWMWorld(config).to(device)
        world.load_state_dict(torch.load(worlds[arm], map_location="cpu", weights_only=False)["state_dict"], strict=True)
        world.eval()
        for name, d in (("fit", fit), ("judge", judge)):
            f = u_world_features(encoder, world, pca, d["frames"], d["actions"], d["successors"], device)
            feats.setdefault(name, {"root_u": f["root_u"], "real_u": f["real_u"]})[f"gen_{arm}"] = f["generated_u"]
        del world
    provenance["worlds"] = {a: _sha256(worlds[a]) for a in arms}
    torch.cuda.empty_cache()

    groups = fit["seed"].unique()
    held = set(groups[torch.randperm(len(groups), generator=torch.Generator().manual_seed(seed))
                      [:max(1, round(0.15 * len(groups)))]].tolist())
    inner = torch.tensor([int(s) in held for s in fit["seed"]])
    train_rows, hold_rows = torch.where(~inner)[0], torch.where(inner)[0]
    judge_rows = torch.arange(len(judge["seed"]))
    strat = strata(judge["visible"])
    seeds = judge["seed"]
    w = torch.load(EVIDENCE / "critical_direction.pt", weights_only=False)["w"]
    centre = lambda x: x - x.mean(1, keepdim=True)

    report = {}
    for outcome in ("death1", "death2"):
        pf, pj = fit[f"p_{outcome}"], judge[f"p_{outcome}"]
        opp_fit = (pf.amax(1) > pf.amin(1)) & ~inner
        prior = int(pf[opp_fit].mean(0).argmin())
        prior_safe, opp = expected_safe(-nn.functional.one_hot(torch.full((len(pj),), prior), N).float(), pj)
        per, block = {"prior": prior_safe}, {"judge_opportunity_roots": int(opp.sum()),
                                             "prior_expected_safe": float(prior_safe[opp].mean()), "arms": {}}
        rungs = ["root_u", "real_u"] + [f"gen_{a}" for a in arms]
        for rung in rungs:
            kind = "vector" if rung == "root_u" else "branch"
            xf, xj = feats["fit"][rung], feats["judge"][rung]
            mean, scale = standardize(xf, train_rows)
            xf, xj = (xf - mean) / scale, (xj - mean) / scale
            safes, chosen, runs = [], [], []
            for probe in range(probe_seeds):
                model, trace = train(kind, xf.shape[1:], xf, pf, train_rows, hold_rows, seed=probe, device=device, steps=steps)
                sj = scores(model, xj, judge_rows, device)
                safe, _ = expected_safe(sj, pj)
                safes.append(safe)
                chosen.append(sj[opp].argmin(1))
                runs.append({"judge_expected_safe": float(safe[opp].mean()), "selected_step": trace["selected_step"]})
                del model
            per[rung] = torch.stack(safes).mean(0)
            hist = torch.bincount(torch.cat(chosen), minlength=N)
            entry = {"runs": runs, "judge_expected_safe": float(per[rung][opp].mean()),
                     "by_stratum": {k: float(per[rung][v & opp].mean()) for k, v in strat.items() if (v & opp).any()},
                     "chosen_actions": hist.tolist(), "sleep_choices": int(hist[6])}
            if rung.startswith("gen_") and outcome == "death1":
                real, gen = feats["judge"]["real_u"], feats["judge"][rung]
                fatal = pj > 0.5
                o = fatal.any(1) & (~fatal).any(1)
                rc, gc = centre(real[o]), centre(gen[o])
                entry["fatal_direction_alignment_auc"] = within_auc((centre(gen) @ w)[o], fatal[o])
                entry["normalized_action_effect_error"] = float((gc - rc).square().sum(-1).mean() / rc.square().sum(-1).mean())
            block["arms"][rung] = entry
            print(json.dumps({"stage": "scored", "panel": panel, "outcome": outcome, "rung": rung,
                              "safe": round(entry["judge_expected_safe"], 4)}), flush=True)
        test = lambda a, b, mask=opp: paired(per[a][mask], per[b][mask], seeds[mask], draws=draws, seed=seed + 13)
        z, lava = opp & strat["zombie_adjacent"], opp & strat["lava_adjacent"]
        block["contrasts"] = {}
        for a in arms:
            if a == "A":
                continue
            block["contrasts"][f"{a}_vs_A"] = test(f"gen_{a}", "gen_A")
            block["contrasts"][f"{a}_vs_A_zombie"] = test(f"gen_{a}", "gen_A", z)
            block["contrasts"][f"{a}_vs_A_lava"] = test(f"gen_{a}", "gen_A", lava)
        if "B" in arms and "C" in arms:
            block["contrasts"]["C_vs_B"] = test("gen_C", "gen_B")
            block["contrasts"]["C_vs_B_zombie"] = test("gen_C", "gen_B", z)
        block["contrasts"]["root_u_vs_A"] = test("root_u", "gen_A")
        report[outcome] = block
    return report, provenance


def decide(panel, report, arms):
    """The declared rules, applied mechanically to one-step death."""
    b = report["death1"]
    c = b["contrasts"]
    safe = {a: b["arms"][f"gen_{a}"]["judge_expected_safe"] for a in arms}
    up = lambda t: bool(t["difference"] > 0 and t["excludes_zero"])
    if panel == "explore":
        gap = b["arms"]["root_u"]["judge_expected_safe"] - safe["A"]
        out = {"gap_root_minus_A": gap}
        if "B" in arms:
            out["B_minus_A"] = safe["B"] - safe["A"]
            out["B_largely_closes_gap"] = bool(out["B_minus_A"] >= 0.5 * gap and up(c["B_vs_A"]))
            out["train_C"] = not out["B_largely_closes_gap"]
        return out
    err = {a: b["arms"][f"gen_{a}"]["normalized_action_effect_error"] for a in arms}
    out = {}
    for x in (a for a in arms if a != "A"):
        lava = c[f"{x}_vs_A_lava"]
        s3 = (not (lava["difference"] < 0 and lava["excludes_zero"])) and err[x] <= 2 * err["A"]
        out[x] = {"S1": up(c[f"{x}_vs_A"]), "S2": up(c[f"{x}_vs_A_zombie"]), "S3": bool(s3)}
        out[x]["succeeds"] = all(out[x].values())
    readings = []
    if out.get("B", {}).get("succeeds"):
        readings.append("exposure_repair_helps")
    if out.get("C", {}).get("succeeds") and "C_vs_B" in c and up(c["C_vs_B"]):
        readings.append("loss_adds_beyond_exposure")
    out["reading"] = readings or ["stop_loss_tweaking"]
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("pool", "train", "score"))
    parser.add_argument("--arm", choices=("A_check", "B", "C"))
    parser.add_argument("--panel", choices=("explore", "sealed"))
    parser.add_argument("--arms", nargs="+", default=["A", "B"])
    args = parser.parse_args(argv)
    device = torch.device("cuda")
    started = time.time()
    log = lambda **kw: print(json.dumps({**kw, "seconds": round(time.time() - started, 1)}), flush=True)

    if args.command == "pool":
        meta = build_pool(device)
        log(status="pool_complete", **{k: v for k, v in meta.items() if not isinstance(v, dict)},
            fatal_direction_loss_share=meta["fatal_direction_loss_share"])
    elif args.command == "train":
        if args.arm == "A_check":
            cache = torch.load(OLD / "evidence/state_cache.pt", weights_only=False, map_location="cpu")
            _, history, _ = train_world(cache, steps=500, device=device)
            recorded = json.loads((ROOT / "artifacts/experiments/20260918_matched_10k/evidence/training.json").read_text())
            want = recorded["arms"]["u->u"]["history"][0]
            log(status="A_check", step=history[-1]["step"], loss=history[-1]["loss"], recorded=want,
                abs_diff=abs(history[-1]["loss"] - want["loss"]))
            return 0
        pool = torch.load(EVIDENCE / "abc_pool.pt", weights_only=False)
        world, history, presented = train_world(pool, steps=10_000, device=device,
                                                weights=pool["weights"] if args.arm == "C" else None)
        out = EVIDENCE / f"abc_world_{args.arm}.pt"
        torch.save({"state_dict": world.state_dict(), "source": "u", "target": "u", "arm": args.arm,
                    "history": history, "terminal_target_presentations": presented,
                    "pool_sha256": _sha256(EVIDENCE / "abc_pool.pt")}, out)
        log(status="trained", arm=args.arm, final_loss=history[-1]["loss"], terminal_presentations=presented)
    else:
        report, provenance = score(args.panel, args.arms, device)
        decision = decide(args.panel, report, args.arms)
        name = f"abc_{args.panel}_{'_'.join(args.arms)}.json"
        (EVIDENCE / name).write_text(json.dumps({"schema": "d4mj_abc_v1", "panel": args.panel, "arms": args.arms,
                                                 "script_sha256": _sha256(Path(__file__)), "decision": decision,
                                                 "provenance": provenance, "outcomes": report}, indent=2) + "\n")
        log(status="scored", panel=args.panel, arms=args.arms, decision=decision)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
