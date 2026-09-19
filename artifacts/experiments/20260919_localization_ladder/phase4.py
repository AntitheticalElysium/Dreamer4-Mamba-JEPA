"""Phase 4: close the audit's six gaps.

Phases 1-3 are exploratory: the objective and head were chosen on the same 36-root DEV panel the
headline reports. This phase runs the work that turns that into evidence, and the work that
tests the claims an audit found overstated.

  transformer   the Transformer arms through the SAME rank / dynamics / derangement protocol,
                because "the sequence mixer is exonerated" was never tested by this ladder
  derangement   a distribution over 20 derangements, not one draw, plus the frozen published
                head evaluated on permuted DEV actions
  health        binary dead/alive readout (scalar R^2 does not test a dead/alive boundary),
                HUD-only and map-only patch rungs, and a structured-state control.  Health
                occupies pixel rows 49-62, measured: patch rows 7-8 are HUD, 0-6 are map
  umem          internal h and [z,h] for the existing z->z and u->u worlds
  paired        per-root scores retained, paired episode-cluster bootstrap on declared contrasts
  confirm       an untouched confirmation panel drawn from the historical replay shards by a
                rule declared before any model is scored, evaluating ONLY predeclared contrasts

Stages hash their inputs and refuse to reuse a cache whose inputs moved.
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path("/home/antithetical/EPITA/PERSO/DynamicHorizons-Mamba-JEPA")
HERE = Path(__file__).parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

from d4mj.m03.cache import resolve_payload
from readout import Fit, evaluate, fit_head, one_hot_actions, scores_of, standardize, summarize
from phase2 import apply_pca, fit_pca, WIDTH

EVAL = ROOT / "artifacts/lewm_gates_20260906/m03_bootstrap/evaluation_v2"
TF = ROOT / "artifacts/lewm_transformer_comparison/m03"
MATCHED = ROOT / "artifacts/lewm_gates_20260918/m03_matched"
SIDECAR = EVAL / "sidecar/sidecar.probe_only.pt"
SEEDS = (0, 1, 2)
HUD_TOKENS = list(range(63, 81))      # patch rows 7-8 -> pixel rows 49-62, measured
MAP_TOKENS = list(range(0, 63))       # patch rows 0-6 -> pixel rows 0-48


def feats(base, arm, split):
    return resolve_payload(torch.load(f"{base}/features/{arm}.{split}.pt", map_location="cpu",
                                      weights_only=False))["features"]


def memfeats(base, arm, split):
    return resolve_payload(torch.load(f"{base}/memory/features/{arm}.primary_{split}.pt",
                                      map_location="cpu", weights_only=False))["features"]


def load_taps(name):
    return torch.load(HERE / f"cache/taps.{name}.pt", map_location="cpu", weights_only=False)


def derangement(n, generator):
    while True:
        c = torch.randperm(n, generator=generator)
        if not bool((c == torch.arange(n)).any()):
            return c


def expand(v):
    return v[:, None].expand(-1, 17, -1)


def fit_and_score(xtr, ytr, xdv, ydv, spec, device, *, family="mlp128", objective="rank",
                  seeds=SEEDS, keep_scores=False):
    xtr, xdv = standardize(xtr, xdv)
    got, kept = [], []
    for seed in seeds:
        model, _, params = fit_head(xtr, ytr, family=family, objective=objective, seed=seed,
                                    spec=spec, device=device)
        s = scores_of(model, xdv, device)
        got.append(summarize(s, ydv))
        if keep_scores:
            kept.append(s)
    out = {"family": family, "objective": objective, "parameters": params,
           "safe_choice": [m["safe_choice"] for m in got],
           "mean_safe": round(float(np.mean([m["safe_choice"] for m in got])), 2),
           "opportunity_roots": got[0]["opportunity_roots"],
           "within_root_auc": [round(m["within_root_auc"], 4) for m in got]}
    return (out, kept) if keep_scores else (out, None)


def paired_bootstrap(a_scores, b_scores, y, episodes, draws=1000, seed=20260919):
    """Paired episode-cluster bootstrap on the difference in safe-choice count."""
    labels = y[..., 0].bool()
    usable = labels.any(1) & (~labels).any(1)

    def count(scores, rows):
        r = rows[usable[rows]]
        if len(r) == 0:
            return None
        return float((~labels)[r, scores[r].argmin(1)].float().mean())

    keys = episodes.unique(sorted=True)
    groups = [torch.where(episodes == k)[0] for k in keys]
    rng = torch.Generator().manual_seed(seed)
    base_a, base_b = count(a_scores, torch.arange(len(episodes))), count(b_scores, torch.arange(len(episodes)))
    samples = []
    for _ in range(draws):
        pick = torch.randint(len(groups), (len(groups),), generator=rng)
        rows = torch.cat([groups[i] for i in pick])
        x, z = count(a_scores, rows), count(b_scores, rows)
        if x is not None and z is not None:
            samples.append(x - z)
    if len(samples) < 0.95 * draws:
        return {"a": base_a, "b": base_b, "difference": base_a - base_b, "interval": None}
    lo, hi = torch.tensor(samples).quantile(torch.tensor([.025, .975])).tolist()
    return {"a": round(base_a, 4), "b": round(base_b, 4), "difference": round(base_a - base_b, 4),
            "interval": [round(lo, 4), round(hi, 4)],
            "excludes_zero": bool(lo > 0 or hi < 0)}


def publish(path, payload, inputs):
    payload["_inputs"] = inputs
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, indent=2) + "\n")
    tmp.replace(path)


def cached(path, inputs):
    if not path.exists():
        return None
    got = json.loads(path.read_text())
    return got if got.get("_inputs") == inputs else None


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()[:16]


# ============================== stages ======================================================

def stage_transformer(side, y, spec, out, device):
    """The Transformer arms through the same rank / dynamics / derangement protocol."""
    inputs = {"tf_features": sha(f"{TF}/features/raw.dev.pt")}
    got = cached(out / "transformer_dynamics.json", inputs)
    if got:
        return got
    rows, gen = [], torch.Generator().manual_seed(20260919)
    order = derangement(17, gen)
    for arm in ("raw", "tc"):
        f = {s: feats(TF, arm, s) for s in ("train", "dev")}
        real = {s: f[s]["observed_successor"].float() for s in ("train", "dev")}
        g = {s: f[s]["generated_successor"].float() for s in ("train", "dev")}
        root = {s: f[s]["projected"].float() for s in ("train", "dev")}
        for name, xtr, xdv in (
                ("real_fit_real", real["train"], real["dev"]),
                ("gen_fit_gen", g["train"], g["dev"]),
                ("gen_fit_gen_action_permuted", g["train"][:, order], g["dev"][:, order]),
                ("root_action", torch.cat((expand(root["train"]), one_hot_actions(len(root["train"]))), -1),
                 torch.cat((expand(root["dev"]), one_hot_actions(len(root["dev"]))), -1))):
            r, _ = fit_and_score(xtr, y["train"], xdv, y["dev"], spec, device)
            rows.append({"arm": f"transformer_{arm}", "condition": name, **r})
            print(json.dumps({"stage": "tf", "arm": arm, "cond": name, "mean": r["mean_safe"]}), flush=True)
        # objective contrast on real z, the symptom-1 test
        for objective in ("bce_death", "rank"):
            r, _ = fit_and_score(real["train"], y["train"], real["dev"], y["dev"], spec, device,
                                 objective=objective)
            rows.append({"arm": f"transformer_{arm}", "condition": f"real_{objective}", **r})
            print(json.dumps({"stage": "tf_obj", "arm": arm, "objective": objective,
                              "mean": r["mean_safe"]}), flush=True)
    payload = {"schema": "d4mj_phase4_transformer_v1", "rows": rows}
    publish(out / "transformer_dynamics.json", payload, inputs)
    return payload


def stage_derangement(side, y, spec, out, device, draws=20):
    """A distribution over derangements, plus the frozen published head on permuted DEV actions."""
    inputs = {"draws": draws, "eval": sha(f"{EVAL}/features/raw.dev.pt")}
    got = cached(out / "derangement_distribution.json", inputs)
    if got:
        return got
    gen = torch.Generator().manual_seed(4242)
    orders = [derangement(17, gen) for _ in range(draws)]
    rows = []
    sources = {"mamba_raw": (EVAL, "raw"), "mamba_tc": (EVAL, "tc"), "direct": (EVAL, "direct_mamba"),
               "world_u_u": (MATCHED, "tc"), "transformer_raw": (TF, "raw")}
    for label, (base, arm) in sources.items():
        f = {s: feats(base, arm, s) for s in ("train", "dev")}
        g = {s: f[s]["generated_successor"].float() for s in ("train", "dev")}
        intact, _ = fit_and_score(g["train"], y["train"], g["dev"], y["dev"], spec, device, seeds=(0,))
        spread = []
        for order in orders:
            r, _ = fit_and_score(g["train"][:, order], y["train"], g["dev"][:, order], y["dev"],
                                 spec, device, seeds=(0,))
            spread.append(r["mean_safe"])
        rows.append({"arm": label, "intact": intact["mean_safe"], "deranged": spread,
                     "deranged_mean": round(float(np.mean(spread)), 2),
                     "deranged_p05_p95": [round(float(np.percentile(spread, 5)), 2),
                                          round(float(np.percentile(spread, 95)), 2)],
                     "cost": round(intact["mean_safe"] - float(np.mean(spread)), 2),
                     "intact_above_all_derangements": bool(intact["mean_safe"] > max(spread))})
        print(json.dumps({"stage": "derange", "arm": label, "intact": intact["mean_safe"],
                          "deranged_mean": rows[-1]["deranged_mean"], "cost": rows[-1]["cost"]}), flush=True)
    payload = {"schema": "d4mj_phase4_derangement_v1", "draws": draws, "rows": rows,
               "note": "one derangement draw is a point estimate; this is its distribution"}
    publish(out / "derangement_distribution.json", payload, inputs)
    return payload


def stage_health(side, y, spec, out, device):
    """Binary dead/alive, HUD-only vs map-only patch rungs, and a structured-state control.

    Scalar health R^2 does not test what `death` needs: a representation can encode the
    dead/alive boundary sharply and regress magnitude poorly. Health is drawn in pixel rows
    49-62, so patch tokens 63-80 are HUD and 0-62 are map -- masking one tests whether the
    patch advantage is a HUD-reading advantage.
    """
    inputs = {"taps": sha(HERE / "cache/taps.mamba_raw.pt")}
    got = cached(out / "health_controls.json", inputs)
    if got:
        return got
    health = {s: side[s]["next_continuous"][:, :, 0].float() for s in ("train", "dev")}
    alive = {s: (health[s] > 0).float().unsqueeze(-1) for s in ("train", "dev")}
    rows = []
    for arm in ("mamba_raw", "mamba_tc"):
        taps = {arm: load_taps(arm)}
        grid = {s: taps[arm][s]["successor"]["patches"].float() for s in ("train", "dev")}
        rungs = {
            "z": {s: taps[arm][s]["successor"]["z"].float() for s in ("train", "dev")},
            "cls": {s: taps[arm][s]["successor"]["cls"].float() for s in ("train", "dev")},
            "patch_mean": {s: taps[arm][s]["successor"]["patch_mean"].float() for s in ("train", "dev")},
            "hud_tokens_mean": {s: grid[s][:, :, HUD_TOKENS].mean(2) for s in ("train", "dev")},
            "map_tokens_mean": {s: grid[s][:, :, MAP_TOKENS].mean(2) for s in ("train", "dev")},
        }
        for name, v in rungs.items():
            # binary dead/alive AUC, fitted with plain BCE on the alive indicator
            xtr, xdv = standardize(v["train"], v["dev"])
            model, _, params = fit_head(xtr, alive["train"], family="mlp128", objective="bce6",
                                        seed=0, spec=spec, device=device)
            with torch.inference_mode():
                pred = torch.cat([model(xdv[i:i + 128].to(device))[..., 0].cpu()
                                  for i in range(0, len(xdv), 128)])
            t = alive["dev"][..., 0].bool().reshape(-1)
            p = pred.reshape(-1)
            pos, neg = p[t], p[~t]
            auc = float((pos[:, None] > neg[None, :]).float().mean())
            choice, _ = fit_and_score(v["train"], y["train"], v["dev"], y["dev"], spec, device)
            rows.append({"arm": arm, "rung": name, "dim": int(v["train"].shape[-1]),
                         "dead_alive_auc": round(auc, 4), "parameters": params,
                         "safe_choice_mean": choice["mean_safe"]})
            print(json.dumps({"stage": "health", **rows[-1]}), flush=True)
        del taps, grid, rungs
    # structured successor state: the simulator's own 16 continuous fields
    sx = {s: side[s]["next_continuous"].float() for s in ("train", "dev")}
    r, _ = fit_and_score(sx["train"], y["train"], sx["dev"], y["dev"], spec, device)
    rows.append({"arm": "structured_state_control", "rung": "next_continuous", "dim": 16,
                 "dead_alive_auc": None, "safe_choice_mean": r["mean_safe"]})
    print(json.dumps({"stage": "health", **rows[-1]}), flush=True)
    payload = {"schema": "d4mj_phase4_health_v1", "hud_tokens": HUD_TOKENS[:2] + ["..."],
               "hud_evidence": "health correlates with pixel rows 49-62 (max r=0.379 at row 50); "
                               "rows 0-48 mean |r|=0.018", "rows": rows}
    publish(out / "health_controls.json", payload, inputs)
    return payload


def stage_umem(side, y, spec, out, device):
    """Internal h and [z,h] for the existing z->z and u->u worlds."""
    inputs = {"matched": sha(f"{MATCHED}/memory/features/tc.primary_dev.pt")}
    got = cached(out / "u_world_memory.json", inputs)
    if got:
        return got
    rows, gen = [], torch.Generator().manual_seed(20260919)
    order = derangement(17, gen)
    for arm, label in (("raw", "world_z_z"), ("tc", "world_u_u")):
        f = {s: memfeats(MATCHED, arm, s) for s in ("train", "dev")}
        g = {s: f[s]["c4_generated_z"].float() for s in ("train", "dev")}
        h = {s: f[s]["c4_next_h"].float() for s in ("train", "dev")}
        j = {s: torch.cat((g[s], h[s]), -1) for s in ("train", "dev")}
        for name, v in (("generated", g), ("next_h", h), ("generated_and_h", j)):
            r, _ = fit_and_score(v["train"], y["train"], v["dev"], y["dev"], spec, device)
            rows.append({"world": label, "condition": name, "dim": int(v["train"].shape[-1]), **r})
            print(json.dumps({"stage": "umem", "world": label, "cond": name,
                              "mean": r["mean_safe"]}), flush=True)
            rd, _ = fit_and_score(v["train"][:, order], y["train"], v["dev"][:, order], y["dev"],
                                  spec, device, seeds=(0,))
            rows.append({"world": label, "condition": f"{name}__action_permuted",
                         "dim": int(v["train"].shape[-1]), **rd})
    payload = {"schema": "d4mj_phase4_umem_v1",
               "role": "u->u is a DIAGNOSTIC for Raw/TC, not a canonical architecture", "rows": rows}
    publish(out / "u_world_memory.json", payload, inputs)
    return payload


PREDECLARED = [
    {"id": "C1_objective", "claim": "within-root ranking beats six-target BCE on real Mamba Raw z",
     "a": "mamba_raw real z, rank", "b": "mamba_raw real z, bce_death", "direction": "a>b"},
    {"id": "C2_export", "claim": "a pooled-patch PCA-192 rung beats z at matched width on real successors",
     "a": "mamba_raw pooled4_pca192", "b": "mamba_raw z", "direction": "a>b"},
    {"id": "C3_generated", "claim": "generated Mamba Raw z beats its own root+action control",
     "a": "mamba_raw generated z", "b": "mamba_raw root+action", "direction": "a>b"},
]


def stage_paired(side, y, spec, out, device):
    """Retain per-root scores and put paired episode-cluster intervals on declared contrasts."""
    inputs = {"eval": sha(f"{EVAL}/features/raw.dev.pt"), "contrasts": len(PREDECLARED)}
    got = cached(out / "paired_intervals.json", inputs)
    if got:
        return got
    episodes = side["dev"]["episode"]
    taps = {"mamba_raw": load_taps("mamba_raw")}
    f = {s: feats(EVAL, "raw", s) for s in ("train", "dev")}
    real = {s: taps["mamba_raw"][s]["successor"]["z"].float() for s in ("train", "dev")}
    pooled = {s: taps["mamba_raw"][s]["successor"]["pooled4"].float().flatten(2) for s in ("train", "dev")}
    mean, basis, _ = fit_pca(pooled["train"], WIDTH)
    pca = {s: apply_pca(pooled[s], mean, basis) for s in ("train", "dev")}
    g = {s: f[s]["generated_successor"].float() for s in ("train", "dev")}
    root = {s: f[s]["projected"].float() for s in ("train", "dev")}
    ra = {s: torch.cat((expand(root[s]), one_hot_actions(len(root[s]))), -1) for s in ("train", "dev")}

    keep = {}
    for name, v, objective in (("real_rank", real, "rank"), ("real_bce", real, "bce_death"),
                               ("pca_rank", pca, "rank"), ("gen_rank", g, "rank"),
                               ("rootaction_rank", ra, "rank")):
        _, scores = fit_and_score(v["train"], y["train"], v["dev"], y["dev"], spec, device,
                                  objective=objective, keep_scores=True)
        keep[name] = scores
    torch.save(keep, out / "dev_scores.mamba_raw.pt")

    results = []
    for spec_row, (a, b) in zip(PREDECLARED, (("real_rank", "real_bce"), ("pca_rank", "real_rank"),
                                              ("gen_rank", "rootaction_rank"))):
        per_seed = [paired_bootstrap(keep[a][i], keep[b][i], y["dev"], episodes) for i in range(len(SEEDS))]
        results.append({**spec_row, "per_seed": per_seed,
                        "mean_difference": round(float(np.mean([p["difference"] for p in per_seed])), 4),
                        "all_seeds_exclude_zero": all(p.get("excludes_zero") for p in per_seed)})
        print(json.dumps({"stage": "paired", "id": spec_row["id"],
                          "mean_diff": results[-1]["mean_difference"],
                          "all_exclude_zero": results[-1]["all_seeds_exclude_zero"]}), flush=True)
    payload = {"schema": "d4mj_phase4_paired_v1", "unit": "fraction of the 36 opportunity roots",
               "scores_file": "dev_scores.mamba_raw.pt", "contrasts": results}
    publish(out / "paired_intervals.json", payload, inputs)
    return payload


def build_confirmation_index(minimum=100):
    """DECLARED rule, fixed before any model is scored.

    Walk the historical replay shards in ascending seed order; keep every root that offers both
    a fatal and a safe action; stop once at least `minimum` such roots are held. Historical
    grouping is preserved -- these shards are a separate TEST population and are NOT relabelled
    as support-v2 TRAIN/DEV.
    """
    shards = sorted((EVAL / "historical/features").glob("replay.*.pt"),
                    key=lambda p: int(p.stem.split(".")[1]))
    kept, seeds = [], []
    for path in shards:
        payload = resolve_payload(torch.load(path, map_location="cpu", weights_only=False))["features"]
        labels = payload["outcomes"][..., 0].bool()
        usable = labels.any(1) & (~labels).any(1)
        if not bool(usable.any()):
            continue
        kept.append({"seed": int(path.stem.split(".")[1]),
                     "rows": torch.where(usable)[0].tolist(),
                     "outcomes": payload["outcomes"][usable].clone()})
        seeds.append(int(path.stem.split(".")[1]))
        if sum(len(k["rows"]) for k in kept) >= minimum:
            break
    return kept, seeds


def stage_confirm(side, y, spec, out, device, minimum=100):
    """Evaluate ONLY the predeclared contrasts on the untouched historical panel."""
    inputs = {"minimum": minimum, "rule": "ascending seed order, roots with both fatal and safe"}
    got = cached(out / "confirmation_panel.json", inputs)
    if got:
        return got
    index, seeds = build_confirmation_index(minimum)
    total = sum(len(k["rows"]) for k in index)
    print(json.dumps({"stage": "confirm_index", "seeds": len(seeds), "opportunity_roots": total}), flush=True)

    def gather(arm, field, base=EVAL):
        out_rows, ep = [], []
        for k in index:
            payload = resolve_payload(torch.load(
                base / f"historical/features/{arm}.{k['seed']}.pt", map_location="cpu",
                weights_only=False))["features"]
            out_rows.append(payload[field][k["rows"]].float())
            ep.extend([k["seed"]] * len(k["rows"]))
        return torch.cat(out_rows), torch.tensor(ep)

    truth = torch.cat([k["outcomes"] for k in index]).float()
    real_dev, episodes = gather("raw", "observed_successor")
    gen_dev, _ = gather("raw", "generated_successor")
    root_dev, _ = gather("raw", "projected")
    ra_dev = torch.cat((expand(root_dev), one_hot_actions(len(root_dev))), -1)

    taps = {"mamba_raw": load_taps("mamba_raw")}
    f = {s: feats(EVAL, "raw", s) for s in ("train",)}
    real_tr = taps["mamba_raw"]["train"]["successor"]["z"].float()
    gen_tr = f["train"]["generated_successor"].float()
    root_tr = f["train"]["projected"].float()
    ra_tr = torch.cat((expand(root_tr), one_hot_actions(len(root_tr))), -1)

    rows = []

    def run(name, xtr, xdv, objective="rank"):
        r, scores = fit_and_score(xtr, y["train"], xdv, truth, spec, device,
                                  objective=objective, keep_scores=True)
        rows.append({"condition": name, **r})
        print(json.dumps({"stage": "confirm", "cond": name, "mean": r["mean_safe"],
                          "roots": r["opportunity_roots"]}), flush=True)
        return scores

    s_real_rank = run("mamba_raw_real_z_rank", real_tr, real_dev)
    s_real_bce = run("mamba_raw_real_z_bce", real_tr, real_dev, objective="bce_death")
    s_gen = run("mamba_raw_generated_z_rank", gen_tr, gen_dev)
    s_ra = run("mamba_raw_root_action_rank", ra_tr, ra_dev)

    intervals = {
        "C1_objective": [paired_bootstrap(s_real_rank[i], s_real_bce[i], truth, episodes)
                         for i in range(len(SEEDS))],
        "C3_generated": [paired_bootstrap(s_gen[i], s_ra[i], truth, episodes)
                         for i in range(len(SEEDS))]}
    payload = {"schema": "d4mj_phase4_confirmation_v1",
               "panel": {"source": "historical replay shards, ascending seed order",
                         "seeds": seeds, "opportunity_roots": total,
                         "grouping": "historical TEST population; NOT relabelled as support-v2 TRAIN/DEV",
                         "fit_on": "support-v2 primary TRAIN (unchanged)"},
               "note": "C2_export is not evaluated here: the historical shards publish z only, "
                       "no patch taps, so the export contrast cannot be scored on this panel "
                       "without re-encoding it.",
               "rows": rows, "intervals": intervals}
    publish(out / "confirmation_panel.json", payload, inputs)
    return payload


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", default="all")
    parser.add_argument("--out", type=Path, default=HERE / "evidence/phase4")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--minimum", type=int, default=100)
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "predeclared_contrasts.json").write_text(
        json.dumps({"schema": "d4mj_phase4_predeclaration_v1",
                    "written_before_scoring": True, "contrasts": PREDECLARED}, indent=2) + "\n")
    side = torch.load(SIDECAR, map_location="cpu", weights_only=False)["splits"]
    y = {s: side[s]["outcomes"].float() for s in ("train", "dev")}
    spec = Fit(steps=args.steps)
    stages = {"transformer": stage_transformer, "derangement": stage_derangement,
              "health": stage_health, "umem": stage_umem, "paired": stage_paired}
    order = list(stages) + ["confirm"]
    chosen = order if args.stage == "all" else [args.stage]
    for name in chosen:
        print(json.dumps({"stage": "begin", "name": name}), flush=True)
        if name == "confirm":
            stage_confirm(side, y, spec, args.out, args.device, args.minimum)
        else:
            stages[name](side, y, spec, args.out, args.device)
        print(json.dumps({"stage": "done", "name": name}), flush=True)
    print(json.dumps({"status": "phase4_complete"}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
