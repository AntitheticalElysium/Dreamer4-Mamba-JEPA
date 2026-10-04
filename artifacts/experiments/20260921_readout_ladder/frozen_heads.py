"""Frozen-T head comparison: which supervision lets a head read safe actions from T's generated state,
and does T's transition add anything beyond passing its input and the action through?

After review of SPATIAL.md. The 2x2's T world is frozen (`spatial_worlds_v1/T.pt`: six-layer
Transformer, per-tile state, H2-style heads; NOT Mamba). Nothing about it is retrained. Only fresh heads
are fitted, all reading T's GENERATED one-step successor (81 tiles, 4-frame context, each of the 17
actions advanced once). The result can establish at most a FORK-SUPERVISED SAFETY READOUT: fork labels
are outside the canonical factual-training contract (TC-17), and the per-tile state deviates from
TC-07/TC-19. World fidelity (whether the generated successor draws the damage) is a separate verdict,
not tested here.

DATA -- the fork-store partition (`evidence/root_partition.json`), respected exactly:
  fit      FIT-train seeds (700): the observability FIT roots, 32-key P(death1) per action (observe_fit_v1)
  select   FIT-dev seeds (350): fork-store rows, realized one-step death per action (single key;
           one-step death is near-deterministic here, 0.17% random pairs), checkpoint selection only
  never    the 50 gate-reserved and 405 unallocated seeds
  judge    NEW sealed block, `observe.py collect --seed-start 54000 --target-opportunity 800
           --max-seeds 1500 --out artifacts/eda/observe_fresh_v5`, collected after this commit, read once

HEADS -- one architecture (spatial_why4's attention probe: learned tile positions, attention pooling,
128 -> 512 -> 1 per branch), three seeds each, 3,000 updates, AdamW 1e-3 / 1e-4, selection every 100
updates on FIT-dev expected safe choice:
  logged       BCE on the collector's own action at each root only (32-key P as soft target)
  uniform      BCE on ONE uniformly sampled action per root (seeded once): factual count without the
               policy's selection
  all_bce      BCE on all 17 actions of every root
  all_rank     frozen_ladder's soft_rank over the 17 actions of every root              (primary)
  root_bce     CONTROL: the same head on the ROOT's last-frame tiles (layer-normed, T's own input) plus
  root_rank    a learned action embedding added to every tile; all-action BCE / ranking
BCE heads draw 256 branches per update; ranking heads 16 roots (272 branches).
References (frozen_ladder, FIT-train fit / FIT-dev selection): the FIT action prior; actions_only
(last 4 actions); root_tokens (tokens_attn on the root's raw patch tokens, the boundary readout).

JUDGEMENT -- one-step death, expected safe = 1 - P(death1 | chosen) on 32-key P, per-root mean over
head seeds, paired episode-seed-clustered 95% intervals, 1,000 draws; opportunity roots.
Declared readings:
  readout     all_rank - prior > 0 and all_rank - actions_only > 0 (both resolved), and all_rank - prior
              > 0 resolved on zombie-adjacent roots  -> fork_supervised_safety_readout_works; else
              -> readout_fails
  transition  all_rank - root_rank: resolved > 0 -> transition_adds; resolved < 0 -> transition_loses;
              otherwise -> no_evidence_transition_adds
  labels      uniform - logged > 0 resolved -> logged_selection_bias;  all_bce - uniform > 0 resolved
              -> coverage_matters;  all_rank - all_bce > 0 resolved -> ranking_objective_matters
  stability   an arm whose three head seeds' judge means span more than 0.10 is reported unstable
Reported: zombie, lava, night and stay-kills-move-survives strata; BCE arms' calibration (Brier and
mean predicted vs mean true P(death1)) on opportunity AND ordinary roots; chosen-action histograms;
per-seed judge means.
"""

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path("/home/antithetical/EPITA/PERSO/DynamicHorizons-Mamba-JEPA")
HERE = Path(__file__).parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

from d4mj.data import _sha256
from d4mj.lewm_diagnostics import FORK_STORE

N, STEPS, SEEDS = 17, 3000, 3
SEALED = ROOT / "artifacts/eda/observe_fresh_v5"
ARMS = ("logged", "uniform", "all_bce", "all_rank", "root_bce", "root_rank")


def dev_rows(seeds):
    """FIT-dev fork rows with four real context frames: frames, one-hot actions, realized death, seed."""
    files = {int(p.stem.split("-")[1]): p for p in FORK_STORE.glob("seed-*.pt")}
    rows = [r for s in sorted(seeds) for r in torch.load(files[s], weights_only=False)
            if bool((r["led_to_action"][-3:] < N).all()) and len(r["frames"]) >= 4]
    stack = lambda k: torch.stack([r[k] for r in rows])
    return {"frames": torch.stack([r["frames"][-4:] for r in rows]),
            "actions": F.one_hot(torch.stack([r["led_to_action"][-4:] for r in rows]).clamp(max=N), N + 1).float(),
            "p_death1": stack("terminated").float(), "seed": torch.tensor([int(r["seed"]) for r in rows])}


def train_head(kind, x, p, bc, fit_rows, device, seed, select):
    """kind in ARMS; x: generated [R,17,81,192] or root tiles [R,81,192]; p: [R,17] targets."""
    from observability import soft_rank
    from spatial_why4 import Probe
    torch.manual_seed(seed)
    root = kind.startswith("root")
    model = Probe(x.shape[-1], root).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    gen = torch.Generator().manual_seed(seed)
    uniform = torch.randint(N, (len(p),), generator=torch.Generator().manual_seed(20261002))
    ranking = kind.endswith("rank")

    def risk(rows, data, acts=None):
        """[len(rows), 17] risk for every action, or [len(rows)] for the given actions."""
        if root:
            tiles = data[rows].float().to(device)
            if acts is None:
                return model(tiles.repeat_interleave(N, 0), torch.arange(N, device=device).repeat(len(rows))).view(len(rows), N)
            return model(tiles, acts.to(device))
        if acts is None:
            return model(data[rows].flatten(0, 1).float().to(device)).view(len(rows), N)
        return model(data[rows, acts].float().to(device))

    best, state, curve = -1.0, None, []
    for step in range(STEPS):
        if ranking:
            rows = fit_rows[torch.randint(len(fit_rows), (16,), generator=gen)]
            loss = soft_rank(risk(rows, x), p[rows].to(device))
        else:
            if kind == "logged" or kind == "uniform":
                rows = fit_rows[torch.randint(len(fit_rows), (256,), generator=gen)]
                acts = bc[rows] if kind == "logged" else uniform[rows]
            else:
                pick = torch.randint(len(fit_rows) * N, (256,), generator=gen)
                rows, acts = fit_rows[pick // N], pick % N
            logit = risk(rows, x, acts)
            loss = F.binary_cross_entropy_with_logits(logit, p[rows, acts].to(device))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if (step + 1) % 100 == 0 or step + 1 == STEPS:
            model.eval()
            value = select(lambda rows, data: risk(rows, data))
            model.train()
            curve.append(round(value, 4))
            if value > best or state is None:
                best, state, chosen = value, {k: v.detach().clone() for k, v in model.state_dict().items()}, step + 1
    model.load_state_dict(state)
    model.eval()

    @torch.no_grad()
    def scorer(data):
        rows = torch.arange(len(data))
        return torch.cat([risk(rows[i:i + 128], data) for i in range(0, len(rows), 128)]).cpu()
    return scorer, {"selected_step": chosen, "select_safe": best, "curve": curve}


def main():
    started = time.time()
    log = lambda **kw: print(json.dumps({**kw, "seconds": round(time.time() - started, 1)}), flush=True)
    device = torch.device("cuda")
    os.chdir(ROOT)
    from boundary import judge_store
    from confirm import seeds_for
    from frozen_ladder import arm_input, scores, standardize, strata, train as ladder_train
    from ladder import paired
    from observability import expected_safe, load
    from spatial import TOKENS, WORLDS, World, bridge, encode
    from spatial_why6 import generate
    from spatial_why8 import bc_actions

    partition = json.loads((HERE / "evidence/root_partition.json").read_text())
    fit_seeds, unallocated = seeds_for(partition, FORK_STORE)
    dev_seeds = sorted(partition["fit_dev"]["seeds"])
    forbidden = set(partition["reserved_for_gate"]["seeds"]) | set(unallocated)
    if forbidden & (set(fit_seeds) | set(dev_seeds)):
        raise SystemExit("partition overlap")
    fit = load(fit_seeds)["fit"]
    bc = bc_actions(fit_seeds)
    dev = dev_rows(dev_seeds)
    judge, manifest, files = judge_store(SEALED)
    new = set(judge["seed"].unique().tolist())
    used = {int(f.stem.split("-")[1]) for store in ("observe_fresh_v1", "observe_fresh_v2", "observe_fresh_v3",
                                                   "observe_fresh_v4") for f in (ROOT / "artifacts/eda" / store).glob("seed-*.pt")}
    if min(new) < 54_000 or new & used or new & (set(fit_seeds) | set(dev_seeds) | forbidden):
        raise SystemExit("judgement seeds are not a new, untouched block")
    log(stage="data", fit=len(bc), dev=len(dev["seed"]), judge=len(judge["seed"]), judge_files=files)

    encoder, config = bridge()
    stored = torch.load(WORLDS / "T.pt", map_location="cpu", weights_only=False)
    world = World(TOKENS, False).to(device)
    world.load_state_dict(stored["world"])
    world.eval()
    sets = {"fit": fit, "dev": dev, "judge": judge}
    for data in sets.values():
        data["generated"] = generate(world, config, encoder, data["frames"], data["actions"], device, "tokens")
        _, tiles = encode(encoder, data["frames"][:, -1:], device)
        data["root_tiles"] = tiles[:, 0]
        with torch.no_grad():
            data["tokens1"] = torch.cat([encoder._hidden(data["frames"][i:i + 64, -1:].to(device))[2].cpu()
                                         for i in range(0, len(data["frames"]), 64)])
        del data["frames"]
    del world, encoder
    torch.cuda.empty_cache()
    log(stage="generated")

    pf, pd, pj = fit["p_death1"], dev["p_death1"], judge["p_death1"]
    fit_rows = torch.arange(len(pf))

    safe, per_seed, calibration, histogram, traces = {}, {}, {}, {}, {}
    _, opp = expected_safe(pj, pj)
    for arm in ARMS:
        key = "root_tiles" if arm.startswith("root") else "generated"

        def select(risk, key=key):
            rows = torch.arange(len(pd))
            with torch.no_grad():
                r = torch.cat([risk(rows[i:i + 128], dev[key]) for i in range(0, len(rows), 128)]).cpu()
            s, o = expected_safe(r, pd)
            return float(s[o].mean())
        runs, probs = [], []
        for seed in range(SEEDS):
            scorer, trace = train_head(arm, fit[key], pf, bc, fit_rows, device, seed, select)
            r = scorer(judge[key])
            runs.append(expected_safe(r, pj)[0])
            probs.append(torch.sigmoid(r))
            histogram.setdefault(arm, np.zeros(N, int))
            histogram[arm] += np.bincount(r[opp].argmin(1).numpy(), minlength=N)
            traces.setdefault(arm, []).append(trace)
        safe[arm] = torch.stack(runs).mean(0)
        per_seed[arm] = [float(x[opp].mean()) for x in runs]
        if not arm.endswith("rank"):
            p = torch.stack(probs).mean(0)
            calibration[arm] = {name: {"brier": float(((p[m] - pj[m]) ** 2).mean()), "mean_predicted": float(p[m].mean()),
                                       "mean_true": float(pj[m].mean())} for name, m in (("opportunity", opp), ("ordinary", ~opp))}
        log(arm=arm, safe=round(float(safe[arm][opp].mean()), 4), per_seed=[round(v, 4) for v in per_seed[arm]])

    # references, frozen_ladder procedure with FIT-train fit and FIT-dev selection
    both = lambda k: torch.cat([fit[k], dev[k]])
    train_rows, hold_rows = torch.arange(len(pf)), torch.arange(len(pf), len(pf) + len(pd))
    p_both = torch.cat([pf, pd])
    prior = int(pf[pf.amax(1) > pf.amin(1)].mean(0).argmin())
    safe["prior"] = expected_safe(-F.one_hot(torch.full((len(pj),), prior), N).float(), pj)[0]
    actions4 = torch.cat([fit["actions"][:, -4:], dev["actions"][:, -4:]]).flatten(1)
    for name, kind, xf, xj in (("actions_only", "vector", actions4, judge["actions"][:, -4:].flatten(1)),
                               ("root_tokens", "tokens_attn", both("tokens1"), judge["tokens1"])):
        mean, scale = standardize(xf, train_rows)
        xf, xj = (xf.float() - mean) / scale, (xj.float() - mean) / scale
        runs = []
        for seed in range(SEEDS):
            model, _ = ladder_train(kind, xf.shape[1:], xf, p_both, train_rows, hold_rows, seed=seed, device=device, steps=STEPS)
            runs.append(expected_safe(scores(model, xj, torch.arange(len(pj)), device), pj)[0])
        safe[name] = torch.stack(runs).mean(0)
        per_seed[name] = [float(x[opp].mean()) for x in runs]
        log(arm=name, safe=round(float(safe[name][opp].mean()), 4))

    strat = strata(judge["visible"])
    seeds = judge["seed"]
    zombie = opp & strat["zombie_adjacent"]
    stay = opp & (pj[:, 0] > 0.5) & (pj[:, 1:5].amin(1) < 0.5)
    test = lambda a, b, m: paired(safe[a][m], safe[b][m], seeds[m], draws=1000, seed=20261003)
    up = lambda r: r["difference"] is not None and r["difference"] > 0 and r["excludes_zero"]
    down = lambda r: r["difference"] is not None and r["difference"] < 0 and r["excludes_zero"]
    rules = {"readout_vs_prior": test("all_rank", "prior", opp), "readout_vs_actions_only": test("all_rank", "actions_only", opp),
             "readout_vs_prior_zombie": test("all_rank", "prior", zombie),
             "transition": test("all_rank", "root_rank", opp),
             "uniform_vs_logged": test("uniform", "logged", opp), "all_bce_vs_uniform": test("all_bce", "uniform", opp),
             "all_rank_vs_all_bce": test("all_rank", "all_bce", opp)}
    readings = {
        "readout": ("fork_supervised_safety_readout_works" if up(rules["readout_vs_prior"]) and
                    up(rules["readout_vs_actions_only"]) and up(rules["readout_vs_prior_zombie"]) else "readout_fails"),
        "transition": ("transition_adds" if up(rules["transition"]) else "transition_loses" if down(rules["transition"])
                       else "no_evidence_transition_adds"),
        "labels": {"logged_selection_bias": up(rules["uniform_vs_logged"]), "coverage_matters": up(rules["all_bce_vs_uniform"]),
                   "ranking_objective_matters": up(rules["all_rank_vs_all_bce"])},
        "unstable": [a for a, v in per_seed.items() if max(v) - min(v) > 0.10]}
    names = (*ARMS, "prior", "actions_only", "root_tokens")
    reported = {"expected_safe": {k: {"overall": float(safe[k][opp].mean()), "zombie": float(safe[k][zombie].mean()),
                                      "stay_kills_move_survives": float(safe[k][stay].mean()),
                                      **{s: float(safe[k][m & opp].mean()) for s, m in strat.items() if (m & opp).any()}}
                                  for k in names},
                "per_seed": per_seed, "calibration": calibration,
                "chosen_histogram": {a: h.tolist() for a, h in histogram.items()},
                "contrasts_zombie": {k: test(a, b, zombie) for k, (a, b) in
                                     {"transition": ("all_rank", "root_rank"), "all_rank_vs_root_tokens": ("all_rank", "root_tokens"),
                                      "all_bce_vs_uniform": ("all_bce", "uniform")}.items()},
                "all_rank_vs_root_tokens": test("all_rank", "root_tokens", opp),
                "traces": traces}
    evidence = {"schema": "d4mj_frozen_heads_v1", "status": "SEALED: new seed block, rules committed before collection",
                "script_sha256": _sha256(Path(__file__)), "world_sha256": _sha256(WORLDS / "T.pt"),
                "judge_store": str(SEALED), "judge_manifest": manifest,
                "roots": {"fit": len(pf), "dev": len(pd), "judge": len(pj), "opportunity": int(opp.sum()),
                          "zombie_opportunity": int(zombie.sum()), "stay_kills_move_survives": int(stay.sum())},
                "prior_action": prior, "rules": rules, "readings": readings, "reported": reported}
    (HERE / "evidence/frozen_heads.json").write_text(json.dumps(evidence, indent=2) + "\n")
    log(status="frozen_heads_complete", **{k: v for k, v in readings.items() if k != "labels"}, labels=readings["labels"])


if __name__ == "__main__":
    main()
