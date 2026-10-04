"""Can ANY head on the canonical H2 world's input or state choose safely on zombie roots?

The agent's proposed next run is an alias-free H2 bridge, judged on within-root zombie action choice
against DOWN (always-DOWN, the FIT action prior) and actions_only. Every CLS-based H2 world receives
only z = projector(CLS) for its 4 context frames plus the actions between them. A world's generated
successor is a function of that input, so if no head on the INPUT can beat DOWN on zombie roots, no
alias-free CLS-H2 can pass that criterion, whatever its terminal layout. The two sealed boundary blocks
already show z (last frame only) at the prior on zombie roots (51k: 0.542 vs 0.567; 53k: 0.555 vs
0.538); this measures the world's full input, and its state, directly.

Model: the canonical H2 bridge checkpoint (sha pinned). `frozen_ladder.materialize` builds every
rung exactly as the gate does: 4 observed frames through `teacher`, one `advance` per action.
Heads: frozen_ladder's harness, unchanged -- expected-risk pair ranking over all 17 actions (the most
favourable, counterfactual supervision), AdamW 1e-3 / 1e-4, 3,000 updates, three seeds. Fit on the
partition's FIT-train roots (observability FIT set, 32-key P); select on FIT-dev fork rows (realized
death); never the gate-reserved or unallocated seeds. Judged on the 54,000-54,395 block
(`observe_fresh_v5`), the block the agent's fork check used: EXPLORATORY, it has been read before.

Arms:
  z4               the world's latent input: z of the 4 context frames (768)                -> 17
  z4_actions       + the 3 actions between them (the world's full input)                    -> 17
  root_features    the world's own readout at the root, after the 4-frame teacher pass      -> 17
  generated_z      the world's generated successor per action, shared branch head            -> 1
  generated_features  H2's agent readout of each generated successor, shared branch head      -> 1
  tokens_attn      the root's patch tokens (positive control: the hazard is visible)          -> 17
  actions_only     the last 4 actions                                                          -> 17
  prior            DOWN

Declared readings (committed before the run), zombie-adjacent opportunity roots, paired
episode-seed-clustered 95% intervals:
  tokens_attn - prior not resolved > 0                         -> void (harness cannot see zombies)
  none of z4, z4_actions, root_features resolved > 0 vs prior  -> cls_input_bound_blocks_zombie_criterion
  otherwise                                                     -> headroom_exists
Reported: every arm overall and by stratum; generated_* vs z4_actions (retention) and vs prior;
per-seed means.
"""

import json
import os
import sys
import time
from pathlib import Path

import torch
from torch import nn

ROOT = Path("/home/antithetical/EPITA/PERSO/DynamicHorizons-Mamba-JEPA")
HERE = Path(__file__).parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

from d4mj.data import _sha256
from d4mj.lewm_diagnostics import FORK_STORE

N, SEEDS, STEPS = 17, 3, 3000
STORE = ROOT / "artifacts/eda/observe_fresh_v5"
CHECKPOINT = ROOT / "artifacts/lewm_m4_canonical/raw/bridge/step-002000.pt"


def main():
    started = time.time()
    log = lambda **kw: print(json.dumps({**kw, "seconds": round(time.time() - started, 1)}), flush=True)
    device = torch.device("cuda")
    os.chdir(ROOT)
    from boundary import judge_store
    from confirm import seeds_for
    from d4mj.experiments import _load_bridge_parent
    from frozen_heads import dev_rows
    from frozen_ladder import materialize, scores, standardize, strata, train
    from ladder import paired
    from observability import expected_safe, load

    partition = json.loads((HERE / "evidence/root_partition.json").read_text())
    fit_seeds, unallocated = seeds_for(partition, FORK_STORE)
    dev_seeds = sorted(partition["fit_dev"]["seeds"])
    if (set(partition["reserved_for_gate"]["seeds"]) | set(unallocated)) & (set(fit_seeds) | set(dev_seeds)):
        raise SystemExit("partition overlap")
    if _sha256(CHECKPOINT) != json.loads((HERE / "evidence/confirm.json").read_text())["checkpoint_sha256"]:
        raise SystemExit("checkpoint differs from the one every earlier rung used")
    fit, dev = load(fit_seeds)["fit"], dev_rows(dev_seeds)
    judge, manifest, files = judge_store(STORE)
    bundle, heads, _ = _load_bridge_parent(CHECKPOINT)
    bundle.encoder.freeze()
    bundle.world.eval()
    for module in bundle.world.modules():
        if isinstance(module, nn.BatchNorm1d):
            module.eval()
    for data in (fit, dev, judge):
        data.update(materialize(bundle, data["frames"], data["actions"]))
        data["z4_actions"] = torch.cat([data["z4"], data["actions"][:, -3:].flatten(1)], 1)
        data["actions_only"] = data["actions"][:, -4:].flatten(1)
        del data["frames"]
    del bundle, heads
    torch.cuda.empty_cache()
    log(stage="materialized", fit=len(fit["seed"]), dev=len(dev["seed"]), judge=len(judge["seed"]), files=files)

    p = torch.cat([fit["p_death1"], dev["p_death1"]])
    train_rows, hold_rows = torch.arange(len(fit["seed"])), torch.arange(len(fit["seed"]), len(p))
    pj, seeds = judge["p_death1"], judge["seed"]
    prior = int(fit["p_death1"][fit["p_death1"].amax(1) > fit["p_death1"].amin(1)].mean(0).argmin())
    safe = {"prior": expected_safe(-nn.functional.one_hot(torch.full((len(pj),), prior), N).float(), pj)[0]}
    _, opp = expected_safe(pj, pj)
    arms = {"z4": ("vector", "z4"), "z4_actions": ("vector", "z4_actions"), "root_features": ("vector", "root_features"),
            "generated_z": ("branch", "generated_z"), "generated_features": ("branch", "generated_features"),
            "tokens_attn": ("tokens_attn", "tokens1"), "actions_only": ("vector", "actions_only")}
    per_seed = {}
    for arm, (kind, key) in arms.items():
        xf = torch.cat([fit[key], dev[key]]).float()
        xj = judge[key].float()
        mean, scale = standardize(xf, train_rows)
        xf, xj = (xf - mean) / scale, (xj - mean) / scale
        runs = []
        for seed in range(SEEDS):
            model, _ = train(kind, xf.shape[1:], xf, p, train_rows, hold_rows, seed=seed, device=device, steps=STEPS)
            runs.append(expected_safe(scores(model, xj, torch.arange(len(pj)), device), pj)[0])
            del model
        safe[arm] = torch.stack(runs).mean(0)
        per_seed[arm] = [float(r[opp].mean()) for r in runs]
        log(arm=arm, safe=round(float(safe[arm][opp].mean()), 4), per_seed=[round(v, 4) for v in per_seed[arm]])

    strat = strata(judge["visible"])
    zombie = opp & strat["zombie_adjacent"]
    test = lambda a, b, m: paired(safe[a][m], safe[b][m], seeds[m], draws=1000, seed=20261007)
    up = lambda r: r["difference"] is not None and r["difference"] > 0 and r["excludes_zero"]
    rules = {"R0_tokens_attn_vs_prior_zombie": test("tokens_attn", "prior", zombie),
             **{f"{a}_vs_prior_zombie": test(a, "prior", zombie) for a in ("z4", "z4_actions", "root_features")}}
    reading = ("void" if not up(rules["R0_tokens_attn_vs_prior_zombie"]) else
               "headroom_exists" if any(up(rules[f"{a}_vs_prior_zombie"]) for a in ("z4", "z4_actions", "root_features"))
               else "cls_input_bound_blocks_zombie_criterion")
    reported = {
        "expected_safe": {k: {"overall": float(v[opp].mean()), "zombie": float(v[zombie].mean()),
                              **{s: float(v[m & opp].mean()) for s, m in strat.items() if (m & opp).any()}}
                          for k, v in safe.items()},
        "per_seed": per_seed,
        "contrasts": {f"{a}_vs_{b}": {"overall": test(a, b, opp), "zombie": test(a, b, zombie)}
                      for a, b in (("generated_z", "prior"), ("generated_features", "prior"),
                                   ("generated_features", "z4_actions"), ("generated_z", "z4_actions"),
                                   ("z4_actions", "actions_only"), ("tokens_attn", "z4_actions"))}}
    evidence = {"schema": "d4mj_h2_bound_v1", "status": "EXPLORATORY: 54k block read before",
                "script_sha256": _sha256(Path(__file__)), "checkpoint_sha256": _sha256(CHECKPOINT),
                "judge_manifest": manifest, "roots": {"fit": len(train_rows), "dev": len(hold_rows), "judge": len(pj),
                                                      "opportunity": int(opp.sum()), "zombie": int(zombie.sum())},
                "prior_action": prior, "rules": rules, "reading": reading, "reported": reported}
    (HERE / "evidence/h2_bound.json").write_text(json.dumps(evidence, indent=2) + "\n")
    log(status="h2_bound_complete", reading=reading)


if __name__ == "__main__":
    main()
