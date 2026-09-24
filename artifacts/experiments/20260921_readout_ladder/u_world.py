"""Does the old u->u world's GENERATED u keep the root-side safety signal the root u carries?

COMPACTNESS.md found the literal old `u` -- the frozen TC-consecutive encoder's 4x4 pooled grid
through its persisted PCA-192 -- ranks root actions at 0.804 on the observability roots, as well as
the grid it compresses. The `u->u` world trained on that state (10k factual next-u MSE,
`../20260918_matched_10k/evidence/world_u_u.pt`) reached 23.0/36 generated against a 24.3 root+action
control -- on an older 36-root panel with a different evaluator. This joins the two on one
population, in one harness.

Construction, reproducing the matched run (`matched_10k.py`: PatchPCAEncoder over the checkpoint's
own bundle; `../20260919_localization_ladder/phase6.py` for the rollout): u over the last 4 context
frames, prefill with the 3 actions between them, then one advance per action. Only
`LeWMWorld.teacher` / `.advance` (`d4mj/lewm.py`) and `mamba_recurrence.py` are called, both
byte-identical to what the old checkpoint recorded; the 17-way fan repeats the context instead of
using the since-changed `world_api` adapter. The world config is the checkpoint's own, asserted to
rebuild exactly; world and encoder weights load strictly. The encoder load is `compactness.old_encoder`.

Arms, frozen-ladder harness (expected-risk ranking over 32-key P, 512-wide head, 3,000 updates,
inner-FIT selection, three probe seeds), observability roots pinned by content hash, EXPLORATORY:

  root_u        u of the last root frame, root-level head                 = COMPACTNESS pca_u_old
  generated_u   the world's predicted next u per action, one shared branch head
  history       the world's Mamba output per action (final_norm), shared branch head
  real_u        u of each action's REAL successor -- hindsight positive control

Declared reading, one-step death, paired seed-clustered 95%:
  real_u does not beat the prior                                -> evaluator_inadequate
  generated_u beats the prior and is not resolved below root_u  -> u_world_preserves_root_signal
  generated_u is resolved below root_u                          -> u_world_loses_root_signal
  otherwise                                                     -> mixed
"""

import argparse
import hashlib
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

import torch
from torch import nn

ROOT = Path("/home/antithetical/EPITA/PERSO/DynamicHorizons-Mamba-JEPA")
HERE = Path(__file__).parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

from d4mj.data import _sha256
from d4mj.lewm_diagnostics import FORK_STORE

from compactness import OLD, OLD_CHECKPOINT, apply_pca, old_encoder  # noqa: E402
from confirm import seeds_for  # noqa: E402
from frozen_ladder import scores, standardize, strata, train  # noqa: E402
from ladder import paired  # noqa: E402
from observability import FIT_STATE, FRESH, expected_safe, load  # noqa: E402

N = 17
WORLD = ROOT / "artifacts/experiments/20260918_matched_10k/evidence/world_u_u.pt"


def successors(fit_seeds):
    """Real successor frames, in exactly `observability.load`'s row order; identity re-derived."""
    store = {int(p.stem.split("-")[1]): p for p in FORK_STORE.glob("seed-*.pt")}
    state = {int(p.stem.split("-")[1]): p for p in FIT_STATE.glob("seed-*.pt")}
    out = {}
    for split, files in (("fit", [(store[s], state[s]) for s in sorted(fit_seeds)]),
                         ("judge", [(p, None) for p in sorted(FRESH.glob("seed-*.pt"))])):
        frames, identity = [], []
        for pixel_file, state_file in files:
            pixels = {int(r["step"]): r for r in torch.load(pixel_file, weights_only=False)}
            order = (pixels if state_file is None else
                     {int(r["step"]): r for r in torch.load(state_file, weights_only=False)})
            for step, f in order.items():
                frames.append(pixels[step]["successors"])
                identity.append((int(f["seed"]), int(pixels[step]["frames"].sum())))
        out[split] = (torch.stack(frames), hashlib.sha256(repr(identity).encode()).hexdigest())
    return out


@torch.no_grad()
def u_world_features(encoder, world, pca, frames, actions, succ, device, batch=16):
    """root u, generated u, history and real u for every root and all 17 actions."""
    out = {k: [] for k in ("root_u", "generated_u", "history", "real_u")}
    for i in range(0, len(frames), batch):
        f = frames[i:i + batch, -4:].to(device)
        n = len(f)
        past = actions[i:i + batch, -3:].argmax(-1)
        if bool((past >= N).any()):
            raise SystemExit("a context action is BOS")
        _, _, grid = encoder.export(f, grid=4)
        u = apply_pca(pca, grid.flatten(2).cpu()).to(device)                  # [n, 4, 192]
        state = world.teacher(u.repeat_interleave(N, 0)[:, :, None], past.to(device).repeat_interleave(N, 0)).state
        acts = torch.arange(N, device=device).repeat(n)[:, None]
        advanced, _ = world.advance(state, acts)
        _, _, sgrid = encoder.export(succ[i:i + batch].flatten(0, 1).unsqueeze(1).to(device), grid=4)
        real = apply_pca(pca, sgrid.flatten(2).cpu())                         # [n*17, 1, 192]
        out["root_u"].append(u[:, -1].cpu())
        out["generated_u"].append(advanced.latent[:, 0, 0].reshape(n, N, -1).cpu())
        out["history"].append(advanced.history[:, 0].reshape(n, N, -1).cpu())
        out["real_u"].append(real[:, 0].reshape(n, N, -1))
    return {k: torch.cat(v) for k, v in out.items()}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--partition", type=Path, default=HERE / "evidence/root_partition.json")
    parser.add_argument("--out", type=Path, default=HERE / "evidence")
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--probe-seeds", type=int, default=3)
    parser.add_argument("--draws", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260924)
    parser.add_argument("--name", default="u_world")
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    started = time.time()
    log = lambda **kw: print(json.dumps({**kw, "seconds": round(time.time() - started, 1)}), flush=True)
    device = torch.device("cuda")

    fit_seeds, _ = seeds_for(json.loads(args.partition.read_text()), FORK_STORE)
    data = load(fit_seeds)
    recorded = json.loads((HERE / "evidence/observability.json").read_text())["identity"]
    if {s: data[s]["identity"] for s in data} != recorded:
        raise SystemExit("data differs from what the observability test measured")
    succ = successors(fit_seeds)
    if {s: succ[s][1] for s in succ} != recorded:
        raise SystemExit("successor rows are not aligned with the measured rows")

    from d4mj.config import config_from_dict
    from d4mj.lewm import LeWMWorld
    stored = torch.load(OLD_CHECKPOINT, map_location="cpu", weights_only=False)
    config = config_from_dict(stored["config"])
    if json.loads(json.dumps(asdict(config.dynamics))) != json.loads(json.dumps(stored["config"]["dynamics"])):
        raise SystemExit("current config code does not rebuild the stored dynamics settings")
    del stored
    saved = torch.load(WORLD, map_location="cpu", weights_only=False)
    if (saved.get("source"), saved.get("target")) != ("u", "u"):
        raise SystemExit("expected the u->u world")
    world = LeWMWorld(config).to(device)
    world.load_state_dict(saved["state_dict"], strict=True)
    world.eval()
    encoder = old_encoder(device)
    pca = torch.load(OLD / "evidence/state_cache.pt", weights_only=False, map_location="cpu")["pca"]
    for split in data:
        data[split].update(u_world_features(encoder, world, pca, data[split]["frames"], data[split]["actions"],
                                            succ[split][0], device))
    del encoder, world, succ
    torch.cuda.empty_cache()
    log(stage="features", fit=len(data["fit"]["seed"]), judge=len(data["judge"]["seed"]))

    groups = data["fit"]["seed"].unique()
    held = set(groups[torch.randperm(len(groups), generator=torch.Generator().manual_seed(args.seed))
                      [:max(1, round(0.15 * len(groups)))]].tolist())
    inner = torch.tensor([int(s) in held for s in data["fit"]["seed"]])
    train_rows, hold_rows = torch.where(~inner)[0], torch.where(inner)[0]
    judge_rows = torch.arange(len(data["judge"]["seed"]))
    strat = strata(data["judge"]["visible"])
    seeds = data["judge"]["seed"]

    report = {}
    for outcome in ("death1", "death2"):
        pf, pj = data["fit"][f"p_{outcome}"], data["judge"][f"p_{outcome}"]
        opp_fit = (pf.amax(1) > pf.amin(1)) & ~inner
        prior = int(pf[opp_fit].mean(0).argmin())
        prior_safe, opp = expected_safe(-nn.functional.one_hot(torch.full((len(pj),), prior), N).float(), pj)
        per_arm, block = {}, {"judge_opportunity_roots": int(opp.sum()),
                              "prior_expected_safe": float(prior_safe[opp].mean()),
                              "prior_by_stratum": {k: float(prior_safe[v & opp].mean()) for k, v in strat.items()
                                                   if (v & opp).any()}, "arms": {}}
        for arm in ("root_u", "generated_u", "history", "real_u"):
            kind = "vector" if arm == "root_u" else "branch"
            xf, xj = data["fit"][arm], data["judge"][arm]
            mean, scale = standardize(xf, train_rows)
            xf, xj = (xf - mean) / scale, (xj - mean) / scale
            runs, safes, chosen = [], [], []
            for probe in range(args.probe_seeds):
                model, trace = train(kind, xf.shape[1:], xf, pf, train_rows, hold_rows, seed=probe,
                                     device=device, steps=args.steps)
                sj = scores(model, xj, judge_rows, device)
                safe, _ = expected_safe(sj, pj)
                fsafe, fopp = expected_safe(scores(model, xf, train_rows, device), pf[train_rows])
                runs.append({"judge_expected_safe": float(safe[opp].mean()), "fit_expected_safe": float(fsafe[fopp].mean()),
                             "inner_safe": trace["inner_safe"], "selected_step": trace["selected_step"],
                             "parameters": trace["parameters"]})
                safes.append(safe)
                chosen.append(sj[opp].argmin(1))
                del model
            per_arm[arm] = torch.stack(safes).mean(0)
            block["arms"][arm] = {"runs": runs, "judge_expected_safe": float(per_arm[arm][opp].mean()),
                                  "vs_prior": paired(per_arm[arm][opp], prior_safe[opp], seeds[opp],
                                                     draws=args.draws, seed=args.seed + 11),
                                  "by_stratum": {k: float(per_arm[arm][v & opp].mean()) for k, v in strat.items()
                                                 if (v & opp).any()},
                                  "chosen_actions": torch.bincount(torch.cat(chosen), minlength=N).tolist()}
            log(stage="arm", outcome=outcome, arm=arm, safe=round(block["arms"][arm]["judge_expected_safe"], 4))
        test = lambda a, b, mask=opp: paired(per_arm[a][mask], (prior_safe if b == "prior" else per_arm[b])[mask],
                                             seeds[mask], draws=args.draws, seed=args.seed + 13)
        z = opp & strat["zombie_adjacent"]
        block["contrasts"] = {"generated_u_vs_root_u": test("generated_u", "root_u"),
                              "generated_u_vs_prior": test("generated_u", "prior"),
                              "history_vs_root_u": test("history", "root_u"),
                              "real_u_vs_prior": test("real_u", "prior"),
                              "generated_u_vs_root_u_zombie": test("generated_u", "root_u", z)}
        report[outcome] = block

    b = report["death1"]
    hold = lambda t: bool(t["difference"] > 0 and t["excludes_zero"])
    below = lambda t: bool(t["difference"] < 0 and t["excludes_zero"])
    c = b["contrasts"]
    reading = ("evaluator_inadequate" if not hold(c["real_u_vs_prior"]) else
               "u_world_loses_root_signal" if below(c["generated_u_vs_root_u"]) else
               "u_world_preserves_root_signal" if hold(c["generated_u_vs_prior"]) else "mixed")
    evidence = {"schema": "d4mj_u_world_v1", "status": "EXPLORATORY: observability roots, already inspected",
                "script_sha256": _sha256(Path(__file__)), "world_sha256": _sha256(WORLD),
                "encoder_checkpoint_sha256": _sha256(OLD_CHECKPOINT), "identity": recorded,
                "reading": reading, "outcomes": report}
    (args.out / f"{args.name}.json").write_text(json.dumps(evidence, indent=2) + "\n")
    log(status="u_world_complete", reading=reading)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
