"""Does the old u->u compression keep the root-side safety signal the full patch grid carries?

The frozen `u->u` world (`../20260917_state_transition`) transitioned `u`: a fixed, TRAIN-fitted,
label-free PCA of the frozen 4x4 pooled patch grid, 3,072 -> 192. It reached 36/36 on real
successors and 23.0/36 generated, below its 24.3 root+action control (`../20260919_localization_ladder`
README), and a 10k follow-up raised generated-state AUC 0.620 -> 0.708 without moving fatal-safe
ranking (`../20260918_matched_10k`). FROZEN_LADDER.md then found the Raw H2 encoder's full patch grid
ranks root actions (0.778) where its CLS does not (0.627). Before any spatial-state architecture,
this separates two readings on those same roots, in the same harness:

  the compression keeps the signal  -> the old u->u failure points at supervision/dynamics: try
                                       all-action consequence training on that architecture first
  the compression loses it          -> a persistently predicted spatial state earns its cost

Arms, the frozen-ladder harness (expected-risk ranking over 32-key P, 512-wide head, 3,000 updates,
inner-FIT selection, three probe seeds), on the observability roots -- pinned by content hash,
EXPLORATORY (already inspected; the 51,000+ block is reserved for boundary.py):

  grid_old     the OLD encoder's 4x4 pooled grid, uncompressed (3,072)   paired_window raw joint 10k
  pca_u_old    that grid through the persisted PCA = the literal old `u` (192)
  grid_raw     the Raw H2 encoder's 4x4 pooled grid, uncompressed (3,072)
  pca_u_raw    the same recipe on Raw H2: label-free PCA-192 fit on FIT non-inner roots only

Paired, seed-clustered contrasts against the prior and between arms, plus against the saved
frozen-ladder rows (patch tokens flat, CLS + pooled, CLS) on the identical judgement order.

Declared reading, one-step death, primary arm the literal old `u`:
  pca_u_old beats the prior AND pca_u_old - grid_old is not resolved below zero
                                                    -> compression_keeps_signal
  grid_old beats the prior AND pca_u_old - grid_old is resolved below zero
                                                    -> compression_loses_signal
  neither                                           -> mixed
The Raw H2 pair is reported beside it as the same test on the current encoder.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from torch import nn

ROOT = Path("/home/antithetical/EPITA/PERSO/DynamicHorizons-Mamba-JEPA")
HERE = Path(__file__).parent
OLD = ROOT / "artifacts/experiments/20260917_state_transition"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

from d4mj.data import _sha256
from d4mj.lewm_diagnostics import FORK_STORE

from confirm import seeds_for  # noqa: E402
from frozen_ladder import materialize, scores, standardize, strata, train  # noqa: E402
from ladder import paired  # noqa: E402
from observability import expected_safe, load  # noqa: E402

N, WIDTH = 17, 192
OLD_CHECKPOINT = ROOT / "artifacts/lewm_gates_20260916/paired_window/raw/joint/step-010000.pt"


def fit_pca(samples, components=WIDTH):
    """`state_transition.fit_pca`, verbatim in effect: label-free basis, null tail dropped."""
    mean = samples.mean(0, keepdim=True)
    _, singular, v = torch.linalg.svd((samples - mean).double(), full_matrices=False)
    scale = singular[:components] / max(len(samples) - 1, 1) ** 0.5
    keep = int((scale > scale[0] * 1e-3).sum())
    return {"mean": mean.float(), "basis": v[:components].T[:, :keep].float(), "rank": keep,
            "components": components}


def apply_pca(pca, x):
    out = torch.zeros(*x.shape[:-1], pca["components"], dtype=torch.float32)
    out[..., :pca["rank"]] = (x - pca["mean"]) @ pca["basis"]
    return out


@torch.no_grad()
def old_grid(frames, device, batch=64):
    """The old encoder's last-frame 4x4 pooled grid, flattened exactly as state_transition did."""
    encoder = old_encoder(device)
    out = []
    for i in range(0, len(frames), batch):
        _, _, patch = encoder.export(frames[i:i + batch, -1:].to(device), grid=4)
        out.append(patch[:, 0].flatten(1).cpu())
    del encoder
    torch.cuda.empty_cache()
    return torch.cat(out)


def old_encoder(device):
    """The 2026-09-16 encoder alone, loaded only because its forward path is provably unchanged.

    The gated loader refuses this checkpoint: the source closure has drifted since, in training,
    gate and data code. The encoder's forward needs none of it. Asserted here rather than assumed:
    `d4mj/lewm.py` (LeWMEncoder, LeWMProjector) and every `transformers` file in the manifest are
    byte-identical to what the checkpoint recorded; the current config code rebuilds exactly the
    stored encoder settings; the weights load strictly. Anything else aborts.
    """
    from dataclasses import asdict
    from d4mj.config import config_from_dict
    from d4mj.lewm import LeWMEncoder
    from d4mj.sources import lewm_source_manifest
    stored = torch.load(OLD_CHECKPOINT, map_location="cpu", weights_only=False)
    config = config_from_dict(stored["config"])
    current = lewm_source_manifest(config)
    forward = ["d4mj/lewm.py"] + [k for k in stored["sources"]["runtime"] if k.startswith("transformers")]
    drift = [k for k in forward if stored["sources"]["runtime"][k] != current["runtime"][k]]
    drift += [s for s in ("pins", "references", "versions", "execution") if stored["sources"][s] != current[s]]
    if drift:
        raise SystemExit(f"old encoder forward path drifted: {drift}")
    rebuilt = json.loads(json.dumps(asdict(config.encoder)))
    if rebuilt != json.loads(json.dumps(stored["config"]["encoder"])):
        raise SystemExit("current config code does not rebuild the stored encoder settings")
    encoder = LeWMEncoder(config).to(device)
    encoder.load_state_dict(stored["modules"]["encoder"], strict=True)
    return encoder.freeze()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--partition", type=Path, default=HERE / "evidence/root_partition.json")
    parser.add_argument("--run", type=Path, default=ROOT / "artifacts/lewm_m4_canonical/raw")
    parser.add_argument("--out", type=Path, default=HERE / "evidence")
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--probe-seeds", type=int, default=3)
    parser.add_argument("--draws", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260924)
    parser.add_argument("--name", default="compactness")
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    started = time.time()
    log = lambda **kw: print(json.dumps({**kw, "seconds": round(time.time() - started, 1)}), flush=True)
    device = torch.device("cuda")

    fit_seeds, _ = seeds_for(json.loads(args.partition.read_text()), FORK_STORE)
    data = load(fit_seeds)
    recorded = json.loads((HERE / "evidence/observability.json").read_text())
    if {s: data[s]["identity"] for s in data} != recorded["identity"]:
        raise SystemExit("data differs from what the observability test and frozen ladder measured")
    checkpoint = args.run / "bridge/step-002000.pt"
    if _sha256(checkpoint) != json.loads((HERE / "evidence/confirm.json").read_text())["checkpoint_sha256"]:
        raise SystemExit("Raw H2 checkpoint differs from the one every earlier rung used")
    old_pca = torch.load(OLD / "evidence/state_cache.pt", weights_only=False, map_location="cpu")["pca"]
    if _sha256(OLD_CHECKPOINT) != torch.load(OLD / "evidence/state_cache.pt", weights_only=False,
                                             map_location="cpu")["checkpoint_sha256"]:
        raise SystemExit("old encoder checkpoint is not the one the persisted PCA was fitted on")

    from d4mj.experiments import _load_bridge_parent
    bundle, heads, _ = _load_bridge_parent(checkpoint)
    bundle.encoder.freeze()
    bundle.world.eval()
    for module in bundle.world.modules():
        if isinstance(module, nn.BatchNorm1d):
            module.eval()
    for split in data:
        data[split]["grid_raw"] = materialize(bundle, data[split]["frames"], data[split]["actions"])["cls_pooled1"][:, WIDTH:]
    del bundle, heads
    torch.cuda.empty_cache()
    for split in data:
        data[split]["grid_old"] = old_grid(data[split]["frames"], device)
        data[split]["pca_u_old"] = apply_pca(old_pca, data[split]["grid_old"])
    log(stage="features", fit=len(data["fit"]["seed"]), judge=len(data["judge"]["seed"]))

    groups = data["fit"]["seed"].unique()
    held = set(groups[torch.randperm(len(groups), generator=torch.Generator().manual_seed(args.seed))
                      [:max(1, round(0.15 * len(groups)))]].tolist())
    inner = torch.tensor([int(s) in held for s in data["fit"]["seed"]])
    train_rows, hold_rows = torch.where(~inner)[0], torch.where(inner)[0]
    raw_pca = fit_pca(data["fit"]["grid_raw"][train_rows])
    for split in data:
        data[split]["pca_u_raw"] = apply_pca(raw_pca, data[split]["grid_raw"])
    judge_rows = torch.arange(len(data["judge"]["seed"]))
    strat = strata(data["judge"]["visible"])
    seeds = data["judge"]["seed"]
    saved = torch.load(HERE / "evidence/frozen_ladder_rows.pt", weights_only=False)

    report = {}
    for outcome in ("death1", "death2"):
        pf, pj = data["fit"][f"p_{outcome}"], data["judge"][f"p_{outcome}"]
        opp_fit = (pf.amax(1) > pf.amin(1)) & ~inner
        prior = int(pf[opp_fit].mean(0).argmin())
        prior_safe, opp = expected_safe(-nn.functional.one_hot(torch.full((len(pj),), prior), N).float(), pj)
        per_arm, block = {}, {"judge_opportunity_roots": int(opp.sum()), "prior_expected_safe": float(prior_safe[opp].mean()),
                              "arms": {}}
        for arm in ("grid_old", "pca_u_old", "grid_raw", "pca_u_raw"):
            xf, xj = data["fit"][arm], data["judge"][arm]
            mean, scale = standardize(xf, train_rows)
            xf, xj = (xf - mean) / scale, (xj - mean) / scale
            runs, safes = [], []
            for probe in range(args.probe_seeds):
                model, trace = train("vector", xf.shape[1:], xf, pf, train_rows, hold_rows, seed=probe,
                                     device=device, steps=args.steps)
                safe, _ = expected_safe(scores(model, xj, judge_rows, device), pj)
                fsafe, fopp = expected_safe(scores(model, xf, train_rows, device), pf[train_rows])
                runs.append({"judge_expected_safe": float(safe[opp].mean()), "fit_expected_safe": float(fsafe[fopp].mean()),
                             "inner_safe": trace["inner_safe"], "selected_step": trace["selected_step"],
                             "parameters": trace["parameters"]})
                safes.append(safe)
                del model
            per_arm[arm] = torch.stack(safes).mean(0)
            block["arms"][arm] = {"runs": runs, "judge_expected_safe": float(per_arm[arm][opp].mean()),
                                  "vs_prior": paired(per_arm[arm][opp], prior_safe[opp], seeds[opp],
                                                     draws=args.draws, seed=args.seed + 11),
                                  "by_stratum": {k: float(per_arm[arm][v & opp].mean()) for k, v in strat.items()
                                                 if (v & opp).any()}}
            log(stage="arm", outcome=outcome, arm=arm, safe=round(block["arms"][arm]["judge_expected_safe"], 4))
        for ref in ("tokens1_flat", "cls_pooled1", "cls1"):
            per_arm[ref] = torch.stack([expected_safe(saved["scores"][f"{outcome}|{ref}|{k}"], pj)[0]
                                        for k in range(3)]).mean(0)
        test = lambda a, b, mask=opp: paired(per_arm[a][mask], per_arm[b][mask], seeds[mask],
                                             draws=args.draws, seed=args.seed + 13)
        z = opp & strat["zombie_adjacent"]
        block["contrasts"] = {"pca_u_old_vs_grid_old": test("pca_u_old", "grid_old"),
                              "pca_u_raw_vs_grid_raw": test("pca_u_raw", "grid_raw"),
                              "grid_old_vs_grid_raw": test("grid_old", "grid_raw"),
                              "grid_raw_vs_tokens1_flat": test("grid_raw", "tokens1_flat"),
                              "pca_u_raw_vs_cls1": test("pca_u_raw", "cls1"),
                              "pca_u_old_vs_grid_old_zombie": test("pca_u_old", "grid_old", z),
                              "pca_u_raw_vs_grid_raw_zombie": test("pca_u_raw", "grid_raw", z)}
        report[outcome] = block

    b = report["death1"]
    beats = lambda arm: bool(b["arms"][arm]["vs_prior"]["difference"] > 0 and b["arms"][arm]["vs_prior"]["excludes_zero"])
    loss = b["contrasts"]["pca_u_old_vs_grid_old"]
    below = bool(loss["difference"] < 0 and loss["excludes_zero"])
    reading = ("compression_keeps_signal" if beats("pca_u_old") and not below else
               "compression_loses_signal" if beats("grid_old") and below else "mixed")
    evidence = {"schema": "d4mj_compactness_v1", "status": "EXPLORATORY: observability roots, already inspected",
                "script_sha256": _sha256(Path(__file__)), "raw_checkpoint_sha256": _sha256(checkpoint),
                "old_checkpoint_sha256": _sha256(OLD_CHECKPOINT), "old_pca_rank": old_pca["rank"],
                "raw_pca_rank": raw_pca["rank"], "identity": {s: data[s]["identity"] for s in data},
                "reading": reading, "outcomes": report}
    (args.out / f"{args.name}.json").write_text(json.dumps(evidence, indent=2) + "\n")
    log(status="compactness_complete", reading=reading)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
