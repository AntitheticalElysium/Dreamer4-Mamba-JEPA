"""Step-4 runner: GRU vs Mamba-2 on the shared-global topology.

Implements reviews/2026-07-14-step4-protocol.md plus the 2026-07-15 amendment
(hierarchical statistics, shuffled controls x3 seeds, GRU-72 conditional
capacity control). Executable checks demanded by the companion audit:

  CHECK-1 paired init: all shared non-temporal parameters are COPIED from a
          per-seed reference initialization; names, shapes and a sha256 over
          values are asserted identical across arms of the same seed.
  CHECK-2 replay pairing: a running sha256 over every sampled batch's
          (actions, rewards, continues) — taken BEFORE action shuffling — is
          recorded per arm and asserted identical across arms of a seed.
  CHECK-3 blindness: the final bundle file is opened only after every arm's
          16k checkpoint exists on disk; its sha256 must match the manifest.
  CHECK-4 evaluation runs under world.eval() (+ no_grad in symmetric_eval).
  CHECK-5 provenance: every checkpoint stores HEAD, software versions, the
          full ModelConfig, encoder/bundle hashes, total+component loss
          histories, NumPy/torch-CPU/CUDA RNG states, VRAM, param counts.

Smoke mode (--smoke) trains 30 steps per arm and evaluates on the MONITOR
bundle (seeds 21-24) only; the final bundle is never touched, so blindness is
preserved while every check executes.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

COMPACT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = COMPACT_ROOT.parent
sys.path.insert(0, str(COMPACT_ROOT))
sys.path.insert(0, str(COMPACT_ROOT / "verification"))

from data import Episode, EpisodeReplay  # noqa: E402
from model import LossConfig, ModelConfig  # noqa: E402
from ssl_ijepa import IJEPAPretrainer  # noqa: E402
from step3_temporal import load_scaled_data  # noqa: E402
from causal_stage_a import stage_a_model  # noqa: E402
from fork_oracle_v2 import BUNDLE, ENCODER_CKPT, sha256_file  # noqa: E402
from consolidation import (  # noqa: E402
    ARTIFACTS, build_world, seed_level_summary, symmetric_eval)

FINAL_BUNDLE = REPO_ROOT / "data" / "final_bundle_79_94.pt"
FINAL_MANIFEST = REPO_ROOT / "reviews" / "artifacts" / "final_bundle_79_94.manifest.json"
TRAIN_SEEDS = (101, 202, 303)
STEPS_TOTAL, CKPT_AT = 16_000, (8_000, 16_000)
REPORT = ARTIFACTS / "step4_report.json"

# (name-template, backend, global_hidden, shuffled)
ARM_SPECS = (
    ("M1_gru64_s{seed}", "global_gru", 64, False),
    ("M2_mamba2_s{seed}", "global_mamba2", 64, False),
    ("M3_gru64_shuf_s{seed}", "global_gru", 64, True),
    ("M3_mamba2_shuf_s{seed}", "global_mamba2", 64, True),
)
# Conditional capacity control (2026-07-15 amendment): run ONLY if the Mamba
# family wins the pooled backend verdict. 245,123 params vs Mamba's 245,083.
GRU72_SPEC = ("M4_gru72_s{seed}", "global_gru", 72, False)


def git_head() -> str:
    return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                          text=True, cwd=REPO_ROOT).stdout.strip()


def shared_param_digest(world) -> str:
    h = hashlib.sha256()
    for n, p in sorted(world.named_parameters()):
        if not n.startswith("temporal."):
            h.update(n.encode())
            h.update(p.detach().cpu().to(torch.float32).numpy().tobytes())
    return h.hexdigest()


def make_paired_world(backend, hidden, seed, reference_shared, device):
    """CHECK-1: construct the arm, then overwrite every shared non-temporal
    parameter from the per-seed reference so arms differ only in the
    temporal core (module construction order consuming RNG becomes moot)."""
    torch.manual_seed(seed)
    world = build_world(backend, hidden, device)
    with torch.no_grad():
        for n, p in world.named_parameters():
            if not n.startswith("temporal."):
                ref = reference_shared[n]
                assert ref.shape == p.shape, f"shape mismatch {n}"
                p.copy_(ref.to(device))
    return world


def train_arm(name, backend, hidden, seed, shuffled, train_data,
              reference_shared, monitor_anchors, encoder, device, steps):
    replay = EpisodeReplay(capacity_steps=500_000)
    for ep in train_data:
        replay.add(Episode(**ep))
    world = make_paired_world(backend, hidden, seed, reference_shared, device)
    shared_digest = shared_param_digest(world)
    n_params = sum(p.numel() for p in world.parameters() if p.requires_grad)
    weights = LossConfig()   # validated defaults: rollout on, streamwise regs off
    trainable = [p for p in world.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=1e-4)
    rng = np.random.default_rng(seed)
    replay_hash = hashlib.sha256()   # CHECK-2
    histories: dict[str, list[float]] = {}
    monitor = []
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    for step in range(1, steps + 1):
        batch = replay.sample(batch=4, observations=16, device=device, rng=rng)
        for key in ("actions", "rewards", "continues"):
            replay_hash.update(batch[key].detach().cpu().numpy().tobytes())
        if shuffled:
            batch["actions"] = batch["actions"].roll(1, 0)
            batch["previous_actions"] = batch["previous_actions"].roll(1, 0)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = world(batch, weights)
        optimizer.zero_grad(set_to_none=True)
        out.loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 100.0)
        optimizer.step()
        world.mark_parameters_updated()
        histories.setdefault("total", []).append(float(out.loss.detach()))
        for key, value in out.metrics.items():
            histories.setdefault(key, []).append(float(value))
        if step in CKPT_AT or step == steps:
            world.eval()
            causal, _ = stage_a_model(world, encoder, monitor_anchors, device)
            world.train()
            monitor.append({"step": step,
                            "monitor_retrieval": causal["retrieval_4way_mean"],
                            "monitor_separation": causal["matched_separation_mean"]})
            torch.save(
                {"trainable": {n: p.detach().cpu() for n, p in world.named_parameters()
                               if p.requires_grad},
                 "optimizer": optimizer.state_dict(),
                 "provenance": {   # CHECK-5
                     "head": git_head(),
                     "torch": torch.__version__,
                     "cuda": torch.version.cuda,
                     "model_config": dataclasses.asdict(world.cfg),
                     "loss_config": dataclasses.asdict(weights),
                     "encoder_sha256": sha256_file(ENCODER_CKPT),
                     "arm": {"name": name, "backend": backend,
                             "global_hidden": hidden, "seed": seed,
                             "shuffled": shuffled, "steps": step,
                             "trainable_params": n_params},
                     "shared_param_digest": shared_digest,
                     "replay_stream_digest": replay_hash.hexdigest(),
                     "numpy_rng": rng.bit_generator.state,
                     "torch_rng_cpu": torch.get_rng_state(),
                     "torch_rng_cuda": torch.cuda.get_rng_state(),
                     "peak_vram_mib": round(torch.cuda.max_memory_allocated() / 2**20, 1),
                 },
                 "loss_histories": histories,
                 "monitor": monitor},
                ARTIFACTS / f"step4_{name}_{step}.pt")
            print(f"[{name}] {step}: monitor retrieval "
                  f"{monitor[-1]['monitor_retrieval']:.3f}", flush=True)
    minutes = round((time.perf_counter() - started) / 60, 1)
    info = {"trainable_params": n_params, "train_minutes": minutes,
            "shared_param_digest": shared_digest,
            "replay_stream_digest": replay_hash.hexdigest(),
            "peak_vram_mib": round(torch.cuda.max_memory_allocated() / 2**20, 1),
            "monitor": monitor}
    del replay
    return world, info


def cluster_bootstrap_ci(diffs_by_env: dict, iters=10_000, seed=0):
    """Env-seed-clustered bootstrap CI on the mean paired difference."""
    rng = np.random.default_rng(seed)
    clusters = [np.asarray(v, dtype=np.float64) for v in diffs_by_env.values()]
    stats = np.empty(iters)
    for i in range(iters):
        picked = rng.integers(len(clusters), size=len(clusters))
        stats[i] = float(np.mean(np.concatenate([clusters[j] for j in picked])))
    point = float(np.mean(np.concatenate(clusters)))
    return point, [float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))]


def two_level_bootstrap_ci(per_seed_diffs: dict, iters=10_000, seed=0):
    """CHECK on pseudoreplication: resample TRAINING SEEDS first, then env
    seeds within each picked training seed (companion-specified hierarchy)."""
    rng = np.random.default_rng(seed)
    seeds = list(per_seed_diffs)
    stats = np.empty(iters)
    for i in range(iters):
        picked_seeds = rng.integers(len(seeds), size=len(seeds))
        vals = []
        for j in picked_seeds:
            clusters = [np.asarray(v, dtype=np.float64)
                        for v in per_seed_diffs[seeds[j]].values()]
            picked_env = rng.integers(len(clusters), size=len(clusters))
            vals.append(np.mean(np.concatenate([clusters[k] for k in picked_env])))
        stats[i] = float(np.mean(vals))
    point = float(np.mean([np.mean(np.concatenate([np.asarray(v) for v in d.values()]))
                           for d in per_seed_diffs.values()]))
    return point, [float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))]


def backend_verdict(rows_by_arm: dict, metric="retrieval_all"):
    """Primary decision rule (pre-declared, 2026-07-15 amendment): all-token
    symmetric retrieval, paired per anchor. Mamba 'wins' only if the pooled
    two-level bootstrap lower bound is positive AND all three per-training-seed
    paired differences are positive. Anything else is parity."""
    per_seed = {}
    per_seed_diffs = {}
    for seed in TRAIN_SEEDS:
        gru = rows_by_arm[f"M1_gru64_s{seed}"]
        mamba = rows_by_arm[f"M2_mamba2_s{seed}"]
        assert len(gru) == len(mamba)
        diffs_by_env: dict = {}
        for rg, rm in zip(gru, mamba):
            assert rg["anchor"] == rm["anchor"] and rg["env_seed"] == rm["env_seed"]
            if rg[metric] is None or rm[metric] is None:
                continue
            diffs_by_env.setdefault(rg["env_seed"], []).append(
                rm[metric] - rg[metric])
        point, ci = cluster_bootstrap_ci(diffs_by_env, seed=seed)
        per_seed[str(seed)] = {"diff": point, "ci95_env_clustered": ci}
        per_seed_diffs[seed] = diffs_by_env
    pooled_point, pooled_ci = two_level_bootstrap_ci(per_seed_diffs)
    seed_means = np.array([per_seed[str(s)]["diff"] for s in TRAIN_SEEDS])
    t_mean = float(seed_means.mean())
    t_se = float(seed_means.std(ddof=1) / np.sqrt(len(seed_means)))
    verdict = "mamba_wins" if (pooled_ci[0] > 0 and (seed_means > 0).all()) else (
        "gru_wins" if (pooled_ci[1] < 0 and (seed_means < 0).all()) else "parity")
    return {"metric": metric, "per_training_seed": per_seed,
            "pooled_two_level": {"diff": pooled_point, "ci95": pooled_ci},
            "seed_level_t": {"mean": t_mean, "se": t_se, "n": len(seed_means),
                             "ci95": [t_mean - 4.303 * t_se, t_mean + 4.303 * t_se]},
            "verdict": verdict}


def family_gates(rows, control_rows_by_seed):
    """G-a retrieval >=27% & seed-level LB >25.5%; G-b separation LB>0;
    G-c >= backend-matched shuffled control +1.5pts; G-d common-mask changed
    LB >25.5%."""
    s_all = seed_level_summary(rows, "retrieval_all")
    s_sep = seed_level_summary(rows, "separation_all")
    s_chg = seed_level_summary(rows, "retrieval_changed")
    control_means = [seed_level_summary(c, "retrieval_all")["mean"]
                     for c in control_rows_by_seed]
    control_mean = float(np.mean(control_means))
    return {
        "retrieval_all": s_all, "separation_all": s_sep,
        "retrieval_changed": s_chg, "control_mean": control_mean,
        "G_a": bool(s_all["mean"] >= 0.27 and s_all["ci95"][0] > 0.255),
        "G_b": bool(s_sep["ci95"][0] > 0),
        "G_c": bool(s_all["mean"] >= control_mean + 0.015),
        "G_d": bool(s_chg["ci95"][0] > 0.255),
    }


def main(smoke=False):
    device = torch.device("cuda")
    steps = 30 if smoke else STEPS_TOTAL
    train, _ = load_scaled_data()
    monitor_anchors = torch.load(BUNDLE, weights_only=False)
    pretrainer = IJEPAPretrainer(
        ModelConfig(temporal_backend="gru", predictor="deterministic", mask_ratio=0.0))
    pretrainer.load_state_dict(
        torch.load(ENCODER_CKPT, weights_only=False)["pretrainer"], strict=True)
    encoder = pretrainer.target_encoder.to(device).eval()

    report_path = REPORT if not smoke else ARTIFACTS / "step4_smoke_report.json"
    report = (json.loads(report_path.read_text()) if report_path.exists()
              else {"protocol": "reviews/2026-07-14-step4-protocol.md (+2026-07-15 amendment)",
                    "head_commit": git_head(), "smoke": smoke,
                    "hashes": {"encoder": sha256_file(ENCODER_CKPT),
                               "monitor_bundle": sha256_file(BUNDLE)},
                    "arms": {}})

    # ---------------- training phase (no final-bundle access) ----------------
    arm_list = [(tmpl.format(seed=seed), backend, hidden, seed, shuffled)
                for seed in TRAIN_SEEDS
                for tmpl, backend, hidden, shuffled in ARM_SPECS]
    for seed in TRAIN_SEEDS:
        torch.manual_seed(seed)
        reference = build_world("global_gru", 64, device)
        reference_shared = {n: p.detach().cpu().clone()
                            for n, p in reference.named_parameters()
                            if not n.startswith("temporal.")}
        del reference
        torch.cuda.empty_cache()
        for name, backend, hidden, arm_seed, shuffled in arm_list:
            if arm_seed != seed or name in report["arms"]:
                continue
            world, info = train_arm(name, backend, hidden, seed, shuffled, train,
                                    reference_shared, monitor_anchors, encoder,
                                    device, steps)
            report["arms"][name] = info
            report_path.write_text(json.dumps(report, indent=2, default=str))
            del world
            torch.cuda.empty_cache()
        # CHECK-1 + CHECK-2 assertions within the seed
        shared = {report["arms"][n]["shared_param_digest"]
                  for n in report["arms"] if n.endswith(f"_s{seed}")}
        replays = {report["arms"][n]["replay_stream_digest"]
                   for n in report["arms"] if n.endswith(f"_s{seed}")}
        assert len(shared) == 1, f"seed {seed}: shared-init digests diverge {shared}"
        assert len(replays) == 1, f"seed {seed}: replay streams diverge {replays}"

    # ---------------- blind evaluation phase ----------------
    ckpt_step = steps if smoke else STEPS_TOTAL
    missing = [name for name, *_ in arm_list
               if not (ARTIFACTS / f"step4_{name}_{ckpt_step}.pt").exists()]
    assert not missing, f"CHECK-3 blindness: arms incomplete {missing}"
    if smoke:
        final_anchors = monitor_anchors   # final bundle NEVER opened in smoke
        report["eval_bundle"] = "monitor_21_24"
    else:
        manifest = json.loads(FINAL_MANIFEST.read_text())
        actual = sha256_file(FINAL_BUNDLE)
        assert actual == manifest["sha256"], "final bundle hash != manifest"
        report["hashes"]["final_bundle"] = actual
        final_anchors = torch.load(FINAL_BUNDLE, weights_only=False)

    rows_by_arm = {}
    for name, backend, hidden, seed, shuffled in arm_list:
        ckpt = torch.load(ARTIFACTS / f"step4_{name}_{ckpt_step}.pt",
                          weights_only=False)
        world = build_world(backend, hidden, device)
        with torch.no_grad():
            for n, p in world.named_parameters():
                if n in ckpt["trainable"]:
                    p.copy_(ckpt["trainable"][n].to(device))
        world.eval()   # CHECK-4
        rows = symmetric_eval(world, encoder, final_anchors, device)
        rows_by_arm[name] = rows
        suffix = "_smoke" if smoke else ""
        (ARTIFACTS / f"step4_rows_{name}{suffix}.json").write_text(json.dumps(rows))
        del world
        torch.cuda.empty_cache()

    report["family_gates"] = {}
    for family, backend_tag in (("M1_gru64", "gru64"), ("M2_mamba2", "mamba2")):
        fam_rows = sum((rows_by_arm[f"{family}_s{s}"] for s in TRAIN_SEEDS), [])
        controls = [rows_by_arm[f"M3_{backend_tag}_shuf_s{s}"] for s in TRAIN_SEEDS]
        report["family_gates"][family] = family_gates(fam_rows, controls)
    report["backend_verdict"] = backend_verdict(rows_by_arm)
    report["backend_verdict_changed"] = backend_verdict(
        rows_by_arm, metric="retrieval_changed")
    report_path.write_text(json.dumps(report, indent=2, default=str))
    print(json.dumps({"verdict": report["backend_verdict"]["verdict"],
                      "pooled": report["backend_verdict"]["pooled_two_level"]},
                     indent=2))
    if report["backend_verdict"]["verdict"] == "mamba_wins" and not smoke:
        print("NOTE: run the pre-registered GRU-72 capacity control (M4) "
              "before attributing the win to the backend.")


if __name__ == "__main__":
    main(smoke="--smoke" in sys.argv)
