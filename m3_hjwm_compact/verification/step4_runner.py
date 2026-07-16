"""Step-4 runner: GRU vs Mamba-2 on the shared-global topology.

Implements reviews/2026-07-14-step4-protocol.md + the 2026-07-15 amendment,
rebuilt after the companion's second audit (2026-07-16). Executable checks:

  CHECK-1 paired init: every shared non-temporal STATE entry (parameters AND
          buffers, with names/shapes/dtypes) is copied from a per-seed
          reference; a sha256 over that full shared state is asserted
          identical across arms of the same seed.
  CHECK-2 replay pairing: a running sha256 over EVERY tensor of every sampled
          batch (sorted keys, taken before action shuffling) is asserted
          identical across arms of a seed; the replay file's own sha256 is
          recorded and pinned.
  CHECK-3 blindness: the final bundle is opened only after all arm
          checkpoints exist; its sha256 must match the manifest.
  CHECK-4 evaluation under world.eval() (+ no_grad inside symmetric_eval).
  CHECK-5 provenance: full state_dict checkpoints; HEAD + clean-tree
          enforcement (tracked files must be clean unless --smoke); a source
          digest over every module the run imports; Python/torch/NumPy/
          mamba_ssm/crafter/GPU versions; encoder/replay/bundle hashes;
          total+component loss histories; NumPy/CPU/CUDA RNG states; peak
          allocated AND reserved VRAM; checkpoint file hashes in the report.

Strict resume: an arm is skipped only when its report entry AND checkpoint
exist AND the checkpoint's source digest, arm configuration, step count, and
input hashes all match the current run; otherwise it retrains.

Family gates follow the REGISTERED per-training-seed majority rule: G-a..G-d
are decided per seed (env-seed-clustered CIs within each model, G-c against
the SAME-SEED shuffled control), then a 2/3 majority is applied.

Strata (mandatory): day/night and action-effective, with PIXEL-effective
(any alternative suffix changes any final-frame pixel vs true) pre-registered
as the primary action-effective notion and task/outcome-effective secondary.

--gru72 runs the pre-registered conditional capacity control (M4 global-GRU-72
x3 seeds) and the paired M4-vs-M2 verdict.
--smoke trains 30 steps/arm and evaluates on the MONITOR bundle only.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import platform
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
from model import LossConfig, ModelConfig, frozen_dynamics_recipe  # noqa: E402
from ssl_ijepa import IJEPAPretrainer  # noqa: E402
from step3_temporal import TRAIN_40K_CACHE, load_scaled_data  # noqa: E402
from causal_stage_a import stage_a_model  # noqa: E402
from fork_oracle_v2 import BUNDLE, ENCODER_CKPT, sha256_file  # noqa: E402
from consolidation import (  # noqa: E402
    ARTIFACTS, build_world, seed_level_summary, symmetric_eval)

FINAL_BUNDLE = REPO_ROOT / "data" / "final_bundle_79_94.pt"
FINAL_MANIFEST = REPO_ROOT / "reviews" / "artifacts" / "final_bundle_79_94.manifest.json"
TRAIN_SEEDS = (101, 202, 303)
STEPS_TOTAL, CKPT_AT = 16_000, (8_000, 16_000)

# Source files whose behavior the run depends on; hashed into every
# checkpoint. Resume and evaluation REQUIRE digest equality.
SOURCE_FILES = (
    "verification/step4_runner.py", "model.py", "data.py",
    "verification/consolidation.py", "verification/step3_temporal.py",
    "ssl_ijepa.py", "verification/fork_oracle_v2.py",
    "verification/causal_stage_a.py", "verification/representation_control.py",
    "verification/microtest.py",
)

# (name-template, backend, global_hidden, shuffled)
ARM_SPECS = (
    ("M1_gru64_s{seed}", "global_gru", 64, False),
    ("M2_mamba2_s{seed}", "global_mamba2", 64, False),
    ("M3_gru64_shuf_s{seed}", "global_gru", 64, True),
    ("M3_mamba2_shuf_s{seed}", "global_mamba2", 64, True),
)
# Conditional capacity control (pre-registered): 245,123 params vs Mamba's
# 245,083. Runs only via --gru72 after a Mamba family win.
GRU72_SPECS = (("M4_gru72_s{seed}", "global_gru", 72, False),)


# --------------------------------------------------------------------------
# provenance helpers
# --------------------------------------------------------------------------

def git_head() -> str:
    return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                          text=True, cwd=REPO_ROOT).stdout.strip()


def tracked_dirty() -> list[str]:
    out = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"],
                         capture_output=True, text=True, cwd=REPO_ROOT).stdout
    return [line for line in out.splitlines() if line.strip()]


def source_digest() -> str:
    h = hashlib.sha256()
    for rel in SOURCE_FILES:
        h.update(rel.encode())
        h.update((COMPACT_ROOT / rel).read_bytes())
    return h.hexdigest()


def software_versions() -> dict:
    import crafter
    import mamba_ssm
    props = torch.cuda.get_device_properties(0)
    return {
        "python": platform.python_version(),
        "torch": torch.__version__, "cuda": torch.version.cuda,
        "numpy": np.__version__,
        "mamba_ssm": getattr(mamba_ssm, "__version__", "unknown"),
        "crafter": getattr(crafter, "__version__", "unknown"),
        "gpu": props.name,
        "gpu_total_mib": round(props.total_memory / 2**20),
    }


def shared_state_digest(module, exclude_prefix: str = "temporal.") -> str:
    """CHECK-1 digest: full state_dict (parameters AND buffers) outside the
    declared temporal-core difference, with names, shapes, and dtypes."""
    h = hashlib.sha256()
    for name, tensor in sorted(module.state_dict().items()):
        if name.startswith(exclude_prefix):
            continue
        h.update(name.encode())
        h.update(str(tuple(tensor.shape)).encode())
        h.update(str(tensor.dtype).encode())
        h.update(tensor.detach().cpu().to(torch.float32).numpy().tobytes())
    return h.hexdigest()


def hash_batch(h, batch: dict) -> None:
    """CHECK-2: fold EVERY sampled tensor into the running digest."""
    for key in sorted(batch):
        h.update(key.encode())
        h.update(batch[key].detach().cpu().numpy().tobytes())


# --------------------------------------------------------------------------
# training
# --------------------------------------------------------------------------

def make_paired_world(backend, hidden, seed, reference_shared, device):
    torch.manual_seed(seed)
    world = build_world(backend, hidden, device)
    state = world.state_dict()
    for name, tensor in reference_shared.items():
        assert name in state, f"shared entry {name} missing from {backend}"
        assert state[name].shape == tensor.shape, f"shape mismatch {name}"
        state[name] = tensor.to(device=device, dtype=state[name].dtype)
    world.load_state_dict(state, strict=True)
    return world


def arm_provenance(name, backend, hidden, seed, shuffled, step, n_params,
                   shared_digest, replay_digest, world, rng):
    return {
        "head": git_head(), "source_digest": source_digest(),
        "versions": software_versions(),
        "model_config": dataclasses.asdict(world.cfg),
        "loss_config": dataclasses.asdict(frozen_dynamics_recipe()),
        "encoder_sha256": sha256_file(ENCODER_CKPT),
        "replay_file_sha256": sha256_file(TRAIN_40K_CACHE),
        "arm": {"name": name, "backend": backend, "global_hidden": hidden,
                "seed": seed, "shuffled": shuffled, "steps": step,
                "trainable_params": n_params},
        "shared_state_digest": shared_digest,
        "replay_stream_digest": replay_digest,
        "numpy_rng": rng.bit_generator.state,
        "torch_rng_cpu": torch.get_rng_state(),
        "torch_rng_cuda": torch.cuda.get_rng_state(),
        "peak_vram_alloc_mib": round(torch.cuda.max_memory_allocated() / 2**20, 1),
        "peak_vram_reserved_mib": round(torch.cuda.max_memory_reserved() / 2**20, 1),
    }


def train_arm(name, backend, hidden, seed, shuffled, train_data,
              reference_shared, monitor_anchors, encoder, device, steps):
    replay = EpisodeReplay(capacity_steps=500_000)
    for ep in train_data:
        replay.add(Episode(**ep))
    world = make_paired_world(backend, hidden, seed, reference_shared, device)
    shared_digest = shared_state_digest(world)
    n_params = sum(p.numel() for p in world.parameters() if p.requires_grad)
    weights = frozen_dynamics_recipe()
    trainable = [p for p in world.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=1e-4)
    rng = np.random.default_rng(seed)
    replay_hash = hashlib.sha256()
    histories: dict[str, list[float]] = {}
    monitor = []
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    for step in range(1, steps + 1):
        batch = replay.sample(batch=4, observations=16, device=device, rng=rng)
        hash_batch(replay_hash, batch)
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
                {"state_dict": {k: v.detach().cpu()
                                for k, v in world.state_dict().items()},
                 "optimizer": optimizer.state_dict(),
                 "provenance": arm_provenance(
                     name, backend, hidden, seed, shuffled, step, n_params,
                     shared_digest, replay_hash.hexdigest(), world, rng),
                 "loss_histories": histories,
                 "monitor": monitor},
                ARTIFACTS / f"step4_{name}_{step}.pt")
            print(f"[{name}] {step}: monitor retrieval "
                  f"{monitor[-1]['monitor_retrieval']:.3f}", flush=True)
    minutes = round((time.perf_counter() - started) / 60, 1)
    info = {"trainable_params": n_params, "train_minutes": minutes,
            "shared_state_digest": shared_digest,
            "replay_stream_digest": replay_hash.hexdigest(),
            "peak_vram_alloc_mib": round(torch.cuda.max_memory_allocated() / 2**20, 1),
            "peak_vram_reserved_mib": round(torch.cuda.max_memory_reserved() / 2**20, 1),
            "checkpoint_sha256": {
                str(s): sha256_file(ARTIFACTS / f"step4_{name}_{s}.pt")
                for s in ({*CKPT_AT, steps} if steps in CKPT_AT or steps < min(CKPT_AT)
                          else {*CKPT_AT})
                if (ARTIFACTS / f"step4_{name}_{s}.pt").exists()},
            "monitor": monitor}
    del replay
    return world, info


def resume_valid(name, backend, hidden, seed, shuffled, ckpt_step) -> bool:
    """Strict resume: report presence alone is NOT trusted."""
    path = ARTIFACTS / f"step4_{name}_{ckpt_step}.pt"
    if not path.exists():
        return False
    prov = torch.load(path, weights_only=False).get("provenance", {})
    arm = prov.get("arm", {})
    return (prov.get("source_digest") == source_digest()
            and prov.get("encoder_sha256") == sha256_file(ENCODER_CKPT)
            and prov.get("replay_file_sha256") == sha256_file(TRAIN_40K_CACHE)
            and arm.get("backend") == backend
            and arm.get("global_hidden") == hidden
            and arm.get("seed") == seed
            and arm.get("shuffled") == shuffled
            and arm.get("steps") == ckpt_step)


def load_arm_for_eval(name, backend, hidden, seed, shuffled, ckpt_step, device):
    ckpt = torch.load(ARTIFACTS / f"step4_{name}_{ckpt_step}.pt", weights_only=False)
    prov = ckpt["provenance"]
    assert prov["source_digest"] == source_digest(), f"{name}: stale source"
    arm = prov["arm"]
    assert (arm["backend"], arm["global_hidden"], arm["seed"], arm["shuffled"],
            arm["steps"]) == (backend, hidden, seed, shuffled, ckpt_step), \
        f"{name}: checkpoint arm config mismatch {arm}"
    world = build_world(backend, hidden, device)
    current = world.state_dict()
    saved = ckpt["state_dict"]
    assert set(current) == set(saved), (
        f"{name}: state_dict keys differ "
        f"(missing {set(current) - set(saved)}, extra {set(saved) - set(current)})")
    for key in current:
        assert current[key].shape == saved[key].shape, f"{name}: shape {key}"
    world.load_state_dict(saved, strict=True)
    world.eval()   # CHECK-4
    return world


# --------------------------------------------------------------------------
# strata + statistics
# --------------------------------------------------------------------------

def anchor_strata(anchors) -> list[dict]:
    """Pre-registered strata. PRIMARY action-effective notion = pixel-effective
    (any alternative suffix changes any final-frame pixel vs true, any branch);
    secondary = task/outcome-effective (reward/termination/task signature)."""
    strata = []
    for a in anchors:
        true_frames = a["branches"]["true"]["frames"][:, -1]
        true_outcomes = a["branches"]["true"]["outcomes"]
        pixel = False
        task = False
        for suffix, branch in a["branches"].items():
            if suffix == "true":
                continue
            if not np.array_equal(branch["frames"][:, -1], true_frames):
                pixel = True
            if branch["outcomes"] != true_outcomes:
                task = True
        strata.append({"night": bool(a["night"]),
                       "pixel_effective": pixel, "task_effective": task})
    return strata


def attach_strata(rows, strata):
    for row in rows:
        row.update(strata[row["anchor"]])
    return rows


def gate_decisions_per_seed(rows_by_seed: dict, control_rows_by_seed: dict) -> dict:
    """REGISTERED family rule: G-a..G-d decided per training seed (env-seed
    clustering preserved within the model; G-c against the SAME-SEED shuffled
    control), then 2/3 majority."""
    per_seed = {}
    for seed, rows in rows_by_seed.items():
        s_all = seed_level_summary(rows, "retrieval_all")
        s_sep = seed_level_summary(rows, "separation_all")
        s_chg = seed_level_summary(rows, "retrieval_changed")
        control = seed_level_summary(control_rows_by_seed[seed], "retrieval_all")
        per_seed[str(seed)] = {
            "retrieval_all": s_all, "separation_all": s_sep,
            "retrieval_changed": s_chg, "control_retrieval_all": control,
            "G_a": bool(s_all["mean"] >= 0.27 and s_all["ci95"][0] > 0.255),
            "G_b": bool(s_sep["ci95"][0] > 0),
            "G_c": bool(s_all["mean"] >= control["mean"] + 0.015),
            "G_d": bool(s_chg["ci95"][0] > 0.255),
        }
    majority = {gate: sum(per_seed[str(s)][gate] for s in rows_by_seed) >= 2
                for gate in ("G_a", "G_b", "G_c", "G_d")}
    majority["all_gates"] = all(majority.values())
    return {"per_seed": per_seed, "majority": majority}


def cluster_bootstrap_ci(diffs_by_env: dict, iters=10_000, seed=0):
    rng = np.random.default_rng(seed)
    clusters = [np.asarray(v, dtype=np.float64) for v in diffs_by_env.values()]
    stats = np.empty(iters)
    for i in range(iters):
        picked = rng.integers(len(clusters), size=len(clusters))
        stats[i] = float(np.mean(np.concatenate([clusters[j] for j in picked])))
    point = float(np.mean(np.concatenate(clusters)))
    return point, [float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))]


def two_level_bootstrap_ci(per_seed_diffs: dict, iters=10_000, seed=0):
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


def backend_verdict(rows_by_arm, left="M1_gru64", right="M2_mamba2",
                    metric="retrieval_all", row_filter=None):
    """Paired per-anchor right-minus-left difference; win requires positive
    pooled two-level LB AND all three per-seed differences agreeing in sign."""
    per_seed = {}
    per_seed_diffs = {}
    for seed in TRAIN_SEEDS:
        rows_l = rows_by_arm[f"{left}_s{seed}"]
        rows_r = rows_by_arm[f"{right}_s{seed}"]
        assert len(rows_l) == len(rows_r)
        diffs_by_env: dict = {}
        for rl, rr in zip(rows_l, rows_r):
            assert rl["anchor"] == rr["anchor"] and rl["env_seed"] == rr["env_seed"]
            if row_filter is not None and not row_filter(rl):
                continue
            if rl[metric] is None or rr[metric] is None:
                continue
            diffs_by_env.setdefault(rl["env_seed"], []).append(rr[metric] - rl[metric])
        point, ci = cluster_bootstrap_ci(diffs_by_env, seed=seed)
        per_seed[str(seed)] = {"diff": point, "ci95_env_clustered": ci,
                               "n_anchors": int(sum(len(v) for v in diffs_by_env.values()))}
        per_seed_diffs[seed] = diffs_by_env
    pooled_point, pooled_ci = two_level_bootstrap_ci(per_seed_diffs)
    seed_means = np.array([per_seed[str(s)]["diff"] for s in TRAIN_SEEDS])
    t_mean = float(seed_means.mean())
    t_se = float(seed_means.std(ddof=1) / np.sqrt(len(seed_means)))
    verdict = (f"{right}_wins" if (pooled_ci[0] > 0 and (seed_means > 0).all())
               else f"{left}_wins" if (pooled_ci[1] < 0 and (seed_means < 0).all())
               else "parity")
    return {"comparison": f"{right} - {left}", "metric": metric,
            "per_training_seed": per_seed,
            "pooled_two_level": {"diff": pooled_point, "ci95": pooled_ci},
            "seed_level_t": {"mean": t_mean, "se": t_se, "n": len(seed_means),
                             "ci95": [t_mean - 4.303 * t_se, t_mean + 4.303 * t_se]},
            "verdict": verdict}


# --------------------------------------------------------------------------
# engineering figures (parity tie-breaker evidence)
# --------------------------------------------------------------------------

def engineering_figures(backend, hidden, device, warm=5, iters=50):
    cfg = ModelConfig(temporal_backend=backend, predictor="deterministic",
                      mask_ratio=0.0, rollout_steps=2, global_hidden=hidden)
    from model import M3HJWM
    torch.manual_seed(0)
    world = M3HJWM(cfg).to(device).eval()
    core = world.temporal.impl
    streams, dim = world.streams, cfg.token_dim
    temporal_params = sum(p.numel() for n, p in world.named_parameters()
                          if n.startswith("temporal."))
    with torch.no_grad():
        state = core.init_state(48, streams, device, torch.float32)
        cache_mib = 0.0
        if state.cache is not None:
            flat = []
            for entry in state.cache:
                flat.extend(entry if isinstance(entry, (tuple, list)) else [entry])
            cache_mib = sum(t.numel() * t.element_size() for t in flat) / 2**20
        x_step = torch.randn(48, streams, dim, device=device)
        for _ in range(warm):
            _, state = core.step(x_step, state)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            _, state = core.step(x_step, state)
        torch.cuda.synchronize()
        step_ms = (time.perf_counter() - t0) / iters * 1e3
        x_seq = torch.randn(4, 16, streams, dim, device=device)
        for _ in range(warm):
            core.sequence(x_seq)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            core.sequence(x_seq)
        torch.cuda.synchronize()
        seq_ms = (time.perf_counter() - t0) / iters * 1e3
    del world
    torch.cuda.empty_cache()
    return {"backend": backend, "global_hidden": hidden,
            "temporal_params": int(temporal_params),
            "cache_mib_B48": round(cache_mib, 4),
            "warm_step_ms_B48": round(step_ms, 4),
            "warm_seq_ms_B4T16": round(seq_ms, 4)}


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------

def main(smoke=False, gru72=False):
    device = torch.device("cuda")
    steps = 30 if smoke else STEPS_TOTAL
    ckpt_step = steps
    dirty = tracked_dirty()
    if dirty and not smoke:
        raise RuntimeError(
            "tracked files are dirty; commit before a real step-4 run:\n"
            + "\n".join(dirty))
    train, _ = load_scaled_data()
    monitor_anchors = torch.load(BUNDLE, weights_only=False)
    pretrainer = IJEPAPretrainer(
        ModelConfig(temporal_backend="gru", predictor="deterministic", mask_ratio=0.0))
    pretrainer.load_state_dict(
        torch.load(ENCODER_CKPT, weights_only=False)["pretrainer"], strict=True)
    encoder = pretrainer.target_encoder.to(device).eval()

    specs = ARM_SPECS + (GRU72_SPECS if gru72 else ())
    arm_list = [(tmpl.format(seed=seed), backend, hidden, seed, shuffled)
                for seed in TRAIN_SEEDS
                for tmpl, backend, hidden, shuffled in specs]

    report_path = ARTIFACTS / ("step4_smoke_report.json" if smoke else "step4_report.json")
    report = (json.loads(report_path.read_text()) if report_path.exists()
              else {"protocol": "reviews/2026-07-14-step4-protocol.md (+amendments)",
                    "smoke": smoke, "arms": {}})
    report["head_commit"] = git_head()
    report["source_digest"] = source_digest()
    report["tracked_dirty"] = dirty
    report["versions"] = software_versions()
    report["hashes"] = {"encoder": sha256_file(ENCODER_CKPT),
                        "replay_file": sha256_file(TRAIN_40K_CACHE),
                        "monitor_bundle": sha256_file(BUNDLE)}
    report["strata_registration"] = {
        "primary_action_effective": "pixel_effective",
        "secondary_action_effective": "task_effective"}

    # ---------------- training phase (no final-bundle access) ----------------
    for seed in TRAIN_SEEDS:
        torch.manual_seed(seed)
        reference = build_world("global_gru", 64, device)
        reference_shared = {n: t.detach().cpu().clone()
                            for n, t in reference.state_dict().items()
                            if not n.startswith("temporal.")}
        del reference
        torch.cuda.empty_cache()
        for name, backend, hidden, arm_seed, shuffled in arm_list:
            if arm_seed != seed:
                continue
            if (name in report["arms"]
                    and resume_valid(name, backend, hidden, seed, shuffled, ckpt_step)):
                print(f"[{name}] resume-valid checkpoint found, skipping", flush=True)
                continue
            report["arms"].pop(name, None)
            world, info = train_arm(name, backend, hidden, seed, shuffled, train,
                                    reference_shared, monitor_anchors, encoder,
                                    device, steps)
            report["arms"][name] = info
            report_path.write_text(json.dumps(report, indent=2, default=str))
            del world
            torch.cuda.empty_cache()
        shared = {report["arms"][n]["shared_state_digest"]
                  for n in report["arms"] if n.endswith(f"_s{seed}")}
        replays = {report["arms"][n]["replay_stream_digest"]
                   for n in report["arms"] if n.endswith(f"_s{seed}")}
        assert len(shared) == 1, f"seed {seed}: shared-state digests diverge"
        assert len(replays) == 1, f"seed {seed}: replay streams diverge"

    # ---------------- blind evaluation phase ----------------
    missing = [name for name, *_ in arm_list
               if not (ARTIFACTS / f"step4_{name}_{ckpt_step}.pt").exists()]
    assert not missing, f"CHECK-3 blindness: arms incomplete {missing}"
    if smoke:
        final_anchors = monitor_anchors
        report["eval_bundle"] = "monitor_21_24"
    else:
        manifest = json.loads(FINAL_MANIFEST.read_text())
        actual = sha256_file(FINAL_BUNDLE)
        assert actual == manifest["sha256"], "final bundle hash != manifest"
        report["hashes"]["final_bundle"] = actual
        final_anchors = torch.load(FINAL_BUNDLE, weights_only=False)
    strata = anchor_strata(final_anchors)
    report["strata_counts"] = {
        "night": int(sum(s["night"] for s in strata)),
        "pixel_effective": int(sum(s["pixel_effective"] for s in strata)),
        "task_effective": int(sum(s["task_effective"] for s in strata)),
        "total": len(strata)}

    rows_by_arm = {}
    suffix = "_smoke" if smoke else ""
    for name, backend, hidden, seed, shuffled in arm_list:
        world = load_arm_for_eval(name, backend, hidden, seed, shuffled,
                                  ckpt_step, device)
        rows = attach_strata(symmetric_eval(world, encoder, final_anchors, device),
                             strata)
        rows_by_arm[name] = rows
        (ARTIFACTS / f"step4_rows_{name}{suffix}.json").write_text(json.dumps(rows))
        del world
        torch.cuda.empty_cache()

    report["family_gates"] = {}
    for family, backend_tag in (("M1_gru64", "gru64"), ("M2_mamba2", "mamba2")):
        report["family_gates"][family] = gate_decisions_per_seed(
            {s: rows_by_arm[f"{family}_s{s}"] for s in TRAIN_SEEDS},
            {s: rows_by_arm[f"M3_{backend_tag}_shuf_s{s}"] for s in TRAIN_SEEDS})

    report["backend_verdict"] = backend_verdict(rows_by_arm)
    report["backend_verdict_changed"] = backend_verdict(
        rows_by_arm, metric="retrieval_changed")
    report["backend_verdict_strata"] = {
        "day": backend_verdict(rows_by_arm, row_filter=lambda r: not r["night"]),
        "night": backend_verdict(rows_by_arm, row_filter=lambda r: r["night"]),
        "pixel_effective": backend_verdict(
            rows_by_arm, row_filter=lambda r: r["pixel_effective"]),
        "task_effective": backend_verdict(
            rows_by_arm, row_filter=lambda r: r["task_effective"])}
    if gru72:
        report["gru72_vs_mamba"] = backend_verdict(
            rows_by_arm, left="M4_gru72", right="M2_mamba2")
    report["engineering"] = {
        "gru64": engineering_figures("global_gru", 64, device),
        "mamba2": engineering_figures("global_mamba2", 64, device),
        **({"gru72": engineering_figures("global_gru", 72, device)} if gru72 else {})}
    for name in list(report["arms"]):
        path = ARTIFACTS / f"step4_{name}_{ckpt_step}.pt"
        if path.exists():
            report["arms"][name].setdefault("checkpoint_sha256", {})[str(ckpt_step)] = \
                sha256_file(path)
    report_path.write_text(json.dumps(report, indent=2, default=str))
    print(json.dumps({"verdict": report["backend_verdict"]["verdict"],
                      "pooled": report["backend_verdict"]["pooled_two_level"],
                      "gates": {f: report["family_gates"][f]["majority"]
                                for f in report["family_gates"]}}, indent=2))
    if report["backend_verdict"]["verdict"] == "M2_mamba2_wins" and not gru72:
        print("Mamba family won: rerun with --gru72 for the pre-registered "
              "capacity control before any backend attribution.")


if __name__ == "__main__":
    main(smoke="--smoke" in sys.argv, gru72="--gru72" in sys.argv)
