"""Matrix step 1: faithful same-frame I-JEPA vs incumbent hybrid vs untrained.

Protocol and pre-registered gates G1-G5: reviews/2026-07-13-step1-protocol.md
(committed before the first run). Artifacts: reviews/artifacts/ssl_step1_*.json.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

COMPACT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(COMPACT_ROOT))
sys.path.insert(0, str(COMPACT_ROOT / "verification"))

from data import Episode, EpisodeReplay  # noqa: E402
from model import LossConfig, M3HJWM, ModelConfig, online_hybrid_recipe  # noqa: E402
from ssl_ijepa import IJEPAPretrainer  # noqa: E402
from representation_control import (  # noqa: E402
    collect, inventory_probe, semantic_probe, target_statistics,
)

ARTIFACTS = Path(__file__).resolve().parents[2] / "reviews" / "artifacts"
# Durable, gitignored cache; scratchpad/tmp caches proved non-reproducible
# (2026-07-13 re-audit, moderate finding 13).
DATA_CACHE = Path(__file__).resolve().parents[2] / "data" / "shared_random_policy_v1.pt"
PROBE_V3_CACHE = Path(__file__).resolve().parents[2] / "data" / "probe_5seed_v3.pt"
PROBE_SEEDS = (2, 5, 6, 7, 8)
# Relative abort (protocol amendment 1e-abort): thresholds must survive
# architecture changes that shift metric scales.
def abort_threshold(untrained_obs_frac: float) -> float:
    return max(0.6 * untrained_obs_frac, untrained_obs_frac - 0.10)


def collect_episodes(seed: int, episodes: int, max_len: int = 200):
    import crafter

    def to_chw(image):
        return np.ascontiguousarray(image.transpose(2, 0, 1))

    env = crafter.Env(seed=seed, length=max_len)
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(episodes):
        obs = env.reset()
        frames, actions, rewards, continues = [to_chw(obs)], [], [], []
        done = False
        while not done:
            action = int(rng.integers(env.action_space.n))
            obs, reward, done, info = env.step(action)
            frames.append(to_chw(obs))
            actions.append(action)
            rewards.append(float(reward))
            continues.append(float(info.get("discount", float(not done))))
        out.append(dict(
            obs=np.stack(frames).astype(np.uint8),
            actions=np.asarray(actions, dtype=np.int64),
            rewards=np.asarray(rewards, dtype=np.float32),
            continues=np.asarray(continues, dtype=np.float32),
        ))
    return out


def load_shared_data():
    if DATA_CACHE.exists():
        return torch.load(DATA_CACHE, weights_only=False)
    data = {
        "train_episodes": collect_episodes(0, 24) + collect_episodes(1, 24),
        "heldout_episodes": collect_episodes(2, 10),
        "heldout_probe": collect(2, 400),
    }
    DATA_CACHE.parent.mkdir(parents=True, exist_ok=True)
    torch.save(data, DATA_CACHE)
    return data


@torch.no_grad()
def encode(encoder, frames: np.ndarray, device, batch=64):
    out = []
    for start in range(0, len(frames), batch):
        obs = torch.from_numpy(frames[start:start + batch]).to(device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out.append(encoder(obs).float().cpu())
    return torch.cat(out)


def probes_block(target_encoder, probe, cfg, device):
    tokens = encode(target_encoder, probe.obs, device)
    grid = cfg.image_size // cfg.patch_size
    stats = target_statistics(tokens, cfg.registers)
    stats.pop("covariance_eigenvalues_desc", None)
    return {
        **stats,
        **semantic_probe(tokens, probe.semantic, probe.player_pos, cfg.registers, grid, device),
        **inventory_probe(tokens, probe.inventory, cfg.registers),
    }


def curve_point(target_encoder, probe, cfg, device, step, loss_mean, with_inventory=False):
    tokens = encode(target_encoder, probe.obs[:200], device)
    stats = target_statistics(tokens, cfg.registers)
    point = {
        "step": step,
        "loss_mean_100": loss_mean,
        "observation_variance_fraction": stats["target_observation_variance_fraction"],
        "stream_rank_mean": stats["target_stream_effective_rank_mean"],
    }
    if with_inventory:
        point["inventory_r2"] = inventory_probe(
            tokens, probe.inventory[:200], cfg.registers
        )["inventory_r2_mean_varying"]
    return point


def g4_seed_noninferiority(encode_fn, untrained_encoder, trained_encoder,
                           probe_by_seed, registers, resamples=1000):
    """Amendment 1g G4': paired degradation per independent probe seed (the
    probe's internal split is fixed and non-overlapping within each seed),
    bootstrap over seeds; PASS iff one-sided 90% UCB <= 0.02."""
    per_seed = {}
    for seed, part in probe_by_seed.items():
        tu = encode_fn(untrained_encoder, part.obs)
        tt = encode_fn(trained_encoder, part.obs)
        ru = inventory_probe(tu, part.inventory, registers)["inventory_r2_mean_varying"]
        rt = inventory_probe(tt, part.inventory, registers)["inventory_r2_mean_varying"]
        per_seed[seed] = float(ru - rt)
    values = np.array(list(per_seed.values()))
    rng = np.random.default_rng(404)
    boot = np.array([
        values[rng.integers(0, len(values), size=len(values))].mean()
        for _ in range(resamples)
    ])
    ucb = float(np.quantile(boot, 0.90))
    return {
        "G4p_per_seed_degradation": per_seed,
        "G4p_paired_degradation_mean": float(values.mean()),
        "G4p_ucb90": ucb,
        "G4p_pass": bool(ucb <= 0.02),
    }


def gates(final: dict, untrained: dict, losses: list[float]) -> dict:
    first = float(np.mean(losses[:100])) if losses else None
    last = float(np.mean(losses[-100:])) if losses else None
    return {
        "G1a_observation_variance_fraction": bool(
            final["target_observation_variance_fraction"]
            >= untrained["target_observation_variance_fraction"] - 0.05
        ),
        "G1b_same_stream_unrelated": bool(
            final["target_same_stream_unrelated_cosine"]
            >= 0.80 * untrained["target_same_stream_unrelated_cosine"]
        ),
        "G2a_stream_rank_mean": bool(
            final["target_stream_effective_rank_mean"]
            >= 0.90 * untrained["target_stream_effective_rank_mean"]
        ),
        "G2b_patch_pool_rank": bool(
            final["target_patch_pool_covariance_rank"]
            >= 0.90 * untrained["target_patch_pool_covariance_rank"]
        ),
        "G3_semantic": bool(
            final.get("semantic_probe_sane", False)
            and final["semantic_token_accuracy"]
            >= untrained["semantic_token_accuracy"] - 0.02
        ),
        "legacy_G4_point": [
            untrained.get("inventory_r2_mean_varying"),
            final.get("inventory_r2_mean_varying"),
        ],
        "loss_first_last_100": [first, last],
    }


def train_ijepa(cfg, frames, probe, steps, device, batch=64, sigreg_weight=0.0,
                untrained_obs_frac=0.0):
    torch.manual_seed(101)
    model = IJEPAPretrainer(cfg).to(device)
    model.sigreg_weight = sigreg_weight
    params = list(model.online_encoder.parameters()) + list(model.predictor.parameters())
    if sigreg_weight != 0.0:
        params += list(model.projector.parameters())
    optimizer = torch.optim.AdamW(params, lr=3e-4, weight_decay=1e-4)
    frame_rng = np.random.default_rng(2027)
    mask_generator = torch.Generator().manual_seed(2027)

    # Held-out pretext bank: fixed frames + fixed masks (amendment 1e / G5').
    from ssl_ijepa import sample_ijepa_masks
    bank_frames = torch.from_numpy(probe.obs[:128]).to(device)
    bank_gen = torch.Generator().manual_seed(31415)
    bank = [sample_ijepa_masks(128, cfg.image_size // cfg.patch_size, bank_gen)
            for _ in range(4)]
    fixed_directions = torch.randn(
        cfg.token_dim, 256, generator=torch.Generator().manual_seed(27182)
    ).to(device)
    fixed_directions = fixed_directions / fixed_directions.norm(dim=0).clamp_min(1e-12)

    @torch.no_grad()
    def bank_eval():
        vals = []
        projector_training = model.projector.training
        model.projector.eval()  # held-out diagnostics must not update BN state
        try:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                for ctx, preds_i in bank:
                    vals.append(float(
                        model.pretext_loss(bank_frames, ctx.to(device), preds_i)
                    ))
                dense = model.online_encoder(bank_frames)
                projected = model.projector(dense.mean(1)).float()[None]
                x_t = (projected @ fixed_directions).unsqueeze(-1) * model.sigreg.t
                err = (
                    (x_t.cos().mean(-3) - model.sigreg.phi).square()
                    + x_t.sin().mean(-3).square()
                )
                sig = float(
                    ((err @ model.sigreg.weights) * projected.size(-2)).mean()
                )
        finally:
            model.projector.train(projector_training)
        return float(np.mean(vals)), sig

    losses, pred_hist, sig_hist, curve, bank_curve, aborted = [], [], [], [], [], False
    started = time.perf_counter()
    for step in range(steps):
        idx = frame_rng.integers(0, len(frames), size=batch)
        obs = torch.from_numpy(frames[idx]).to(device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss, pred_component, sig_component = model.losses(obs, mask_generator)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 10.0)
        optimizer.step()
        model.update_target()
        losses.append(float(loss.detach()))
        pred_hist.append(float(pred_component))
        sig_hist.append(float(sig_component))
        if step % 50 == 0 or step == steps - 1:
            held_pred, held_sig = bank_eval()
            bank_curve.append({"step": step + 1, "heldout_pretext": held_pred,
                               "heldout_global_sigreg_fixed_directions": held_sig})
        if (step + 1) % 25 == 0:
            point = curve_point(
                model.target_encoder, probe, cfg, device, step + 1,
                float(np.mean(losses[-100:])), with_inventory=((step + 1) % 50 == 0),
            )
            curve.append(point)
            print(f"[ijepa] step {step+1} loss {point['loss_mean_100']:.4f} "
                  f"obs_frac {point['observation_variance_fraction']:.3f} "
                  f"stream_rank {point['stream_rank_mean']:.2f}", flush=True)
            if point["observation_variance_fraction"] < abort_threshold(untrained_obs_frac):
                aborted = True
                break
    minutes = round((time.perf_counter() - started) / 60, 2)
    return (model, losses, curve, aborted, minutes,
            {"pred_hist": pred_hist, "sig_hist": sig_hist, "bank_curve": bank_curve,
             "optimizer_state": optimizer.state_dict(),
             "mask_gen_state": mask_generator.get_state(),
             "rng_state": frame_rng.bit_generator.state})


def train_hybrid(cfg, episodes, steps, device):
    replay = EpisodeReplay()
    for ep in episodes:
        replay.add(Episode(**ep))
    torch.manual_seed(101)
    model = M3HJWM(cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    rng = np.random.default_rng(2027)
    losses = []
    for _ in range(steps):
        batch = replay.sample(batch=4, observations=16, device=device, rng=rng)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            # Online joint baseline: this path trains the encoder, so it uses
            # the explicit online recipe (2026-07-15 phase-recipe split).
            output = model(batch, online_hybrid_recipe())
        optimizer.zero_grad(set_to_none=True)
        output.loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        optimizer.step()
        model.mark_parameters_updated()
        model.update_target()
        losses.append(float(output.metrics["jepa"]))
    return model, losses


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--tag", default="step1")
    parser.add_argument("--arms", nargs="+", default=["ijepa", "hybrid"])
    args = parser.parse_args()
    device = torch.device("cuda")

    data = load_shared_data()
    frames = np.concatenate([ep["obs"] for ep in data["train_episodes"]])
    from representation_control import concatenate
    if PROBE_V3_CACHE.exists():
        probe_by_seed = torch.load(PROBE_V3_CACHE, weights_only=False)
    else:
        probe_by_seed = {s: collect(s, 400) for s in PROBE_SEEDS}
        torch.save(probe_by_seed, PROBE_V3_CACHE)
    probe = concatenate(list(probe_by_seed.values()))

    torch.manual_seed(101)
    cfg = ModelConfig(temporal_backend="gru", predictor="deterministic", mask_ratio=0.0)
    untrained_model = IJEPAPretrainer(cfg).to(device)
    untrained = probes_block(untrained_model.target_encoder, probe, cfg, device)
    untrained_encoder_ref = untrained_model.target_encoder
    torch.cuda.empty_cache()

    report = {
        "protocol": "reviews/2026-07-13-step1-protocol.md",
        "steps": args.steps,
        "train_frames": len(frames),
        "untrained": untrained,
        "arms": {},
    }
    for arm_name, weight in (("ijepa", 0.0), ("lejepa_global", 0.01)):
        if arm_name in args.arms:
            model, losses, curve, aborted, minutes, extras = train_ijepa(
                cfg, frames, probe, args.steps, device, sigreg_weight=weight,
                untrained_obs_frac=untrained["target_observation_variance_fraction"])
            pred_hist, sig_hist = extras["pred_hist"], extras["sig_hist"]
            bank_curve = extras["bank_curve"]
            optimizer_state = extras["optimizer_state"]
            rng_state = extras["rng_state"]
            mask_gen_state = extras["mask_gen_state"]
            final = probes_block(model.target_encoder, probe, cfg, device)
            arm_gates = gates(final, untrained, losses)
            encode_fn = lambda enc, arr: encode(enc, arr, device)
            arm_gates.update(g4_seed_noninferiority(
                encode_fn, untrained_encoder_ref, model.target_encoder,
                probe_by_seed, cfg.registers))
            arm_gates["G5p_bank_first_last"] = [
                bank_curve[0]["heldout_pretext"], bank_curve[-1]["heldout_pretext"]]
            arm_gates["G5p_pass"] = bool(
                bank_curve[-1]["heldout_pretext"]
                <= 0.70 * bank_curve[0]["heldout_pretext"]
            )
            report["arms"][arm_name] = {
                "minutes": minutes, "aborted": aborted, "curve": curve,
                "bank_curve": bank_curve,
                "component_first_last_100": {
                    "prediction": [
                        float(np.mean(pred_hist[:100])),
                        float(np.mean(pred_hist[-100:])),
                    ],
                    "sigreg": [
                        float(np.mean(sig_hist[:100])),
                        float(np.mean(sig_hist[-100:])),
                    ],
                },
                "sigreg_weight": weight,
                "final": final, "gates": arm_gates,
            }
            torch.save(
                {"pretrainer": model.state_dict(),
                 "optimizer": optimizer_state,
                 "config": {"sigreg_weight": weight, "steps": args.steps,
                            "sigreg_application": "global_projected_batch",
                            "lr": 3e-4, "weight_decay": 1e-4, "batch": 64},
                 "numpy_rng": rng_state,
                 "torch_rng_cpu": torch.get_rng_state(),
                 "torch_rng_cuda": torch.cuda.get_rng_state_all(),
                 "mask_generator_state": mask_gen_state,
                 "model_config": __import__("dataclasses").asdict(cfg),
                 "component_histories": {
                     "total": losses, "prediction": pred_hist, "sigreg": sig_hist,
                 },
                 "steps": args.steps},
                ARTIFACTS / f"ssl_step1_{arm_name}_{args.tag}.pt",
            )
            del model
            torch.cuda.empty_cache()
    if "hybrid" in args.arms:
        model, losses = train_hybrid(cfg, data["train_episodes"], args.steps, device)
        final = probes_block(model.target_encoder, probe, cfg, device)
        report["arms"]["hybrid"] = {
            "final": final, "gates": gates(final, untrained, losses),
        }
        del model
        torch.cuda.empty_cache()

    out = ARTIFACTS / f"ssl_step1_{args.tag}.json"
    out.write_text(json.dumps(report, indent=2))
    summary = {
        arm: {k: v for k, v in body["gates"].items()}
        for arm, body in report["arms"].items()
    }
    print(json.dumps(summary, indent=2))
    print(f"saved {out}")


if __name__ == "__main__":
    main()
