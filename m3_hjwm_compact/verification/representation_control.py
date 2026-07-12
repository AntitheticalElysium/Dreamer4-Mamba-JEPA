"""Small real-Crafter representation control (Phase B).

This trains no temporal model, reward head, or policy. Dense-token and pooled-state
predictors use identical encoders, data, action conditioning, and optimizer budgets.
Held-out controls include shuffled targets/actions and a copy-latent baseline.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import Tensor, nn
import torch.nn.functional as F


COMPACT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(COMPACT_ROOT))

from model import (  # noqa: E402
    EMARepresentationEncoder,
    FuturePredictor,
    ModelConfig,
    RepresentationEncoder,
    cosine_distance,
    effective_rank,
    multi_block_mask,
)


INVENTORY_KEYS = (
    "health",
    "food",
    "drink",
    "energy",
    "sapling",
    "wood",
    "stone",
    "coal",
    "iron",
    "diamond",
)


@dataclass
class Transitions:
    obs: np.ndarray
    actions: np.ndarray
    next_obs: np.ndarray
    semantic: np.ndarray
    inventory: np.ndarray
    player_pos: np.ndarray

    def __len__(self):
        return len(self.actions)


def chw(obs: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(obs.transpose(2, 0, 1))


def collect(seed: int, count: int) -> Transitions:
    import crafter

    env = crafter.Env(seed=seed, length=max(10_000, count + 10))
    rng = np.random.default_rng(seed)

    def labelled_reset():
        observation = env.reset()
        observation, _, done, info = env.step(int(rng.integers(env.action_space.n)))
        if done:
            return labelled_reset()
        return observation, info

    current_obs, current_info = labelled_reset()
    observations, actions, next_observations, semantics, inventories, positions = [], [], [], [], [], []
    while len(actions) < count:
        action = int(rng.integers(env.action_space.n))
        next_obs, _, done, next_info = env.step(action)
        observations.append(chw(current_obs))
        actions.append(action)
        next_observations.append(chw(next_obs))
        semantics.append(np.asarray(current_info["semantic"], dtype=np.uint8))
        inventory = current_info["inventory"]
        inventories.append([float(inventory.get(key, 0.0)) for key in INVENTORY_KEYS])
        positions.append(np.asarray(current_info["player_pos"], dtype=np.int64))
        if done:
            current_obs, current_info = labelled_reset()
        else:
            current_obs, current_info = next_obs, next_info
    return Transitions(
        obs=np.stack(observations),
        actions=np.asarray(actions, dtype=np.int64),
        next_obs=np.stack(next_observations),
        semantic=np.stack(semantics),
        inventory=np.asarray(inventories, dtype=np.float32),
        player_pos=np.stack(positions),
    )


def concatenate(parts: list[Transitions]) -> Transitions:
    return Transitions(
        **{
            name: np.concatenate([getattr(part, name) for part in parts], axis=0)
            for name in Transitions.__dataclass_fields__
        }
    )


class RepresentationControl(nn.Module):
    def __init__(self, cfg: ModelConfig, variant: str):
        super().__init__()
        self.cfg = cfg
        self.variant = variant
        self.online = RepresentationEncoder(cfg)
        self.target = EMARepresentationEncoder(self.online, cfg.ema_decay)
        self.predictor = FuturePredictor(cfg)

    def project(self, tokens: Tensor) -> Tensor:
        if self.variant == "dense":
            return tokens
        if self.cfg.registers:
            return tokens[:, : self.cfg.registers].mean(1, keepdim=True)
        return tokens.mean(1, keepdim=True)

    def forward(self, obs: Tensor, action: Tensor, next_obs: Tensor):
        grid = self.cfg.image_size // self.cfg.patch_size
        mask = multi_block_mask(
            obs.shape[0],
            grid,
            self.cfg.mask_ratio,
            self.cfg.target_blocks,
            obs.device,
        )
        context = self.project(self.online(obs, mask))
        with torch.no_grad():
            target = self.project(self.target(next_obs))
        horizon = torch.ones(obs.shape[0], dtype=torch.long, device=obs.device)
        return self.predictor(context, action, horizon, target), target


def minibatch(data: Transitions, indices: np.ndarray, device):
    return (
        torch.from_numpy(data.obs[indices]).to(device),
        torch.from_numpy(data.actions[indices]).to(device),
        torch.from_numpy(data.next_obs[indices]).to(device),
    )


@torch.no_grad()
def encode_all(encoder, array: np.ndarray, batch: int, device) -> Tensor:
    encoded = []
    for start in range(0, len(array), batch):
        obs = torch.from_numpy(array[start : start + batch]).to(device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            encoded.append(encoder(obs).float().cpu())
    return torch.cat(encoded)


@torch.no_grad()
def prediction_controls(model, data: Transitions, batch: int, device):
    predictions, targets, currents = [], [], []
    for start in range(0, len(data), batch):
        stop = min(start + batch, len(data))
        obs = torch.from_numpy(data.obs[start:stop]).to(device)
        next_obs = torch.from_numpy(data.next_obs[start:stop]).to(device)
        action = torch.from_numpy(data.actions[start:stop]).to(device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            context_tokens = model.online(obs)
            current_target = model.project(model.target(obs))
            next_target = model.project(model.target(next_obs))
            context = model.project(context_tokens)
            horizon = torch.ones(stop - start, dtype=torch.long, device=device)
            modes, _ = model.predictor.all_predictions(context, action, horizon)
            predictions.append(modes[:, 0].float().cpu())
            targets.append(next_target.float().cpu())
            currents.append(current_target.float().cpu())
    prediction = torch.cat(predictions)
    target = torch.cat(targets)
    current = torch.cat(currents)
    shuffled_target = target.roll(1, 0)

    # Copy-latent is a near-unbeatable bar on the *mean over all tokens* because
    # most Crafter tokens do not change between frames (untrained-encoder copy
    # cosine is already ~0.02). Score the tokens that actually moved, and report
    # the unrelated-pair distance as the scale ruler.
    per_token_pred = cosine_distance(prediction, target)
    per_token_copy = cosine_distance(current, target)
    ruler = float(cosine_distance(current, target.roll(7, 0)).mean())
    if per_token_copy.dim() > 1:  # dense variant: [N, S]
        motion = per_token_copy
        changed = motion >= motion.flatten().quantile(0.75)
        pred_changed = float(per_token_pred[changed].mean())
        copy_changed = float(per_token_copy[changed].mean())
    else:  # pooled variant: single stream
        pred_changed = float(per_token_pred.mean())
        copy_changed = float(per_token_copy.mean())

    shuffled_predictions = []
    shuffled_actions = np.roll(data.actions, 1)
    for start in range(0, len(data), batch):
        stop = min(start + batch, len(data))
        obs = torch.from_numpy(data.obs[start:stop]).to(device)
        action = torch.from_numpy(shuffled_actions[start:stop].copy()).to(device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            context = model.project(model.online(obs))
            horizon = torch.ones(stop - start, dtype=torch.long, device=device)
            modes, _ = model.predictor.all_predictions(context, action, horizon)
            shuffled_predictions.append(modes[:, 0].float().cpu())
    shuffled_prediction = torch.cat(shuffled_predictions)
    return {
        "heldout_one_step_cosine": float(per_token_pred.mean()),
        "shuffled_target_cosine": float(cosine_distance(prediction, shuffled_target).mean()),
        "shuffled_action_cosine": float(cosine_distance(shuffled_prediction, target).mean()),
        "copy_latent_cosine": float(per_token_copy.mean()),
        "unrelated_pair_cosine": ruler,
        "pred_cosine_changed_tokens": pred_changed,
        "copy_cosine_changed_tokens": copy_changed,
        "improvement_over_copy_changed_tokens": copy_changed - pred_changed,
    }


def target_statistics(tokens: Tensor):
    flat = tokens.reshape(-1, tokens.shape[-1]).float()
    centered = flat - flat.mean(0, keepdim=True)
    covariance = centered.T @ centered / max(1, len(centered) - 1)
    eigenvalues = torch.linalg.eigvalsh(covariance).clamp_min(0).flip(0)
    normalized = eigenvalues / eigenvalues.sum().clamp_min(1e-12)
    covariance_rank = torch.exp(
        -(normalized * normalized.clamp_min(1e-12).log()).sum()
    )
    return {
        "target_element_variance": float(flat.var(0, unbiased=False).mean()),
        "target_effective_rank_singular": float(effective_rank(tokens)),
        "target_effective_rank_covariance": float(covariance_rank),
        "covariance_eigenvalues_desc": [float(value) for value in eigenvalues],
    }


def local_view_labels(semantic: np.ndarray, player_pos: np.ndarray, grid: int):
    """Per-token semantic labels aligned with Crafter's *rendered local view*.

    `info["semantic"]` is the GLOBAL world map indexed [world_x, world_y]
    (crafter SemanticView returns world._mat_map with objects overlaid); the
    observation is the local egocentric view. Geometry from crafter env.render()
    and engine.LocalView at the pinned commit:
      - final image rows = world y, cols = world x (canvas is transposed);
      - 9x7 world tiles of `unit = 64 // 9 = 7` px; image rows >= 49 are the
        inventory HUD, row/col 63 is border padding;
      - tile (tx, ty) shows world cell player_pos + (tx - 4, ty - 3).
    Each token is labelled by the tile under its centre pixel; tokens whose
    centre falls in the HUD or outside the world are masked invalid.
    """
    n = len(semantic)
    area_x, area_y = semantic.shape[1], semantic.shape[2]
    unit = 64 // 9
    world_rows = 7 * unit  # image rows showing world tiles
    patch = 64 // grid
    labels = np.zeros((n, grid, grid), dtype=np.int64)
    valid = np.zeros((n, grid, grid), dtype=bool)
    for i in range(grid):        # token row -> image row -> world y offset
        centre_row = i * patch + patch // 2
        if centre_row >= world_rows:
            continue
        ty = centre_row // unit
        for j in range(grid):    # token col -> image col -> world x offset
            centre_col = j * patch + patch // 2
            tx = centre_col // unit
            if tx >= 9:
                continue
            wx = player_pos[:, 0] + (tx - 4)
            wy = player_pos[:, 1] + (ty - 3)
            inside = (wx >= 0) & (wx < area_x) & (wy >= 0) & (wy < area_y)
            labels[inside, i, j] = semantic[np.arange(n)[inside], wx[inside], wy[inside]]
            valid[:, i, j] = inside
    return labels, valid


def semantic_probe(
    tokens: Tensor, semantic: np.ndarray, player_pos: np.ndarray,
    registers: int, grid: int, device,
):
    labels_np, valid_np = local_view_labels(semantic, player_pos, grid)
    local = tokens[:, registers:].reshape(len(tokens), grid, grid, -1)
    labels = torch.from_numpy(labels_np)
    valid = torch.from_numpy(valid_np)

    # Random frame-level split (tokens of one frame never straddle the split).
    rng = np.random.default_rng(91)
    order = rng.permutation(len(tokens))
    half = len(tokens) // 2
    train_f, test_f = order[:half], order[half:]

    def gather(frames):
        x = local[frames][valid[frames]]
        y = labels[frames][valid[frames]]
        return x.to(device), y.to(device)

    train_x, train_y = gather(train_f)
    test_x, test_y = gather(test_f)
    classes = int(labels[valid].max()) + 1
    torch.manual_seed(91)
    probe = nn.Linear(tokens.shape[-1], classes).to(device)
    optimizer = torch.optim.AdamW(probe.parameters(), lr=1e-2, weight_decay=1e-3)
    for _ in range(500):
        loss = F.cross_entropy(probe(train_x), train_y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        train_accuracy = (probe(train_x).argmax(-1) == train_y).float().mean()
        accuracy = (probe(test_x).argmax(-1) == test_y).float().mean()
        majority = torch.bincount(train_y, minlength=classes).argmax()
        baseline = (test_y == majority).float().mean()
    result = {
        "semantic_token_accuracy": float(accuracy),
        "semantic_train_accuracy": float(train_accuracy),
        "semantic_majority_accuracy": float(baseline),
        "semantic_classes": classes,
        "semantic_valid_fraction": float(valid.float().mean()),
    }
    # A converged linear probe can never sit below majority; if it does, the
    # labels are misaligned with the features and the probe result is invalid.
    result["semantic_probe_sane"] = bool(train_accuracy >= baseline - 1e-3)
    return result


def inventory_probe(tokens: Tensor, inventory: np.ndarray, registers: int):
    features = (
        tokens[:, :registers].mean(1) if registers else tokens.mean(1)
    ).double()
    labels = torch.from_numpy(inventory).double()
    split = len(features) // 2
    train_x = torch.cat([features[:split], torch.ones(split, 1)], dim=1)
    test_x = torch.cat(
        [features[split:], torch.ones(len(features) - split, 1)], dim=1
    )
    train_y, test_y = labels[:split], labels[split:]
    ridge = 1e-2 * torch.eye(train_x.shape[1], dtype=train_x.dtype)
    weights = torch.linalg.solve(train_x.T @ train_x + ridge, train_x.T @ train_y)
    prediction = test_x @ weights
    denominator = ((test_y - train_y.mean(0)) ** 2).sum(0)
    r2 = 1.0 - ((test_y - prediction) ** 2).sum(0) / denominator.clamp_min(1e-12)
    valid = denominator > 1e-8
    return {
        "inventory_r2_mean_varying": float(r2[valid].mean()) if bool(valid.any()) else None,
        "inventory_r2": {
            key: (float(value) if bool(is_valid) else None)
            for key, value, is_valid in zip(INVENTORY_KEYS, r2, valid)
        },
    }


def train_variant(
    variant: str,
    cfg: ModelConfig,
    train_data: Transitions,
    heldout: Transitions,
    steps: int,
    batch_size: int,
    device,
):
    torch.manual_seed(101)
    rng = np.random.default_rng(101)
    model = RepresentationControl(cfg, variant).to(device)
    parameters = list(model.online.parameters()) + list(model.predictor.parameters())
    optimizer = torch.optim.AdamW(parameters, lr=3e-4, weight_decay=1e-4)
    losses = []
    peak_mib = 0.0
    started = time.perf_counter()
    for step in range(steps):
        indices = rng.integers(0, len(train_data), size=batch_size)
        obs, action, next_obs = minibatch(train_data, indices, device)
        if step == 1:
            torch.cuda.reset_peak_memory_stats()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            prediction, _ = model(obs, action, next_obs)
            loss = prediction.regression
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters, 10.0)
        optimizer.step()
        model.target.update(model.online)
        losses.append(float(loss.detach()))
        if step >= 1:
            peak_mib = max(peak_mib, torch.cuda.max_memory_allocated() / 2**20)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started

    target_tokens = encode_all(model.target, heldout.obs, batch_size, device)
    controls = prediction_controls(model, heldout, batch_size, device)
    grid = cfg.image_size // cfg.patch_size
    result = {
        "variant": variant,
        "steps": steps,
        "transitions": len(train_data),
        "heldout_transitions": len(heldout),
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "initial_loss_mean_20": float(np.mean(losses[:20])) if losses else None,
        "final_loss_mean_20": float(np.mean(losses[-20:])) if losses else None,
        "elapsed_seconds": elapsed,
        "peak_allocated_mib": peak_mib,
        **controls,
        **target_statistics(target_tokens),
        **semantic_probe(
            target_tokens, heldout.semantic, heldout.player_pos, cfg.registers, grid, device
        ),
        **inventory_probe(target_tokens, heldout.inventory, cfg.registers),
    }
    del model, optimizer
    torch.cuda.empty_cache()
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", nargs="+", choices=["dense", "pooled"], default=["dense", "pooled"])
    parser.add_argument("--train-per-seed", type=int, default=500)
    parser.add_argument("--heldout", type=int, default=500)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--batch", type=int, default=16)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the target-hardware control")
    device = torch.device("cuda")

    collection_start = time.perf_counter()
    train_data = concatenate(
        [collect(seed, args.train_per_seed) for seed in (0, 1)]
    )
    heldout = collect(2, args.heldout)
    collection_seconds = time.perf_counter() - collection_start
    cfg = ModelConfig(
        patch_size=8,
        token_dim=64,
        registers=2,
        spatial_depth=1,
        spatial_heads=4,
        temporal_backend="gru",
        predictor="deterministic",
        predictor_depth=2,
        mask_ratio=0.6,
    )
    # Untrained-encoder baseline first: every probe number must be read relative
    # to this row, and a converged probe below majority marks the probe invalid.
    results = [
        train_variant("dense", cfg, train_data, heldout, 0, args.batch, device)
    ]
    results[0]["variant"] = "untrained_baseline"
    results += [
        train_variant(
            variant,
            cfg,
            train_data,
            heldout,
            args.steps,
            args.batch,
            device,
        )
        for variant in args.variant
    ]
    print(
        json.dumps(
            {
                "protocol": {
                    "train_seeds": [0, 1],
                    "heldout_seed": 2,
                    "random_policy": True,
                    "collection_seconds": collection_seconds,
                    "mask_ratio": cfg.mask_ratio,
                    "token_dim": cfg.token_dim,
                    "grid": [8, 8],
                    "registers": cfg.registers,
                },
                "results": results,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
