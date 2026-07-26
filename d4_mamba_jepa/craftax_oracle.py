"""Representation oracle (diagnostic Stage 2) for the Craftax world model.

Question, answered per target and without ambiguity: does the world encoder's
latent still contain the privileged simulator state a competent policy needs
(vitals, inventory, achievements), or has the representation discarded it?

Trust properties (why its verdicts can be believed):

- Every target is scored INDIVIDUALLY with an episode-level bootstrap CI -- a
  model can lose a rare diamond count while keeping a good mean inventory R^2,
  so no group average is a gate.
- The latent is compared against three references: a CONSTANT floor, a
  TIMESTEP-only control (vitals/inventory drift with episode age, so a latent
  must beat time), and a LINEAR RAW-PIXEL ceiling (no destructive pooling; the
  SVD ridge handles p >> n). A target is ``preserved`` only if the latent beats
  both floors AND is non-inferior to the pixel ceiling.
- A self-audit runs the same machinery on PERFECT, CONSTANT, MISALIGNED and
  TIMESTEP-SHIFTED (off-by-one) inputs; a trustworthy oracle scores them
  ~1 / ~0 / ~0 / low.
- Encoding preserves and restores every module's train/eval mode exactly
  (generalizes the D062 fix so a diagnostic can never corrupt BatchNorm).
- Probe data is isolated in both directions: ``.probe_only.pt`` payloads carry a
  marker the replay loader rejects, and privileged labels never enter training.

Continuous targets use R^2; binary achievements use tie-aware AUROC + average
precision + Brier (see ``oracle_metrics``). Self-contained numpy.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from .oracle_metrics import (
    auroc,
    average_precision,
    brier,
    episode_bootstrap_ci,
    r2_per_target,
    select_and_predict,
)

PROBE_ONLY_SUFFIX = ".probe_only.pt"
PROBE_ONLY_MARKER = "d4_mamba_jepa_craftax_probe_only_v1"

VITAL_NAMES = ["health", "food", "drink", "energy"]
INVENTORY_NAMES = [
    "wood", "stone", "coal", "iron", "diamond", "sapling",
    "wood_pickaxe", "stone_pickaxe", "iron_pickaxe",
    "wood_sword", "stone_sword", "iron_sword",
]


# ---------------------------------------------------------------------------
# Probe-data collection (imports craftax via craftax_env; run as a job).
# ---------------------------------------------------------------------------
@dataclass
class ProbeData:
    frames: np.ndarray        # uint8 [N, 3, H, W]
    vitals: np.ndarray        # float32 [N, 4]
    inventory: np.ndarray     # float32 [N, 12]
    achievements: np.ndarray  # bool [N, 22]
    episode_id: np.ndarray    # int [N]


def collect_probe_data(
    *, seeds, action_fn_factory, max_steps: int, target_size: int = 64
) -> ProbeData:
    """Roll episodes and pair every frame with its ground-truth labels."""
    from .craftax_env import CraftaxPixelEnv

    frames, vitals, inventory, achievements, episode_id = [], [], [], [], []
    for ep_idx, seed in enumerate(seeds):
        env = CraftaxPixelEnv(seed=int(seed), target_size=target_size)
        obs = env.reset()
        action_fn = action_fn_factory(int(seed))

        def record(frame):
            labels = env.privileged()
            frames.append(frame)
            vitals.append(labels["vitals"])
            inventory.append(labels["inventory"])
            achievements.append(labels["achievements"])
            episode_id.append(ep_idx)

        record(obs)
        for t in range(int(max_steps)):
            result = env.step(int(action_fn(obs, t)))
            obs = result.obs
            record(obs)
            if result.done:
                break
    return ProbeData(
        frames=np.stack(frames).astype(np.uint8),
        vitals=np.stack(vitals).astype(np.float32),
        inventory=np.stack(inventory).astype(np.float32),
        achievements=np.stack(achievements).astype(bool),
        episode_id=np.asarray(episode_id, dtype=np.int64),
    )


def save_probe_data(path, data: ProbeData) -> Path:
    out = Path(path)
    if not out.name.endswith(PROBE_ONLY_SUFFIX):
        raise ValueError(f"probe payload must end with {PROBE_ONLY_SUFFIX}")
    torch.save(
        {
            "marker": PROBE_ONLY_MARKER,
            "frames": torch.from_numpy(data.frames),
            "vitals": torch.from_numpy(data.vitals),
            "inventory": torch.from_numpy(data.inventory),
            "achievements": torch.from_numpy(data.achievements),
            "episode_id": torch.from_numpy(data.episode_id),
        },
        out,
    )
    return out


def load_probe_data(path) -> ProbeData:
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or payload.get("marker") != PROBE_ONLY_MARKER:
        raise RuntimeError("not a probe-only payload")
    return ProbeData(
        frames=payload["frames"].numpy(),
        vitals=payload["vitals"].numpy(),
        inventory=payload["inventory"].numpy(),
        achievements=payload["achievements"].numpy(),
        episode_id=payload["episode_id"].numpy(),
    )


# ---------------------------------------------------------------------------
# Mode preservation + encoding.
# ---------------------------------------------------------------------------
@contextmanager
def preserved_modes(module: torch.nn.Module):
    """Record every submodule's training flag and restore it exactly on exit.

    Generalizes the D062 fix: a diagnostic may switch modes for inference but
    must never leave the module in a different heterogeneous mode map than it
    found (which is how the D062 BatchNorm bug was introduced).
    """
    saved = {name: m.training for name, m in module.named_modules()}
    try:
        yield
    finally:
        for name, m in module.named_modules():
            if name in saved:
                m.train(saved[name])


@torch.inference_mode()
def encode_latents(world, frames: np.ndarray, *, batch: int = 256) -> np.ndarray:
    """Encode ``[N,C,H,W]`` uint8 frames to per-frame flattened online latents.

    The world's mode map is preserved and restored around the call.
    """
    device = next(world.parameters()).device
    out = []
    with preserved_modes(world):
        world.eval()
        for start in range(0, frames.shape[0], batch):
            chunk = torch.from_numpy(frames[start:start + batch]).to(device)
            packed = world.encode_frames(chunk[:, None], frozen=True).packed
            out.append(packed.reshape(chunk.shape[0], -1).float().cpu().numpy())
    return np.concatenate(out, axis=0)


# ---------------------------------------------------------------------------
# Feature sources.
# ---------------------------------------------------------------------------
def _within_episode_index(episode_id: np.ndarray) -> np.ndarray:
    """Per-frame position within its episode (frames are stored in order)."""
    idx = np.zeros(len(episode_id), dtype=np.float64)
    counts: dict[int, int] = {}
    for i, e in enumerate(episode_id):
        c = counts.get(int(e), 0)
        idx[i] = c
        counts[int(e)] = c + 1
    return idx


def constant_features(n: int) -> np.ndarray:
    return np.ones((n, 1), dtype=np.float32)


def timestep_features(episode_id: np.ndarray) -> np.ndarray:
    """Within-episode age plus its square (captures monotone drift with time)."""
    t = _within_episode_index(episode_id)
    t = t / max(1.0, t.max())
    return np.stack([t, t ** 2], axis=1).astype(np.float32)


def pixel_features(frames: np.ndarray) -> np.ndarray:
    """Raw flattened pixels in [0,1] (the linear information ceiling)."""
    return frames.reshape(frames.shape[0], -1).astype(np.float32) / 255.0


def _mlp_predict(
    x_train, y_train, x_val, y_val, x_test,
    *, hidden: int = 128, max_steps: int = 400, lr: float = 1e-3,
    weight_decay: float = 1e-4, seed: int = 0, device=None,
) -> np.ndarray:
    """A high-capacity MLP probe with val early-stopping.

    Linear ridge answers "is the information similarly EASY to extract"; this
    nonlinear probe answers "does the information EXIST at all", so a target that
    a linear probe misses but an MLP recovers is present-but-nonlinear, not lost.
    """
    torch.manual_seed(seed)
    device = torch.device(device) if device is not None else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu")

    def std(a, mu, sd):
        return (a - mu) / sd

    xmu = x_train.mean(0, keepdims=True)
    xsd = np.where(x_train.std(0, keepdims=True) < 1e-8, 1.0, x_train.std(0, keepdims=True))
    ymu = y_train.mean(0, keepdims=True)
    ysd = np.where(y_train.std(0, keepdims=True) < 1e-8, 1.0, y_train.std(0, keepdims=True))
    to = lambda a: torch.tensor(a, dtype=torch.float32, device=device)
    xtr, xva, xte = to(std(x_train, xmu, xsd)), to(std(x_val, xmu, xsd)), to(std(x_test, xmu, xsd))
    ytr, yva = to(std(y_train, ymu, ysd)), to(std(y_val, ymu, ysd))

    net = torch.nn.Sequential(
        torch.nn.Linear(xtr.shape[1], hidden), torch.nn.ReLU(),
        torch.nn.Linear(hidden, ytr.shape[1]),
    ).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)
    best_val, best_state, patience, waited = float("inf"), None, 40, 0
    for step in range(max_steps):
        net.train()
        opt.zero_grad()
        loss = ((net(xtr) - ytr) ** 2).mean()
        loss.backward()
        opt.step()
        if step % 10 == 0:
            net.eval()
            with torch.no_grad():
                vloss = float(((net(xva) - yva) ** 2).mean())
            if vloss < best_val - 1e-4:
                best_val, best_state = vloss, {k: v.detach().clone() for k, v in net.state_dict().items()}
                waited = 0
            else:
                waited += 1
                if waited >= patience:
                    break
    if best_state is not None:
        net.load_state_dict(best_state)
    net.eval()
    with torch.no_grad():
        pred = net(xte).cpu().numpy()
    return pred * ysd + ymu


def _cnn_predict(
    frames_train, y_train, frames_val, y_val, frames_test,
    *, max_steps: int = 400, lr: float = 1e-3, weight_decay: float = 1e-4,
    seed: int = 0, device=None,
) -> np.ndarray:
    """Small CNN probe over raw [N,3,H,W] frames = the nonlinear "does the
    information EXIST in the pixels" ceiling.

    A dense MLP on flattened pixels has ~1.5M params for 64x64 and overfits
    catastrophically at probe sample sizes; a conv net has spatial inductive
    bias and ~20k params, so it is the correct high-capacity pixel ceiling.
    """
    torch.manual_seed(seed)
    device = torch.device(device) if device is not None else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu")
    ymu = y_train.mean(0, keepdims=True)
    ysd = np.where(y_train.std(0, keepdims=True) < 1e-8, 1.0, y_train.std(0, keepdims=True))

    def img(a):
        return torch.tensor(a.astype(np.float32) / 255.0, device=device)

    def tgt(a):
        return torch.tensor((a - ymu) / ysd, dtype=torch.float32, device=device)

    xtr, xva, xte = img(frames_train), img(frames_val), img(frames_test)
    ytr, yva = tgt(y_train), tgt(y_val)
    net = torch.nn.Sequential(
        torch.nn.Conv2d(3, 16, 3, stride=2, padding=1), torch.nn.ReLU(),
        torch.nn.Conv2d(16, 32, 3, stride=2, padding=1), torch.nn.ReLU(),
        torch.nn.Conv2d(32, 32, 3, stride=2, padding=1), torch.nn.ReLU(),
        # Keep a 4x4 spatial grid (not global-average-pool) so the probe can
        # still read position-specific HUD/inventory cells, then flatten.
        torch.nn.AdaptiveAvgPool2d(4), torch.nn.Flatten(),
        torch.nn.Linear(32 * 16, ytr.shape[1]),
    ).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)
    best_val, best_state, patience, waited = float("inf"), None, 40, 0
    for step in range(max_steps):
        net.train()
        opt.zero_grad()
        loss = ((net(xtr) - ytr) ** 2).mean()
        loss.backward()
        opt.step()
        if step % 10 == 0:
            net.eval()
            with torch.no_grad():
                vloss = float(((net(xva) - yva) ** 2).mean())
            if vloss < best_val - 1e-4:
                best_val, best_state = vloss, {k: v.detach().clone() for k, v in net.state_dict().items()}
                waited = 0
            else:
                waited += 1
                if waited >= patience:
                    break
    if best_state is not None:
        net.load_state_dict(best_state)
    net.eval()
    with torch.no_grad():
        pred = net(xte).cpu().numpy()
    return pred * ysd + ymu


# ---------------------------------------------------------------------------
# Per-target continuous probing with episode CIs.
# ---------------------------------------------------------------------------
def _episode_three_way(episode_id, *, seed, val_frac=0.2, test_frac=0.3):
    ids = np.unique(episode_id)
    rng = np.random.default_rng(seed)
    rng.shuffle(ids)
    n_val = max(1, int(round(val_frac * len(ids))))
    n_test = max(1, int(round(test_frac * len(ids))))
    test_ids = set(int(i) for i in ids[:n_test])
    val_ids = set(int(i) for i in ids[n_test:n_test + n_val])
    train = np.array(
        [int(e) not in test_ids and int(e) not in val_ids for e in episode_id]
    )
    val = np.array([int(e) in val_ids for e in episode_id])
    test = np.array([int(e) in test_ids for e in episode_id])
    return train, val, test


def _per_target_r2_ci(y_true, pred, ep_test, *, seed, names):
    out = {}
    for k, name in enumerate(names):
        yk, pk = y_true[:, k:k + 1], pred[:, k:k + 1]

        def stat(idx_vals):
            idx = idx_vals.astype(int)
            return float(r2_per_target(yk[idx], pk[idx])[0])

        row = np.arange(y_true.shape[0], dtype=np.float64)
        point, ci = episode_bootstrap_ci(row, ep_test, stat, seed=seed + k, draws=800)
        out[name] = {"r2": float(r2_per_target(yk, pk)[0]), "r2_ci": ci}
    return out


def _r2_vec(y_true, pred):
    return {n: float(v) for n, v in zip(range(pred.shape[1]), r2_per_target(y_true, pred))}


def continuous_group(
    probe: ProbeData, group: str, targets: np.ndarray, names, latent, *,
    split_seed, margin=0.05,
):
    train, val, test = _episode_three_way(probe.episode_id, seed=split_seed)
    ep_test = probe.episode_id[test]
    yte = targets[test]

    # Linear (ridge) extractability for every source.
    lin = {}
    for s, x in {
        "constant": constant_features(len(targets)),
        "timestep": timestep_features(probe.episode_id),
        "pixel": pixel_features(probe.frames),
        "latent": latent,
    }.items():
        pred = select_and_predict(x[train], targets[train], x[val], targets[val], x[test])[0]
        lin[s] = _r2_vec(yte, pred)
        if s == "latent":
            latent_ci = _per_target_r2_ci(yte, pred, ep_test, seed=split_seed, names=names)

    # Nonlinear existence ceiling: a CNN over raw frames for pixels (spatial
    # inductive bias; a dense MLP on flattened pixels overfits), an MLP for the
    # low-dimensional latent.
    non = {
        "pixel": _r2_vec(yte, _cnn_predict(
            probe.frames[train], targets[train], probe.frames[val],
            targets[val], probe.frames[test], seed=split_seed)),
        "latent": _r2_vec(yte, _mlp_predict(
            latent[train], targets[train], latent[val], targets[val],
            latent[test], seed=split_seed)),
    }

    verdicts = {}
    for k, name in enumerate(names):
        r_const, r_time = lin["constant"][k], lin["timestep"][k]
        floor = max(r_const, r_time)
        pix_lin, pix_non = lin["pixel"][k], non["pixel"][k]
        lat_lin, lat_non = lin["latent"][k], non["latent"][k]
        exists = max(pix_lin, pix_non)           # can pixels reveal it at all?
        latent_exists = max(lat_lin, lat_non)
        if exists - floor <= 0.02:
            verdict = "inconclusive_ceiling"     # not in the frame beyond time/const
        elif latent_exists - floor <= 0.02:
            verdict = "lost"                     # absent from latent even nonlinearly
        elif latent_exists >= exists - margin:
            verdict = "preserved"                # present in latent as much as pixels
        else:
            verdict = "degraded"                 # partially present
        verdicts[name] = {
            "verdict": verdict,
            "constant_r2": r_const, "timestep_r2": r_time,
            "pixel_linear_r2": pix_lin, "pixel_nonlinear_r2": pix_non,
            "latent_linear_r2": lat_lin, "latent_nonlinear_r2": lat_non,
            "latent_linear_ci": latent_ci[name]["r2_ci"],
            "beats_timestep": bool(latent_exists - r_time > 0.02),
            "linearly_as_extractable": bool(lat_lin >= pix_lin - margin),
        }
    return {"per_target": verdicts}


def achievement_group(probe: ProbeData, latent, *, split_seed):
    """Per-achievement AUROC/AP/Brier from the latent, where supported."""
    from .craftax_env import achievement_names

    names = achievement_names()
    train, val, test = _episode_three_way(probe.episode_id, seed=split_seed)
    y = probe.achievements.astype(np.float32)
    # ridge scores as a monotone probe signal for ranking metrics
    scores = select_and_predict(latent[train], y[train], latent[val], y[val], latent[test])[0]
    yt = probe.achievements[test]
    out = {}
    for k, name in enumerate(names):
        pos = int(yt[:, k].sum())
        out[name] = {
            "positives": pos,
            "auroc": auroc(scores[:, k], yt[:, k]),
            "ap": average_precision(scores[:, k], yt[:, k]),
            "brier": brier(np.clip(scores[:, k], 0, 1), yt[:, k]),
        }
    return out


# ---------------------------------------------------------------------------
# Self-audit.
# ---------------------------------------------------------------------------
def audit_probe_machinery(*, seed: int = 0, n_ep: int = 20, per: int = 20) -> dict:
    """perfect~1, constant~0, misaligned~0, timestep-shift low."""
    rng = np.random.default_rng(seed)
    episode_id = np.repeat(np.arange(n_ep), per)
    n = episode_id.shape[0]
    targets = rng.normal(size=(n, 3)).astype(np.float32)
    train, val, test = _episode_three_way(episode_id, seed=seed)

    def r2(x):
        pred = select_and_predict(x[train], targets[train], x[val], targets[val], x[test])[0]
        return float(np.mean(r2_per_target(targets[test], pred)))

    # shift labels by one frame within each episode -> off-by-one detector
    shifted = targets.copy()
    for e in np.unique(episode_id):
        m = np.where(episode_id == e)[0]
        shifted[m] = np.roll(targets[m], 1, axis=0)

    def r2_shift(x):
        pred = select_and_predict(x[train], shifted[train], x[val], shifted[val], x[test])[0]
        return float(np.mean(r2_per_target(targets[test], pred)))

    perfect = r2(targets.copy())
    constant = r2(np.ones((n, 4), dtype=np.float32))
    misaligned = r2(rng.normal(size=(n, 6)).astype(np.float32))
    shift = r2_shift(targets.copy())
    return {
        "perfect_r2": perfect,
        "constant_r2": constant,
        "misaligned_r2": misaligned,
        "timestep_shift_r2": shift,
        "pass": perfect > 0.95 and abs(constant) < 0.05
        and misaligned < 0.1 and shift < 0.5,
    }


def representation_oracle(world, probe: ProbeData, *, split_seed: int = 20260726):
    """Full Stage-2 report: per-target vitals + inventory + achievements."""
    latent = encode_latents(world, probe.frames)
    return {
        "audit": audit_probe_machinery(),
        "vitals": continuous_group(
            probe, "vitals", probe.vitals, VITAL_NAMES, latent, split_seed=split_seed),
        "inventory": continuous_group(
            probe, "inventory", probe.inventory, INVENTORY_NAMES, latent,
            split_seed=split_seed),
        "achievements": achievement_group(probe, latent, split_seed=split_seed),
    }


__all__ = [
    "PROBE_ONLY_SUFFIX", "PROBE_ONLY_MARKER", "ProbeData",
    "collect_probe_data", "save_probe_data", "load_probe_data",
    "preserved_modes", "encode_latents",
    "constant_features", "timestep_features", "pixel_features",
    "continuous_group", "achievement_group",
    "audit_probe_machinery", "representation_oracle",
]
