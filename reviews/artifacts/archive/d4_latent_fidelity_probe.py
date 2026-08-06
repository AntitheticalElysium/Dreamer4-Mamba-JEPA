"""Gate-3 latent fidelity probe: do imagined latents still track real state?

Trains a probe on REAL agent tokens to predict the four CartPole state
variables (x, x_dot, theta, theta_dot) and terminal-within-5, then evaluates
that frozen probe on IMAGINED agent tokens at horizons h in {1,4,8,16,32}, for
BC and anti-BC closed-loop rollouts.

Ground truth at horizon h is obtained by replaying the exact action sequence the
policy chose in imagination through the pinned CartPole dynamics, starting from
the true state at the end of the real context. CartPole is deterministic given
(state, action), so this is exact.

Three curves are reported, which together separate the three failure modes the
single latent-cosine number cannot:

  1. probe R^2 / AUC vs horizon      -> does the latent still carry real state?
  2. across-batch variance vs horizon -> does the batch collapse to a point?
  3. |pred(BC) - pred(anti-BC)| vs h  -> does action-conditioned movement survive?

Usage:
  python d4_latent_fidelity_probe.py --world <world.pt> --bc <policy.pt> \
      --label M-JEPA --output <report.json>
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from d4_mamba_jepa.cartpole_baseline import (
    ACTION_REPEAT,
    CartPolePixels,
    _clean_agent_tokens,
    load_bc_policy,
)
from d4_mamba_jepa.checkpoint import file_sha256, load_checkpoint
from d4_mamba_jepa.data import SequenceBatch
from d4_mamba_jepa.rollout import sample_next_packed

HORIZONS = (1, 4, 8, 16, 32)
STATE_NAMES = ("cart_x", "cart_x_dot", "pole_theta", "pole_theta_dot")
CONTEXT = 8
MAX_HORIZON = max(HORIZONS)
TERMINAL_WITHIN = 5


class Scaled(nn.Module):
    """anti-BC = the BC head with inverted logits (a deliberately bad policy)."""

    def __init__(self, base: nn.Module, scale: float):
        super().__init__()
        self.base, self.scale = base, scale

    def forward(self, x):
        return self.base(x) * self.scale


# --------------------------------------------------------------------------
# Episode collection with ground-truth state
# --------------------------------------------------------------------------
def collect_episodes(
    world, bc, *, episodes: int, seed0: int, device, epsilon: float
) -> list[dict]:
    """Roll real episodes, recording observations, actions and true states."""
    env = CartPolePixels(image_size=world.cfg.image_size)
    rng = np.random.default_rng(seed0)
    out = []
    try:
        for index in range(episodes):
            obs = env.reset(seed=seed0 + index)
            observations = [obs]
            states = [env.state.copy()]
            actions: list[int] = []
            dones: list[bool] = []
            previous_action = -1
            while True:
                # Act from the frozen BC head on the clean context so the data
                # distribution matches deployment; epsilon adds state coverage.
                if rng.random() < epsilon:
                    action = int(rng.integers(0, 2))
                else:
                    window = observations[-CONTEXT:]
                    led = ([-1] + actions)[-len(window):]
                    batch = _batch_from(window, led, device, world)
                    with torch.inference_mode():
                        tokens = _clean_agent_tokens(world, batch)
                        logits = bc(tokens[:, -1:].float())[:, 0]
                    action = int(logits.argmax(-1).item())
                obs, _, _, terminated, truncated = env.step(action)
                actions.append(action)
                observations.append(obs)
                states.append(env.state.copy())
                done = bool(terminated or truncated)
                dones.append(done)
                previous_action = action
                if done or len(actions) >= 500:
                    break
            out.append(
                {
                    "obs": np.stack(observations),
                    "states": np.stack(states),
                    "actions": np.array(actions, dtype=np.int64),
                    "terminated": bool(dones[-1]) if dones else False,
                }
            )
    finally:
        env.close()
    return out


def _batch_from(obs_window, led_actions, device, world) -> SequenceBatch:
    observations = torch.from_numpy(np.stack(obs_window))[None].to(device)
    led = torch.tensor(led_actions, dtype=torch.long, device=device)[None]
    time = observations.shape[1]
    if led.shape[1] < time:  # pad the start action
        pad = torch.full(
            (1, time - led.shape[1]), -1, dtype=torch.long, device=device
        )
        led = torch.cat([pad, led], dim=1)
    return SequenceBatch(
        observations=observations,
        led_to_actions=led[:, :time],
        led_to_rewards=torch.zeros(1, time, device=device),
        led_to_continues=torch.ones(1, time, device=device),
        outcome_valid=torch.ones(1, time, device=device),
    )


# --------------------------------------------------------------------------
# Probe fitting on real latents
# --------------------------------------------------------------------------
def pooled(tokens: torch.Tensor) -> torch.Tensor:
    """[.., N, D] agent tokens -> flat feature vector."""
    return tokens.reshape(*tokens.shape[:-2], -1)


def _encode_windows(world, windows, leds, device, chunk=128):
    """Batched clean agent tokens for many [CONTEXT] windows."""
    feats = []
    for start in range(0, len(windows), chunk):
        obs = torch.from_numpy(
            np.stack(windows[start : start + chunk])
        ).to(device)
        led = torch.from_numpy(
            np.stack(leds[start : start + chunk])
        ).long().to(device)
        batch = SequenceBatch(
            observations=obs,
            led_to_actions=led,
            led_to_rewards=torch.zeros(*led.shape, device=device),
            led_to_continues=torch.ones(*led.shape, device=device),
            outcome_valid=torch.ones(*led.shape, device=device),
        )
        with torch.inference_mode():
            tok = _clean_agent_tokens(world, batch)[:, -1]
        feats.append(pooled(tok).float().cpu().numpy())
    return np.concatenate(feats)


def build_real_dataset(world, episodes, device):
    windows, leds, states, terminal = [], [], [], []
    for ep in episodes:
        n = len(ep["actions"])
        if n < CONTEXT + 1:
            continue
        obs, acts, sts = ep["obs"], ep["actions"], ep["states"]
        for t in range(CONTEXT, n + 1):
            windows.append(obs[t - CONTEXT : t])
            leds.append(np.concatenate([[-1], acts[: t - 1]])[-CONTEXT:])
            states.append(sts[t - 1])
            steps_left = n - (t - 1)
            terminal.append(
                1.0 if (ep["terminated"] and steps_left <= TERMINAL_WITHIN) else 0.0
            )
    feats = _encode_windows(world, windows, leds, device)
    return (
        feats,
        np.stack(states).astype(np.float64),
        np.array(terminal),
    )


def fit_ridge(X, Y, alpha=1.0):
    Xb = np.concatenate([X, np.ones((len(X), 1))], axis=1)
    A = Xb.T @ Xb + alpha * np.eye(Xb.shape[1])
    return np.linalg.solve(A, Xb.T @ Y)


def apply_ridge(W, X):
    return np.concatenate([X, np.ones((len(X), 1))], axis=1) @ W


def fit_logistic(X, y, alpha=1.0, iters=300, lr=0.5):
    Xb = np.concatenate([X, np.ones((len(X), 1))], axis=1)
    w = np.zeros(Xb.shape[1])
    for _ in range(iters):
        p = 1.0 / (1.0 + np.exp(-Xb @ w))
        grad = Xb.T @ (p - y) / len(y) + alpha * w / len(y)
        w -= lr * grad
    return w


def auc(scores, labels):
    if labels.sum() == 0 or labels.sum() == len(labels):
        return float("nan")
    order = np.argsort(scores)
    ranks = np.empty(len(scores), dtype=float)
    ranks[order] = np.arange(1, len(scores) + 1)
    pos, neg = labels == 1, labels == 0
    return float(
        (ranks[pos].sum() - pos.sum() * (pos.sum() + 1) / 2)
        / (pos.sum() * neg.sum())
    )


def r2(pred, true):
    ss_res = ((pred - true) ** 2).sum(axis=0)
    ss_tot = ((true - true.mean(axis=0)) ** 2).sum(axis=0)
    return 1.0 - ss_res / np.maximum(ss_tot, 1e-12)


# --------------------------------------------------------------------------
# Imagined rollouts + matched ground truth
# --------------------------------------------------------------------------
@torch.inference_mode()
def imagine_with_policy(world, policy, packed, led, device, horizon):
    """Closed-loop imagination. Returns imagined agent tokens and the actions."""
    steps = torch.full(
        (packed.shape[0], packed.shape[1]),
        world.cfg.max_step_index,
        device=device,
        dtype=torch.long,
    )
    signals = torch.full_like(steps, world.cfg.k_max)
    _, agent = world.forward_dynamics(packed, led, steps, signals)
    past, cur_led = packed, led
    tokens, actions = [], []
    state_tok = agent[:, -1:]
    for _ in range(horizon):
        logits = policy(state_tok.float())[:, 0]
        action = logits.argmax(-1)
        actions.append(action)
        cur_led = torch.cat([cur_led, action[:, None]], dim=1)
        nxt, new_agent = sample_next_packed(
            world, past_packed=past, led_to_actions=cur_led,
            schedule=None, use_cache=False,
        )
        past = torch.cat([past, nxt[:, None]], dim=1)
        state_tok = new_agent
        tokens.append(new_agent[:, 0])
    return torch.stack(tokens, dim=1), torch.stack(actions, dim=1)


def true_rollout(start_states, action_seq):
    """Replay actions through the pinned CartPole dynamics from start_states."""
    import gymnasium as gym

    env = gym.make("CartPole-v1")
    B, H = action_seq.shape
    out = np.full((B, H, 4), np.nan)
    alive = np.ones((B, H), dtype=bool)
    try:
        for b in range(B):
            env.reset(seed=0)
            env.unwrapped.state = np.array(start_states[b], dtype=np.float64)
            dead = False
            for h in range(H):
                if dead:
                    alive[b, h:] = False
                    break
                for _ in range(ACTION_REPEAT):
                    s, _, term, trunc, _ = env.step(int(action_seq[b, h]))
                    if term or trunc:
                        dead = True
                        break
                out[b, h] = np.asarray(s, dtype=np.float64)
    finally:
        env.close()
    return out, alive


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--world", type=Path, required=True)
    ap.add_argument("--bc", type=Path, required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--train-episodes", type=int, default=40)
    ap.add_argument("--eval-contexts", type=int, default=64)
    ap.add_argument("--seed", type=int, default=910000)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    world_sha = file_sha256(args.world)
    world, _, _ = load_checkpoint(
        args.world, device=device, expected_sha256=world_sha,
        strict_implementation=False,
    )
    world.eval()
    bc, _ = load_bc_policy(
        args.bc, expected_sha256=file_sha256(args.bc),
        expected_world_sha256=world_sha, device=device,
    )
    bc.eval()
    anti = Scaled(bc, -4.0).to(device).eval()
    print(f"[{args.label}] arm={world.cfg.arm_id} backend={world.cfg.temporal_backend}")

    # --- probe training set from real rollouts -------------------------------
    eps = collect_episodes(
        world, bc, episodes=args.train_episodes, seed0=args.seed,
        device=device, epsilon=0.2,
    )
    X, Y, term = build_real_dataset(world, eps, device)
    mu, sd = X.mean(0), X.std(0) + 1e-6
    Xn = (X - mu) / sd
    # Shuffle before splitting: consecutive rows come from the same episode, so
    # a sequential split puts train and test on different state distributions
    # and makes held-out R^2 meaningless.
    order = np.random.default_rng(args.seed).permutation(len(Xn))
    Xn, Y, term = Xn[order], Y[order], term[order]
    split = int(0.8 * len(Xn))
    W = fit_ridge(Xn[:split], Y[:split])
    wt = fit_logistic(Xn[:split], term[:split])
    real_r2 = r2(apply_ridge(W, Xn[split:]), Y[split:])
    real_auc = auc(Xn[split:] @ wt[:-1] + wt[-1], term[split:])
    print(f"  probe dataset: {len(Xn)} samples, {Xn.shape[1]} features, "
          f"terminal positives {int(term.sum())}")
    print(f"  real held-out R2 per state: "
          f"{dict(zip(STATE_NAMES, np.round(real_r2, 4)))}")
    print(f"  real held-out terminal-within-{TERMINAL_WITHIN} AUC: {real_auc:.4f}")

    # --- evaluation contexts (with true state at context end) ---------------
    # Draw contexts spread through each episode, not just the opening frames,
    # so the evaluation covers the state distribution the actor actually meets.
    ctxs, starts = [], []
    rng_ctx = np.random.default_rng(args.seed + 1)
    per_episode = max(1, args.eval_contexts // max(1, len(eps)) + 1)
    for ep in eps:
        n = len(ep["actions"])
        if n < CONTEXT + 2:
            continue
        hi = max(CONTEXT + 1, n - 1)
        picks = rng_ctx.choice(
            np.arange(CONTEXT, hi), size=min(per_episode, hi - CONTEXT),
            replace=False,
        )
        for t in picks:
            t = int(t)
            ctxs.append((ep["obs"][t - CONTEXT : t],
                         np.concatenate([[-1], ep["actions"][: t - 1]])[-CONTEXT:]))
            starts.append(ep["states"][t - 1])
    if len(ctxs) > args.eval_contexts:
        keep = rng_ctx.choice(len(ctxs), size=args.eval_contexts, replace=False)
        ctxs = [ctxs[i] for i in keep]
        starts = [starts[i] for i in keep]
    obs = torch.from_numpy(np.stack([c[0] for c in ctxs])).to(device)
    led = torch.from_numpy(np.stack([c[1] for c in ctxs])).long().to(device)
    starts = np.stack(starts)
    with torch.inference_mode():
        packed = world.encode_frames(obs, frozen=True).packed

    # Real-data spread of each state variable: the denominator for NRMSE.
    state_sd = Y.std(axis=0) + 1e-9

    results = {}
    preds = {}
    for name, pol in (("bc", bc), ("anti_bc", anti)):
        toks, acts = imagine_with_policy(
            world, pol, packed, led, device, MAX_HORIZON
        )
        acts_np = acts.cpu().numpy()
        truth, alive = true_rollout(starts, acts_np)
        feats = pooled(toks).float().cpu().numpy()
        per_h = {}
        preds[name] = {}
        for h in HORIZONS:
            f = (feats[:, h - 1] - mu) / sd
            p = apply_ridge(W, f)
            preds[name][h] = p
            m = alive[:, h - 1]
            # R^2 is unstable here: as h grows the alive subset shrinks and its
            # true-state variance collapses, so R^2 explodes negative even for
            # small absolute error. NRMSE (error in units of the real-data
            # standard deviation of each state variable) is scale-stable and is
            # the metric to read for fidelity.
            nrmse = (
                np.sqrt(((p[m] - truth[m, h - 1]) ** 2).mean(axis=0)) / state_sd
                if m.sum() > 3 else None
            )
            per_h[h] = {
                "n_alive": int(m.sum()),
                "n_total": int(len(m)),
                "nrmse_per_state": (
                    dict(zip(STATE_NAMES, np.round(nrmse, 4).tolist()))
                    if nrmse is not None else None
                ),
                "nrmse_mean": float(nrmse.mean()) if nrmse is not None else None,
                "r2_per_state": (
                    dict(zip(STATE_NAMES, np.round(r2(p[m], truth[m, h - 1]), 4).tolist()))
                    if m.sum() > 3 else None
                ),
                "terminal_auc_vs_alive": auc(
                    -(f @ wt[:-1] + wt[-1]), (~m).astype(float)
                ),
                "pred_state_variance": float(p.var(axis=0).mean()),
            }
        results[name] = per_h

    divergence = {
        h: float(np.abs(preds["bc"][h] - preds["anti_bc"][h]).mean())
        for h in HORIZONS
    }

    report = {
        "label": args.label,
        "arm_id": world.cfg.arm_id,
        "temporal_backend": world.cfg.temporal_backend,
        "world_sha256": world_sha,
        "bc_sha256": file_sha256(args.bc),
        "protocol": {
            "context": CONTEXT, "horizons": list(HORIZONS),
            "eval_contexts": len(ctxs), "train_episodes": args.train_episodes,
            "terminal_within": TERMINAL_WITHIN,
            "ground_truth": "same action sequence replayed through pinned CartPole dynamics",
        },
        "real_heldout": {
            "r2_per_state": dict(zip(STATE_NAMES, np.round(real_r2, 4).tolist())),
            "r2_mean": float(real_r2.mean()),
            "terminal_auc": real_auc,
        },
        "imagined": results,
        "bc_vs_antibc_pred_state_l1": divergence,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")

    def fmt(v, w=9):
        return f"{v:>{w}.4f}" if v is not None and np.isfinite(v) else " " * (w - 1) + "-"

    print(f"\n  {'h':>3} | {'alive':>11} {'NRMSE(BC)':>10} {'var(BC)':>9} "
          f"| {'alive':>11} {'NRMSE(anti)':>12} | {'|BC-anti|':>10}")
    for h in HORIZONS:
        b, a = results["bc"][h], results["anti_bc"][h]
        print(f"  {h:>3} | {b['n_alive']:>4}/{b['n_total']:<6} {fmt(b['nrmse_mean'],10)} "
              f"{b['pred_state_variance']:>9.4f} | {a['n_alive']:>4}/{a['n_total']:<6} "
              f"{fmt(a['nrmse_mean'],12)} | {divergence[h]:>10.4f}")
    print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
