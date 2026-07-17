"""Phase E: task-head validation of the committed full-grid checkpoints.

Protocol: reviews/2026-07-18-phase-e-protocol.md (pre-registered). Evaluation
only: fixed checkpoints, deterministic windows, no training, no selection.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

COMPACT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = COMPACT_ROOT.parent
sys.path.insert(0, str(COMPACT_ROOT))
sys.path.insert(0, str(COMPACT_ROOT / "verification"))

from data import Episode, EpisodeReplay  # noqa: E402
from model import M3HJWM, ModelConfig, WorldState, TemporalState, symexp  # noqa: E402
from ssl_ijepa import IJEPAPretrainer  # noqa: E402
from step3_temporal import load_scaled_data  # noqa: E402
from fork_oracle_v2 import ENCODER_CKPT, sha256_file  # noqa: E402
from consolidation import ARTIFACTS, build_world  # noqa: E402
from step4_runner import git_head, software_versions, source_digest  # noqa: E402
from exploratory_topology import build_exploratory_world  # noqa: E402
from run_exploratory_topology import BUNDLE_PATH, MANIFEST_PATH  # noqa: E402

TERMINAL_SET = REPO_ROOT / "data" / "terminal_enriched_900_915.pt"
REPORT_PATH = ARTIFACTS / "phase_e_report.json"
GAMMA = 0.997
PREFIX_OBS = 8            # imagined-horizon evaluation prefix
HORIZONS = (1, 2, 4, 8)

CHECKPOINTS = (
    [("fullgrid_mamba2", f"X-FLM_s{s}", "X-FLM", s) for s in (505, 606, 707)]
    + [("fullgrid_gru", f"X-FLG_s{s}", "X-FLG", s) for s in (505, 606, 707)]
)
POOLED_REFERENCE = [("pooled_gru64", f"M1_gru64_s{s}", s) for s in (101, 202, 303)]


def decode_reward(logits: torch.Tensor, cfg: ModelConfig) -> torch.Tensor:
    centers = torch.linspace(cfg.reward_low, cfg.reward_high, cfg.reward_bins,
                             device=logits.device)
    return symexp((logits.softmax(-1) * centers).sum(-1))


def auroc(scores: np.ndarray, labels: np.ndarray) -> float | None:
    pos, neg = scores[labels], scores[~labels]
    if len(pos) == 0 or len(neg) == 0:
        return None
    greater = (pos[:, None] > neg[None, :]).mean()
    ties = (pos[:, None] == neg[None, :]).mean()
    return float(greater + 0.5 * ties)


def ece(probabilities: np.ndarray, labels: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0, 1, bins + 1)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (probabilities >= lo) & (probabilities < hi)
        if mask.any():
            total += mask.mean() * abs(probabilities[mask].mean()
                                       - labels[mask].mean())
    return float(total)


def clone_world_state(state: WorldState) -> WorldState:
    cache = state.temporal.cache
    if cache is None:
        cloned = None
    else:
        cloned = []
        for entry in cache:
            if isinstance(entry, tuple):
                cloned.append(tuple(t.clone() for t in entry))
            else:
                cloned.append(entry.clone())
    return WorldState(TemporalState(cloned, state.temporal.output.clone()),
                      state.tokens.clone(), state.revision)


def collect_terminal_enriched():
    """Random-policy episodes run to termination on eval-only seeds 900-915."""
    import crafter
    episodes = []
    for env_seed in range(900, 916):
        env = crafter.Env(seed=env_seed, length=10_000)
        rng = np.random.default_rng(env_seed)
        obs = env.reset()
        frames = [np.ascontiguousarray(obs.transpose(2, 0, 1))]
        actions, rewards, dones = [], [], []
        done = False
        while not done:
            action = int(rng.integers(env.action_space.n))
            obs, reward, done, _ = env.step(action)
            frames.append(np.ascontiguousarray(obs.transpose(2, 0, 1)))
            actions.append(action)
            rewards.append(float(reward))
            dones.append(bool(done))
        episodes.append({
            "env_seed": env_seed,
            "obs": np.stack(frames).astype(np.uint8),
            "actions": np.asarray(actions, dtype=np.int64),
            "rewards": np.asarray(rewards, dtype=np.float32),
            "continues": (1.0 - np.asarray(dones, dtype=np.float32)),
        })
        del env
    torch.save(episodes, TERMINAL_SET)
    return episodes


def terminal_windows(episodes, window: int = 16, rng_seed: int = 977):
    """One terminal-crossing window per episode + one matched non-terminal."""
    rng = np.random.default_rng(rng_seed)
    batches = []
    for ep in episodes:
        n = len(ep["obs"])
        if n < window + 1:
            continue
        starts = [n - window]                       # crosses the terminal step
        if n - window - 8 > 0:
            starts.append(int(rng.integers(0, n - window - 8)))
        for start in starts:
            previous = np.full(window, -1, dtype=np.int64)
            if start > 0:
                previous[0] = ep["actions"][start - 1]
            previous[1:] = ep["actions"][start:start + window - 1]
            batches.append({
                "obs": ep["obs"][start:start + window],
                "actions": ep["actions"][start:start + window - 1],
                "rewards": ep["rewards"][start:start + window - 1],
                "continues": ep["continues"][start:start + window - 1],
                "previous_actions": previous,
            })
    return batches


def load_checkpoint_world(name, arm, seed, device):
    ckpt = torch.load(ARTIFACTS / f"xtopo_{name}_16000.pt", weights_only=False)
    world = build_exploratory_world(arm, seed, device)
    world.load_state_dict(ckpt["state_dict"], strict=True)
    return world.eval()


def load_pooled_world(name, device):
    ckpt = torch.load(ARTIFACTS / f"step4_{name}_16000.pt", weights_only=False)
    world = build_world("global_gru", 64, device)
    world.load_state_dict(ckpt["state_dict"], strict=True)
    return world.eval()


@torch.no_grad()
def teacher_forced_metrics(world, batches, device) -> dict:
    nll, mae, scores, labels = [], [], [], []
    for batch in batches:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = world(batch)
        logits = out.reward_logits.float()
        actual = batch["rewards"].float()
        nll.append(float(world.reward.loss(logits, actual).mean()))
        decoded = decode_reward(logits, world.cfg)
        mae.append(float((decoded - actual).abs().mean()))
        scores.append(decoded.abs().flatten().cpu().numpy())
        labels.append((actual.abs() > 1e-6).flatten().cpu().numpy())
    scores = np.concatenate(scores)
    labels = np.concatenate(labels).astype(bool)
    return {"nll": float(np.mean(nll)), "mae": float(np.mean(mae)),
            "event_auroc": auroc(scores, labels),
            "event_rate": float(labels.mean()), "n_steps": int(len(labels))}


@torch.no_grad()
def imagined_horizon_metrics(world, batches, device) -> dict:
    per_h = {k: {"nll": [], "mae": [], "scores": [], "labels": []}
             for k in range(1, 9)}
    pred_returns, real_returns = [], []
    for batch in batches:
        b = batch["obs"].shape[0]
        state = world.initial_state(b, device)
        for t in range(PREFIX_OBS):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                state = world.observe_step(
                    batch["obs"][:, t], batch["previous_actions"][:, t], state)
        pred_sum = torch.zeros(b, device=device)
        real_sum = torch.zeros(b, device=device)
        for k in range(1, 9):
            action = batch["actions"][:, PREFIX_OBS - 1 + k - 1]
            with torch.autocast("cuda", dtype=torch.bfloat16):
                state, reward_logits, _, _ = world.imagine_step(
                    state, action, deterministic_mode=True)
            logits = reward_logits.float()
            actual = batch["rewards"][:, PREFIX_OBS - 1 + k - 1].float()
            decoded = decode_reward(logits, world.cfg)
            pred_sum = pred_sum + decoded
            real_sum = real_sum + actual
            slot = per_h[k]
            slot["nll"].append(float(world.reward.loss(logits, actual).mean()))
            slot["mae"].append(float((decoded - actual).abs().mean()))
            slot["scores"].append(decoded.abs().cpu().numpy())
            slot["labels"].append((actual.abs() > 1e-6).cpu().numpy())
        pred_returns.append(pred_sum.cpu().numpy())
        real_returns.append(real_sum.cpu().numpy())
    pred_returns = np.concatenate(pred_returns)
    real_returns = np.concatenate(real_returns)
    def corr(a, b):
        if a.std() == 0 or b.std() == 0:
            return None
        return float(np.corrcoef(a, b)[0, 1])
    def spearman(a, b):
        ra, rb = a.argsort().argsort().astype(float), b.argsort().argsort().astype(float)
        return corr(ra, rb)
    out = {}
    for k in HORIZONS:
        slot = per_h[k]
        scores = np.concatenate(slot["scores"])
        labels = np.concatenate(slot["labels"]).astype(bool)
        out[f"h{k}"] = {"nll": float(np.mean(slot["nll"])),
                        "mae": float(np.mean(slot["mae"])),
                        "event_auroc": auroc(scores, labels),
                        "event_rate": float(labels.mean())}
    out["return8_pearson"] = corr(pred_returns, real_returns)
    out["return8_spearman"] = spearman(pred_returns, real_returns)
    out["n_windows"] = int(len(pred_returns))
    return out


@torch.no_grad()
def ranking_metrics(world, anchors, device) -> dict:
    rows = []
    for anchor in anchors:
        obs = torch.from_numpy(anchor["obs_hist"][None]).to(device)
        act_hist = anchor["act_hist"]
        state = world.initial_state(1, device)
        for index in range(len(act_hist)):
            previous = torch.tensor([int(act_hist[index])], device=device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                state = world.observe_step(obs[:, index], previous, state)
        j_gated, j_sum, actual = {}, {}, {}
        for name, suffix in anchor["suffixes"].items():
            branch_state = clone_world_state(state)
            discount = torch.ones(1, device=device)
            alive = torch.ones(1, device=device)
            total_g = torch.zeros(1, device=device)
            total_s = torch.zeros(1, device=device)
            for k, action_value in enumerate(suffix):
                action = torch.tensor([int(action_value)], device=device)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    branch_state, reward_logits, continue_logits, _ = \
                        world.imagine_step(branch_state, action,
                                           deterministic_mode=True)
                reward = decode_reward(reward_logits.float(), world.cfg)
                total_g = total_g + (GAMMA ** k) * alive * reward
                total_s = total_s + reward
                alive = alive * torch.sigmoid(continue_logits.float())
            j_gated[name] = float(total_g)
            j_sum[name] = float(total_s)
            actual[name] = float(np.mean(
                [o["reward_sum"] for o in anchor["branches"][name]["outcomes"]]))
        names = list(anchor["suffixes"])
        actual_values = np.array([actual[n] for n in names])
        gated_values = np.array([j_gated[n] for n in names])
        chosen = names[int(gated_values.argmax())]
        rows.append({
            "env_seed": int(anchor["env_seed"]), "night": bool(anchor["night"]),
            "j_gated": j_gated, "j_sum": j_sum, "actual": actual,
            "differs": bool(actual_values.std() > 1e-9),
            "chosen_minus_random": float(actual[chosen] - actual_values.mean()),
            "regret": float(actual_values.max() - actual[chosen]),
        })
    differing = [r for r in rows if r["differs"]]
    def within_corrs(kind):
        pearson, spear = [], []
        for r in differing:
            a = np.array(list(r["actual"].values()))
            p = np.array([r[kind][n] for n in r["actual"]])
            if a.std() == 0 or p.std() == 0:
                continue
            pearson.append(float(np.corrcoef(p, a)[0, 1]))
            ra = a.argsort().argsort().astype(float)
            rp = p.argsort().argsort().astype(float)
            spear.append(float(np.corrcoef(rp, ra)[0, 1]))
        return (float(np.mean(pearson)) if pearson else None,
                float(np.mean(spear)) if spear else None)
    p_g, s_g = within_corrs("j_gated")
    p_s, s_s = within_corrs("j_sum")
    adv = np.array([r["chosen_minus_random"] for r in differing])
    by_env = {}
    for r in differing:
        by_env.setdefault(r["env_seed"], []).append(r["chosen_minus_random"])
    rng = np.random.default_rng(0)
    clusters = [np.array(v) for v in by_env.values()]
    stats = np.empty(10_000)
    for i in range(10_000):
        picked = rng.integers(len(clusters), size=len(clusters))
        stats[i] = float(np.mean(np.concatenate([clusters[j] for j in picked])))
    return {"rows": rows, "n_differing": len(differing),
            "pearson_gated": p_g, "spearman_gated": s_g,
            "pearson_sum": p_s, "spearman_sum": s_s,
            "chosen_minus_random_mean": float(adv.mean()) if len(adv) else None,
            "chosen_minus_random_ci95": [float(np.percentile(stats, 2.5)),
                                         float(np.percentile(stats, 97.5))],
            "regret_mean": float(np.mean([r["regret"] for r in differing]))
            if differing else None}


@torch.no_grad()
def continuation_metrics(world, windows, device) -> dict:
    probabilities, labels = [], []
    for w in windows:
        batch = {k: torch.from_numpy(np.asarray(v)[None]).to(device)
                 for k, v in w.items()}
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = world(batch)
        p = torch.sigmoid(out.continue_logits.float()).flatten().cpu().numpy()
        y = np.asarray(w["continues"], dtype=np.float32).flatten()
        probabilities.append(p)
        labels.append(y)
    probabilities = np.concatenate(probabilities)
    labels = np.concatenate(labels)
    binary = labels < 0.5   # terminal steps are the positive class
    return {"brier": float(np.mean((probabilities - labels) ** 2)),
            "terminal_auroc": auroc(1.0 - probabilities, binary),
            "ece": ece(probabilities, labels),
            "n_steps": int(len(labels)),
            "terminal_rate": float(binary.mean())}


def main():
    device = torch.device("cuda")
    manifest = json.loads(MANIFEST_PATH.read_text())
    assert sha256_file(BUNDLE_PATH) == manifest["sha256"]
    anchors = torch.load(BUNDLE_PATH, weights_only=False)
    _, heldout = load_scaled_data()
    replay = EpisodeReplay(capacity_steps=500_000)
    for ep in heldout:
        replay.add(Episode(**ep))
    rng = np.random.default_rng(4242)
    tf_batches = [replay.sample(batch=8, observations=16, device=device, rng=rng)
                  for _ in range(8)]
    rng = np.random.default_rng(2424)
    im_batches = [replay.sample(batch=8, observations=16, device=device, rng=rng)
                  for _ in range(7)]
    if TERMINAL_SET.exists():
        terminal_eps = torch.load(TERMINAL_SET, weights_only=False)
    else:
        terminal_eps = collect_terminal_enriched()
    windows = terminal_windows(terminal_eps)

    report = {"protocol": "reviews/2026-07-18-phase-e-protocol.md",
              "head": git_head(), "source_digest": source_digest(),
              "versions": software_versions(),
              "hashes": {"bundle": sha256_file(BUNDLE_PATH),
                         "terminal_set": sha256_file(TERMINAL_SET),
                         "encoder": sha256_file(ENCODER_CKPT)},
              "gamma": GAMMA, "checkpoints": {}}

    pretrainer = IJEPAPretrainer(
        ModelConfig(temporal_backend="gru", predictor="deterministic", mask_ratio=0.0))
    pretrainer.load_state_dict(
        torch.load(ENCODER_CKPT, weights_only=False)["pretrainer"], strict=True)
    _ = pretrainer  # encoder identity pinned by hash; worlds embed their own

    def evaluate(tag, world):
        block = {"teacher_forced": teacher_forced_metrics(world, tf_batches, device),
                 "imagined": imagined_horizon_metrics(world, im_batches, device),
                 "ranking": ranking_metrics(world, anchors, device),
                 "continuation": continuation_metrics(world, windows, device)}
        rows = block["ranking"].pop("rows")
        (ARTIFACTS / f"phase_e_ranking_rows_{tag}.json").write_text(json.dumps(rows))
        report["checkpoints"][tag] = block
        REPORT_PATH.write_text(json.dumps(report, indent=2, default=str))
        r = block["ranking"]
        print(f"[{tag}] rank_spear {r['spearman_gated']} adv "
              f"{r['chosen_minus_random_mean']} CI {r['chosen_minus_random_ci95']} "
              f"h1_auroc {block['imagined']['h1']['event_auroc']} "
              f"cont_auroc {block['continuation']['terminal_auroc']}", flush=True)

    for family, name, arm, seed in CHECKPOINTS:
        evaluate(f"{family}_s{seed}", load_checkpoint_world(name, arm, seed, device))
        torch.cuda.empty_cache()
    for family, name, seed in POOLED_REFERENCE:
        evaluate(f"{family}_s{seed}", load_pooled_world(name, device))
        torch.cuda.empty_cache()

    # -------- pre-registered gate (margins pending consensus) --------
    def gate(family):
        seeds = [s for f, n, a, s in CHECKPOINTS if f == family] or [101, 202, 303]
        blocks = [report["checkpoints"][f"{family}_s{s}"] for s in seeds]
        advantages = [b["ranking"]["chosen_minus_random_mean"] for b in blocks]
        signs_consistent = all(a is not None for a in advantages) and (
            all(a > 0 for a in advantages) or all(a < 0 for a in advantages))
        cis = [b["ranking"]["chosen_minus_random_ci95"] for b in blocks]
        pooled_ci_excludes = all(c[0] > 0 for c in cis) or all(c[1] < 0 for c in cis)
        h1 = float(np.mean([b["imagined"]["h1"]["event_auroc"] or 0 for b in blocks]))
        h8 = float(np.mean([b["imagined"]["h8"]["event_auroc"] or 0 for b in blocks]))
        cont_auroc = float(np.mean(
            [b["continuation"]["terminal_auroc"] or 0 for b in blocks]))
        brier = float(np.mean([b["continuation"]["brier"] for b in blocks]))
        return {"G_E1": bool(signs_consistent and pooled_ci_excludes
                             and advantages[0] > 0),
                "G_E2": bool(h1 >= 0.75 and h8 >= 0.65),
                "G_E3": bool(cont_auroc >= 0.80 and brier <= 0.20),
                "advantages": advantages, "h1_auroc": h1, "h8_auroc": h8,
                "continuation_auroc": cont_auroc, "brier": brier}
    report["gates"] = {"fullgrid_mamba2": gate("fullgrid_mamba2"),
                       "fullgrid_gru": gate("fullgrid_gru")}
    for fam, g in report["gates"].items():
        g["planner_ready"] = bool(g["G_E1"] and g["G_E2"] and g["G_E3"])
    report["gate_note"] = ("margins are the protocol PROPOSAL; acting on the "
                           "gate requires user+companion consensus")
    REPORT_PATH.write_text(json.dumps(report, indent=2, default=str))
    print(json.dumps(report["gates"], indent=2))


if __name__ == "__main__":
    main()
