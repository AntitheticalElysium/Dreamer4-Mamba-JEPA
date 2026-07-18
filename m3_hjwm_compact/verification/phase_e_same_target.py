"""Same-target Phase-E depth diagnostic.

Protocol: reviews/2026-07-18-phase-e-depth-diagnostic-protocol.md.
Evaluation only: six fixed full-grid checkpoints, one common held-out target
set, and suffix replacement at K in {0, 1, 2, 4, 8}.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

COMPACT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = COMPACT_ROOT.parent
sys.path.insert(0, str(COMPACT_ROOT))
sys.path.insert(0, str(COMPACT_ROOT / "verification"))

from checkpoint import sprint_candidate_config  # noqa: E402
from fork_oracle_v2 import sha256_file  # noqa: E402
from model import (M3HJWM, TemporalState, WorldState, cosine_distance)  # noqa: E402
from phase_e_taskheads import auroc, clone_world_state  # noqa: E402
from step3_temporal import HELDOUT_20_CACHE  # noqa: E402
from step4_runner import git_head, software_versions, source_digest  # noqa: E402

ARTIFACTS = REPO_ROOT / "reviews" / "artifacts"
REPORT_PATH = ARTIFACTS / "phase_e_same_target_depth.json"
ROWS_PATH = ARTIFACTS / "phase_e_same_target_rows.json"
PROTOCOL = "reviews/2026-07-18-phase-e-depth-diagnostic-protocol.md"

HISTORY = 8
MAX_DEPTH = 8
WINDOW_OBS = HISTORY + MAX_DEPTH
HORIZONS = (0, 1, 2, 4, 8)
BATCH = 64
BOOTSTRAP_DRAWS = 1000

CHECKPOINTS = tuple(
    (backend, seed, ARTIFACTS / f"xtopo_X-FL{code}_s{seed}_16000.pt")
    for backend, code in (("mamba2", "M"), ("gru", "G"))
    for seed in (505, 606, 707)
)


def target_rows(episodes: list[dict]) -> list[dict[str, int]]:
    """All transitions sharing eligibility for K=0,1,2,4,8."""
    rows = []
    first_target = HISTORY + MAX_DEPTH - 2  # j=14 for 16 observations.
    for episode, item in enumerate(episodes):
        for transition in range(first_target, len(item["rewards"])):
            rows.append({"episode": episode, "transition": transition})
    return rows


def window_arrays(episodes: list[dict], rows: list[dict[str, int]]) -> dict:
    """Build aligned 16-observation windows ending at o[j+1]."""
    obs, actions, previous, rewards = [], [], [], []
    for row in rows:
        episode = episodes[row["episode"]]
        target = row["transition"]
        start = target - (HISTORY + MAX_DEPTH - 2)
        obs.append(episode["obs"][start:start + WINDOW_OBS])
        actions.append(episode["actions"][start:start + WINDOW_OBS - 1])
        prev = np.full(WINDOW_OBS, -1, dtype=np.int64)
        if start:
            prev[0] = episode["actions"][start - 1]
        prev[1:] = episode["actions"][start:start + WINDOW_OBS - 1]
        previous.append(prev)
        rewards.append(episode["rewards"][target])
    return {
        "obs": np.stack(obs).astype(np.uint8),
        "actions": np.stack(actions).astype(np.int64),
        "previous_actions": np.stack(previous).astype(np.int64),
        "rewards": np.asarray(rewards, dtype=np.float32),
        "episodes": np.asarray([row["episode"] for row in rows], dtype=np.int64),
        "transitions": np.asarray(
            [row["transition"] for row in rows], dtype=np.int64),
    }


def suffix_partition(depth: int) -> tuple[range, range]:
    if depth not in HORIZONS:
        raise ValueError(f"unsupported depth {depth}")
    # Tokens 0..7 initialize the shared base state. At depth K, observations
    # 8..15-K are real and action indices 15-K..14 are imagined.
    return range(HISTORY, WINDOW_OBS - depth), range(WINDOW_OBS - 1 - depth,
                                                     WINDOW_OBS - 1)


def _rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1)
        start = end
    return ranks


def _corr(a: np.ndarray, b: np.ndarray, rank: bool = False) -> float | None:
    if rank:
        a, b = _rankdata(a), _rankdata(b)
    if a.std() == 0 or b.std() == 0:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def average_precision(scores: np.ndarray, labels: np.ndarray) -> float | None:
    labels = labels.astype(bool)
    positives = int(labels.sum())
    if positives == 0:
        return None
    order = np.argsort(-scores, kind="mergesort")
    scores, labels = scores[order], labels[order]
    tp = fp = 0
    recall_before = 0.0
    result = 0.0
    start = 0
    while start < len(scores):
        end = start + 1
        while end < len(scores) and scores[end] == scores[start]:
            end += 1
        tp += int(labels[start:end].sum())
        fp += int(end - start - labels[start:end].sum())
        recall = tp / positives
        result += (recall - recall_before) * (tp / (tp + fp))
        recall_before = recall
        start = end
    return float(result)


def bootstrap_indices(clusters: np.ndarray, draws: int = BOOTSTRAP_DRAWS):
    unique = np.unique(clusters)
    members = {value: np.flatnonzero(clusters == value) for value in unique}
    rng = np.random.default_rng(1807)
    return [
        np.concatenate([members[value] for value in
                        rng.choice(unique, len(unique), replace=True)])
        for _ in range(draws)
    ]


def prediction_metrics(decoded: np.ndarray, nll: np.ndarray,
                       actual: np.ndarray, clusters: np.ndarray,
                       boot_indices: list[np.ndarray]) -> dict:
    events = np.abs(actual) > 1e-6
    zero = ~events
    scores = np.abs(decoded)
    auc = auroc(scores, events)
    ap = average_precision(scores, events)
    auc_boot, ap_boot = [], []
    for index in boot_indices:
        value = auroc(scores[index], events[index])
        if value is not None:
            auc_boot.append(value)
        value = average_precision(scores[index], events[index])
        if value is not None:
            ap_boot.append(value)

    def mean(mask, values):
        return float(values[mask].mean()) if bool(mask.any()) else None

    positive, negative = actual > 1e-6, actual < -1e-6
    return {
        "n": int(len(actual)),
        "events": int(events.sum()),
        "event_rate": float(events.mean()),
        "event_auroc": auc,
        "event_auroc_ci95": [float(x) for x in np.percentile(
            auc_boot, (2.5, 97.5))],
        "event_average_precision": ap,
        "event_average_precision_ci95": [float(x) for x in np.percentile(
            ap_boot, (2.5, 97.5))],
        "nll": float(nll.mean()),
        "nll_event": mean(events, nll),
        "nll_zero": mean(zero, nll),
        "mae": float(np.abs(decoded - actual).mean()),
        "mae_event": mean(events, np.abs(decoded - actual)),
        "mae_zero": mean(zero, np.abs(decoded - actual)),
        "reward_pearson": _corr(decoded, actual),
        "reward_spearman": _corr(decoded, actual, rank=True),
        "decoded_positive_mean": mean(positive, decoded),
        "decoded_negative_mean": mean(negative, decoded),
        "decoded_zero_mean": mean(zero, decoded),
        "decoded_abs_event_mean": mean(events, scores),
        "decoded_abs_zero_mean": mean(zero, scores),
    }


def load_world(backend: str, path: Path, device: torch.device) -> M3HJWM:
    checkpoint = torch.load(path, weights_only=False)
    world = M3HJWM(sprint_candidate_config(backend)).to(device)
    world.load_state_dict(checkpoint["state_dict"], strict=True)
    return world.eval()


@torch.no_grad()
def evaluate_world(world: M3HJWM, arrays: dict, device: torch.device) -> dict:
    actual_all = arrays["rewards"]
    cluster_all = arrays["episodes"]
    boot = bootstrap_indices(cluster_all)
    predictions = {depth: [] for depth in HORIZONS}
    nlls = {depth: [] for depth in HORIZONS}
    token_drift = {depth: [] for depth in HORIZONS}
    pooled_drift = {depth: [] for depth in HORIZONS}

    for start in range(0, len(actual_all), BATCH):
        stop = min(start + BATCH, len(actual_all))
        obs = torch.from_numpy(arrays["obs"][start:stop]).to(device)
        actions = torch.from_numpy(arrays["actions"][start:stop]).to(device)
        previous = torch.from_numpy(
            arrays["previous_actions"][start:stop]).to(device)
        actual = torch.from_numpy(actual_all[start:stop]).to(device)
        batch = stop - start

        encoded = []
        with torch.autocast("cuda", dtype=torch.bfloat16):
            for time in range(WINDOW_OBS):
                encoded.append(world.online_encoder(obs[:, time]))

        state = world.initial_state(batch, device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            for time in range(HISTORY):
                index = world._previous_action_indices(previous[:, time])
                value = encoded[time] + world.action_input(index)[:, None]
                output, temporal = world.temporal.step(value, state.temporal)
                state = WorldState(temporal, output, state.revision)
        base = state

        final_states = {}
        logits_by_depth = {}
        for depth in HORIZONS:
            state = clone_world_state(base)
            real_times, imagined_actions = suffix_partition(depth)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                for time in real_times:
                    index = world._previous_action_indices(previous[:, time])
                    value = encoded[time] + world.action_input(index)[:, None]
                    output, temporal = world.temporal.step(value, state.temporal)
                    state = WorldState(temporal, output, state.revision)
                if depth:
                    logits = None
                    for action_index in imagined_actions:
                        state, logits, _, _ = world.imagine_step(
                            state, actions[:, action_index],
                            deterministic_mode=True)
                    assert logits is not None
                else:
                    logits = world.reward(world.pool(state.tokens))
            final_states[depth] = state.tokens.float()
            logits_by_depth[depth] = logits.float()

        real_state = final_states[0]
        for depth in HORIZONS:
            logits = logits_by_depth[depth]
            predictions[depth].append(
                world.reward.decode(logits).float().cpu().numpy())
            nlls[depth].append(
                world.reward.loss(logits, actual).float().cpu().numpy())
            token_drift[depth].append(
                cosine_distance(final_states[depth], real_state).mean(-1)
                .cpu().numpy())
            pooled_drift[depth].append(
                cosine_distance(world.pool(final_states[depth]),
                                world.pool(real_state)).cpu().numpy())

    output = {"metrics": {}, "predictions": {}}
    for depth in HORIZONS:
        decoded = np.concatenate(predictions[depth])
        nll = np.concatenate(nlls[depth])
        metrics = prediction_metrics(
            decoded, nll, actual_all, cluster_all, boot)
        metrics["token_context_cosine_drift"] = float(
            np.concatenate(token_drift[depth]).mean())
        metrics["pooled_context_cosine_drift"] = float(
            np.concatenate(pooled_drift[depth]).mean())
        output["metrics"][f"k{depth}"] = metrics
        output["predictions"][f"k{depth}"] = decoded.tolist()
    return output


def _file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    device = torch.device("cuda")
    episodes = torch.load(HELDOUT_20_CACHE, weights_only=False)
    rows = target_rows(episodes)
    arrays = window_arrays(episodes, rows)
    event_count = int((np.abs(arrays["rewards"]) > 1e-6).sum())
    assert len(rows) == 3262 and event_count == 140

    report = {
        "protocol": PROTOCOL,
        "head": git_head(),
        "source_digest": source_digest(),
        "script_sha256": _file_sha(Path(__file__)),
        "versions": software_versions(),
        "hashes": {
            "heldout": sha256_file(HELDOUT_20_CACHE),
            "checkpoints": {
                f"fullgrid_{backend}_s{seed}": sha256_file(path)
                for backend, seed, path in CHECKPOINTS
            },
        },
        "construction": {
            "history": HISTORY,
            "max_depth": MAX_DEPTH,
            "horizons": HORIZONS,
            "n_targets": len(rows),
            "n_events": event_count,
            "n_episodes": len(episodes),
            "bootstrap_draws": BOOTSTRAP_DRAWS,
        },
        "checkpoints": {},
    }
    raw = {
        "episode": arrays["episodes"].tolist(),
        "transition": arrays["transitions"].tolist(),
        "actual_reward": arrays["rewards"].tolist(),
        "predictions": {},
    }

    for backend, seed, path in CHECKPOINTS:
        tag = f"fullgrid_{backend}_s{seed}"
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        world = load_world(backend, path, device)
        result = evaluate_world(world, arrays, device)
        report["checkpoints"][tag] = {
            "metrics": result["metrics"],
            "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
            "peak_reserved_mib": torch.cuda.max_memory_reserved() / 2**20,
        }
        raw["predictions"][tag] = result["predictions"]
        print(
            tag,
            "AUC",
            {key: round(value["event_auroc"], 4)
             for key, value in result["metrics"].items()},
            flush=True,
        )
        del world

    families = {}
    for backend in ("mamba2", "gru"):
        blocks = [
            report["checkpoints"][f"fullgrid_{backend}_s{seed}"]["metrics"]
            for seed in (505, 606, 707)
        ]
        families[backend] = {}
        for depth in HORIZONS:
            key = f"k{depth}"
            families[backend][key] = {}
            for metric in (
                "event_auroc", "event_average_precision", "nll", "mae",
                "reward_pearson", "reward_spearman",
                "token_context_cosine_drift", "pooled_context_cosine_drift",
            ):
                values = [block[key][metric] for block in blocks]
                families[backend][key][metric] = {
                    "mean": float(np.mean(values)),
                    "min": float(np.min(values)),
                    "max": float(np.max(values)),
                    "per_seed": values,
                }
    report["families"] = families
    ROWS_PATH.write_text(json.dumps(raw))
    report["hashes"]["raw_rows"] = sha256_file(ROWS_PATH)
    REPORT_PATH.write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

