"""Frozen-context task-information probe for the Phase-E depth diagnosis.

Protocol: the frozen-context task-information probe in
reviews/2026-07-18-phase-e-depth-diagnostic-protocol.md.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

COMPACT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = COMPACT_ROOT.parent
sys.path.insert(0, str(COMPACT_ROOT))
sys.path.insert(0, str(COMPACT_ROOT / "verification"))

from fork_oracle_v2 import sha256_file  # noqa: E402
from model import WorldState  # noqa: E402
from phase_e_same_target import (  # noqa: E402
    ARTIFACTS,
    BATCH,
    CHECKPOINTS,
    HORIZONS,
    HISTORY,
    WINDOW_OBS,
    average_precision,
    load_world,
    suffix_partition,
    target_rows,
    window_arrays,
)
from phase_e_taskheads import auroc, clone_world_state  # noqa: E402
from step3_temporal import (  # noqa: E402
    HELDOUT_20_CACHE,
    TRAIN_40K_CACHE,
    load_scaled_data,
)
from step4_runner import git_head, software_versions, source_digest  # noqa: E402

REPORT_PATH = ARTIFACTS / "phase_e_context_probe.json"
ROWS_PATH = ARTIFACTS / "phase_e_context_probe_rows.json"
PROTOCOL = (
    "reviews/2026-07-18-phase-e-depth-diagnostic-protocol.md"
    "#frozen-context-task-information-probe"
)
PROBE_STEPS = 300
PROBE_LR = 1e-3
PROBE_WEIGHT_DECAY = 1e-4
BOOTSTRAP_DRAWS = 1000
SELECTION_SEED = 1820
PROBE_SEED = 1821


def task_labels(episodes: list[dict],
                rows: list[dict[str, int]]) -> dict[str, np.ndarray]:
    rewards = np.asarray(
        [episodes[row["episode"]]["rewards"][row["transition"]]
         for row in rows],
        dtype=np.float32,
    )
    continues = np.asarray(
        [episodes[row["episode"]]["continues"][row["transition"]]
         for row in rows],
        dtype=np.float32,
    )
    return {
        "reward_event": np.abs(rewards) > 1e-6,
        "reward_positive": rewards > 1e-6,
        "reward_negative": rewards < -1e-6,
        "terminal": continues < 0.5,
    }


def balanced_task_indices(
    labels: dict[str, np.ndarray],
    rng: np.random.Generator,
) -> dict[str, np.ndarray]:
    tasks = {
        "reward_event": (
            np.flatnonzero(labels["reward_event"]),
            np.flatnonzero(~labels["reward_event"]),
        ),
        "reward_sign": (
            np.flatnonzero(labels["reward_positive"]),
            np.flatnonzero(labels["reward_negative"]),
        ),
        "terminal": (
            np.flatnonzero(labels["terminal"]),
            np.flatnonzero(~labels["terminal"]),
        ),
    }
    output = {}
    for task, (positive, negative) in tasks.items():
        count = min(len(positive), len(negative))
        positive = np.sort(rng.choice(positive, count, replace=False))
        negative = np.sort(rng.choice(negative, count, replace=False))
        output[task] = np.sort(np.concatenate([positive, negative]))
    return output


def binary_targets(task: str, labels: dict[str, np.ndarray],
                   indices: np.ndarray | None = None) -> np.ndarray:
    if task == "reward_event":
        result = labels["reward_event"]
    elif task == "reward_sign":
        result = labels["reward_positive"]
    elif task == "terminal":
        result = labels["terminal"]
    else:
        raise ValueError(task)
    return result if indices is None else result[indices]


@torch.no_grad()
def extract_contexts(world, arrays: dict, device: torch.device) -> dict:
    contexts = {depth: [] for depth in HORIZONS}
    for start in range(0, len(arrays["rewards"]), BATCH):
        stop = min(start + BATCH, len(arrays["rewards"]))
        obs = torch.from_numpy(arrays["obs"][start:stop]).to(device)
        actions = torch.from_numpy(arrays["actions"][start:stop]).to(device)
        previous = torch.from_numpy(
            arrays["previous_actions"][start:stop]).to(device)
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

        for depth in HORIZONS:
            state = clone_world_state(base)
            real_times, imagined_actions = suffix_partition(depth)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                for time in real_times:
                    index = world._previous_action_indices(previous[:, time])
                    value = encoded[time] + world.action_input(index)[:, None]
                    output, temporal = world.temporal.step(
                        value, state.temporal)
                    state = WorldState(temporal, output, state.revision)
                for action_index in imagined_actions:
                    state, _, _, _ = world.imagine_step(
                        state,
                        actions[:, action_index],
                        deterministic_mode=True,
                    )
                pooled = world.pool(state.tokens)
            contexts[depth].append(pooled.float().cpu())
    return {
        f"k{depth}": torch.cat(contexts[depth]).numpy()
        for depth in HORIZONS
    }


class BinaryProbe(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, 2 * dim),
            nn.SiLU(),
            nn.Linear(2 * dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def _bootstrap_metrics(
    scores: np.ndarray,
    labels: np.ndarray,
    clusters: np.ndarray,
) -> tuple[list[float], list[float]]:
    unique = np.unique(clusters)
    members = {value: np.flatnonzero(clusters == value) for value in unique}
    rng = np.random.default_rng(1822)
    aucs, aps = [], []
    for _ in range(BOOTSTRAP_DRAWS):
        index = np.concatenate(
            [members[value]
             for value in rng.choice(unique, len(unique), replace=True)]
        )
        value = auroc(scores[index], labels[index])
        if value is not None:
            aucs.append(value)
        value = average_precision(scores[index], labels[index])
        if value is not None:
            aps.append(value)
    return (
        [float(x) for x in np.percentile(aucs, (2.5, 97.5))],
        [float(x) for x in np.percentile(aps, (2.5, 97.5))],
    )


def fit_probe(
    train_context: np.ndarray,
    train_label: np.ndarray,
    eval_context: np.ndarray,
    device: torch.device,
) -> tuple[np.ndarray, dict]:
    torch.manual_seed(PROBE_SEED)
    probe = BinaryProbe(train_context.shape[-1]).to(device)
    optimizer = torch.optim.AdamW(
        probe.parameters(), lr=PROBE_LR, weight_decay=PROBE_WEIGHT_DECAY)
    x = torch.from_numpy(train_context).to(device)
    y = torch.from_numpy(train_label.astype(np.float32)).to(device)
    probe.train()
    for _ in range(PROBE_STEPS):
        logits = probe(x)
        loss = F.binary_cross_entropy_with_logits(logits, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    probe.eval()
    with torch.no_grad():
        train_scores = torch.sigmoid(probe(x)).cpu().numpy()
        eval_scores = torch.sigmoid(
            probe(torch.from_numpy(eval_context).to(device))).cpu().numpy()
    return eval_scores, {
        "final_loss": float(loss.detach()),
        "train_auroc": auroc(train_scores, train_label),
    }


def _file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    device = torch.device("cuda")
    train_episodes, heldout_episodes = load_scaled_data()
    train_rows_all = target_rows(train_episodes)
    heldout_rows = target_rows(heldout_episodes)
    train_labels_all = task_labels(train_episodes, train_rows_all)
    heldout_labels = task_labels(heldout_episodes, heldout_rows)
    selected = balanced_task_indices(
        train_labels_all, np.random.default_rng(SELECTION_SEED))
    union = np.unique(np.concatenate(list(selected.values())))
    union_position = {int(index): position
                      for position, index in enumerate(union)}
    selected_in_union = {
        task: np.asarray([union_position[int(index)] for index in indices])
        for task, indices in selected.items()
    }
    train_rows = [train_rows_all[int(index)] for index in union]
    train_arrays = window_arrays(train_episodes, train_rows)
    heldout_arrays = window_arrays(heldout_episodes, heldout_rows)
    heldout_clusters_all = heldout_arrays["episodes"]

    report = {
        "protocol": PROTOCOL,
        "head": git_head(),
        "source_digest": source_digest(),
        "script_sha256": _file_sha(Path(__file__)),
        "versions": software_versions(),
        "hashes": {
            "train_replay": sha256_file(TRAIN_40K_CACHE),
            "heldout": sha256_file(HELDOUT_20_CACHE),
            "checkpoints": {
                f"fullgrid_{backend}_s{seed}": sha256_file(path)
                for backend, seed, path in CHECKPOINTS
            },
        },
        "probe_contract": {
            "steps": PROBE_STEPS,
            "learning_rate": PROBE_LR,
            "weight_decay": PROBE_WEIGHT_DECAY,
            "selection_seed": SELECTION_SEED,
            "probe_seed": PROBE_SEED,
            "bootstrap_draws": BOOTSTRAP_DRAWS,
            "train_union_rows": int(len(union)),
            "heldout_rows": int(len(heldout_rows)),
            "task_train_rows": {
                task: int(len(indices)) for task, indices in selected.items()
            },
        },
        "checkpoints": {},
    }
    raw = {
        "train_selected_rows": {
            task: [train_rows_all[int(index)] for index in indices]
            for task, indices in selected.items()
        },
        "heldout_rows": heldout_rows,
        "heldout_labels": {
            key: value.astype(bool).tolist()
            for key, value in heldout_labels.items()
        },
        "probe_scores": {},
    }

    for backend, seed, path in CHECKPOINTS:
        tag = f"fullgrid_{backend}_s{seed}"
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        world = load_world(backend, path, device)
        train_contexts = extract_contexts(world, train_arrays, device)
        heldout_contexts = extract_contexts(world, heldout_arrays, device)
        del world

        block = {}
        raw["probe_scores"][tag] = {}
        for task in ("reward_event", "reward_sign", "terminal"):
            train_index = selected_in_union[task]
            train_label = binary_targets(
                task, train_labels_all, selected[task])
            if task == "reward_sign":
                eval_mask = heldout_labels["reward_event"]
            else:
                eval_mask = np.ones(
                    len(heldout_rows), dtype=bool)
            eval_label = binary_targets(
                task, heldout_labels)[eval_mask]
            eval_clusters = heldout_clusters_all[eval_mask]
            block[task] = {}
            raw["probe_scores"][tag][task] = {}
            for depth in HORIZONS:
                key = f"k{depth}"
                scores, fit = fit_probe(
                    train_contexts[key][train_index],
                    train_label,
                    heldout_contexts[key][eval_mask],
                    device,
                )
                auc = auroc(scores, eval_label)
                ap = average_precision(scores, eval_label)
                auc_ci, ap_ci = _bootstrap_metrics(
                    scores, eval_label, eval_clusters)
                block[task][key] = {
                    **fit,
                    "heldout_n": int(len(eval_label)),
                    "heldout_positive": int(eval_label.sum()),
                    "heldout_auroc": auc,
                    "heldout_auroc_ci95": auc_ci,
                    "heldout_average_precision": ap,
                    "heldout_average_precision_ci95": ap_ci,
                }
                raw["probe_scores"][tag][task][key] = scores.tolist()
        block["peak_allocated_mib"] = \
            torch.cuda.max_memory_allocated() / 2**20
        block["peak_reserved_mib"] = \
            torch.cuda.max_memory_reserved() / 2**20
        report["checkpoints"][tag] = block
        print(
            tag,
            {
                task: {
                    key: round(value["heldout_auroc"], 4)
                    for key, value in block[task].items()
                }
                for task in ("reward_event", "reward_sign", "terminal")
            },
            flush=True,
        )

    ROWS_PATH.write_text(json.dumps(raw))
    report["hashes"]["raw_rows"] = sha256_file(ROWS_PATH)
    REPORT_PATH.write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
