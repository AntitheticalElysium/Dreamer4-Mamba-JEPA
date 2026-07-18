"""Same-target continuation-depth supplement for Phase E.

Protocol: the continuation-depth supplement in
reviews/2026-07-18-phase-e-depth-diagnostic-protocol.md.
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
    bootstrap_indices,
    load_world,
    suffix_partition,
    target_rows,
    window_arrays,
)
from phase_e_taskheads import auroc, clone_world_state  # noqa: E402
from step3_temporal import HELDOUT_20_CACHE  # noqa: E402
from step4_runner import git_head, software_versions, source_digest  # noqa: E402

REPORT_PATH = ARTIFACTS / "phase_e_same_target_continuation.json"
ROWS_PATH = ARTIFACTS / "phase_e_same_target_continuation_rows.json"
PROTOCOL = (
    "reviews/2026-07-18-phase-e-depth-diagnostic-protocol.md"
    "#continuation-depth-supplement"
)
BOOTSTRAP_DRAWS = 1000


def continuation_targets(episodes: list[dict],
                         rows: list[dict[str, int]]) -> np.ndarray:
    return np.asarray(
        [episodes[row["episode"]]["continues"][row["transition"]]
         for row in rows],
        dtype=np.float32,
    )


def continuation_metrics(
    predicted_continue: np.ndarray,
    actual_continue: np.ndarray,
    boot_indices: list[np.ndarray],
) -> dict:
    terminal = actual_continue < 0.5
    termination_probability = 1.0 - predicted_continue
    auc = auroc(termination_probability, terminal)
    ap = average_precision(termination_probability, terminal)
    brier = float(np.mean((predicted_continue - actual_continue) ** 2))
    climatology = float(np.mean(
        (actual_continue - actual_continue.mean()) ** 2))
    eps = 1e-7
    p = np.clip(predicted_continue, eps, 1.0 - eps)
    nll = -(actual_continue * np.log(p)
            + (1.0 - actual_continue) * np.log(1.0 - p))

    auc_boot, ap_boot, brier_skill_boot = [], [], []
    for index in boot_indices:
        value = auroc(termination_probability[index], terminal[index])
        if value is not None:
            auc_boot.append(value)
        value = average_precision(
            termination_probability[index], terminal[index])
        if value is not None:
            ap_boot.append(value)
        y = actual_continue[index]
        base = float(np.mean((y - y.mean()) ** 2))
        if base > 0:
            score = float(np.mean((predicted_continue[index] - y) ** 2))
            brier_skill_boot.append(1.0 - score / base)

    def mean(mask: np.ndarray, values: np.ndarray) -> float | None:
        return float(values[mask].mean()) if bool(mask.any()) else None

    return {
        "n": int(len(actual_continue)),
        "terminals": int(terminal.sum()),
        "terminal_rate": float(terminal.mean()),
        "terminal_auroc": auc,
        "terminal_auroc_ci95": [
            float(x) for x in np.percentile(auc_boot, (2.5, 97.5))
        ],
        "terminal_average_precision": ap,
        "terminal_average_precision_ci95": [
            float(x) for x in np.percentile(ap_boot, (2.5, 97.5))
        ],
        "brier": brier,
        "climatology_brier": climatology,
        "brier_skill": 1.0 - brier / climatology,
        "brier_skill_ci95": [
            float(x) for x in np.percentile(
                brier_skill_boot, (2.5, 97.5))
        ],
        "binary_nll": float(nll.mean()),
        "binary_nll_terminal": mean(terminal, nll),
        "binary_nll_nonterminal": mean(~terminal, nll),
        "predicted_termination_terminal_mean": mean(
            terminal, termination_probability),
        "predicted_termination_nonterminal_mean": mean(
            ~terminal, termination_probability),
        "terminal_recall_at_0_5": float(
            (termination_probability[terminal] >= 0.5).mean()),
        "nonterminal_false_positive_at_0_5": float(
            (termination_probability[~terminal] >= 0.5).mean()),
    }


@torch.no_grad()
def evaluate_world(world, arrays: dict, actual_continue: np.ndarray,
                   device: torch.device) -> dict:
    boot = bootstrap_indices(
        arrays["episodes"], draws=BOOTSTRAP_DRAWS)
    predictions = {depth: [] for depth in HORIZONS}

    for start in range(0, len(actual_continue), BATCH):
        stop = min(start + BATCH, len(actual_continue))
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
                logits = world.continuation(world.pool(state.tokens))
            predictions[depth].append(
                torch.sigmoid(logits.float()).cpu().numpy())

    result = {"metrics": {}, "predictions": {}}
    for depth in HORIZONS:
        predicted = np.concatenate(predictions[depth])
        result["metrics"][f"k{depth}"] = continuation_metrics(
            predicted, actual_continue, boot)
        result["predictions"][f"k{depth}"] = predicted.tolist()
    return result


def _file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    device = torch.device("cuda")
    episodes = torch.load(HELDOUT_20_CACHE, weights_only=False)
    rows = target_rows(episodes)
    arrays = window_arrays(episodes, rows)
    actual_continue = continuation_targets(episodes, rows)
    assert len(rows) == 3262
    # Six held-out episodes were capped at 200 steps without a recorded
    # terminal. Never relabel those truncations as absorbing transitions.
    assert int((actual_continue < 0.5).sum()) == 14

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
            "max_depth": 8,
            "horizons": HORIZONS,
            "n_targets": len(rows),
            "n_terminals": int((actual_continue < 0.5).sum()),
            "n_episodes": len(episodes),
            "bootstrap_draws": BOOTSTRAP_DRAWS,
        },
        "checkpoints": {},
    }
    raw = {
        "episode": arrays["episodes"].tolist(),
        "transition": arrays["transitions"].tolist(),
        "actual_continue": actual_continue.tolist(),
        "predictions": {},
    }

    for backend, seed, path in CHECKPOINTS:
        tag = f"fullgrid_{backend}_s{seed}"
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        world = load_world(backend, path, device)
        result = evaluate_world(world, arrays, actual_continue, device)
        report["checkpoints"][tag] = {
            "metrics": result["metrics"],
            "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
            "peak_reserved_mib": torch.cuda.max_memory_reserved() / 2**20,
        }
        raw["predictions"][tag] = result["predictions"]
        print(
            tag,
            "AUROC",
            {key: round(value["terminal_auroc"], 4)
             for key, value in result["metrics"].items()},
            flush=True,
        )
        del world

    ROWS_PATH.write_text(json.dumps(raw))
    report["hashes"]["raw_rows"] = sha256_file(ROWS_PATH)
    REPORT_PATH.write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
