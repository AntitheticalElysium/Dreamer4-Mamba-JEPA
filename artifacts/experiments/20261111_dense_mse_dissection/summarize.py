"""Paired scene summaries from the frozen readouts."""
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def read(name):
    return json.loads((HERE / name).read_text())["checkpoints"]


def paired(values):
    values = np.asarray(values)
    rng = np.random.default_rng(5517)
    ids = rng.integers(len(values), size=(10_000, len(values)))
    lo, hi = np.quantile(values[ids].mean(axis=1), [0.025, 0.975])
    return {"mean": float(values.mean()), "lo": float(lo), "hi": float(hi)}


def native():
    runs = read("NATIVE.json")
    cols = ("paired_first", "paired_final")
    out = {}
    for name, run in runs.items():
        result = out[name] = {}
        for steps in ("8", "5", "2", "1"):
            rows = [row["steps"][steps] for row in run["rows"]]
            result[steps] = {col: paired([np.mean(row[col]) for row in rows]) for col in cols}
            result[steps]["field_calls"] = np.mean(
                [row["mean_difference_per_field_call"] for row in rows], axis=0
            ).tolist()
            result[steps]["mean_final_by_phase"] = np.mean(
                [row["mean_final"] for row in rows], axis=0
            ).tolist()
            result[steps]["token_difference_rms_mean"] = float(np.mean([
                row["final_difference_rms_in_digit_steps"] for row in rows
            ]))

    def contrast(left, right, h, col):
        a, b = runs[left]["rows"], runs[right]["rows"]
        assert [r["e"] for r in a] == [r["e"] for r in b]
        return [np.mean(x["steps"][h][col]) - np.mean(y["steps"][h][col])
                for x, y in zip(a, b)]

    if {"dense_2000", "endpoint_2000", "dense_384", "parent"} <= runs.keys():
        out["due_fork_checkpoint_contrasts"] = {
            left + "-" + right: {col: paired(contrast(left, right, "1", col)) for col in cols}
            for left, right in [("dense_2000", "dense_384"), ("dense_2000", "endpoint_2000"),
                                ("dense_2000", "parent")]
        }
    return out


def blind():
    keys = ("actual_mse", "actual_erasure_gap", "oracle_gap", "same_future_input_teacher_gap")
    result = {}
    for name, run in read("BELIEF.json").items():
        if "blind" not in run:
            continue
        rows = run["blind"]["rows"]
        result[name] = {}
        for period, mask in (("all", slice(None)), ("before_observation", slice(0, 7)),
                             ("different_visible_digit", slice(7, 15)), ("endpoint", slice(15, 16))):
            p = result[name][period] = {
                key: paired([np.mean(row[key][mask]) for row in rows]) for key in keys
            }
            err = p["actual_mse"]["mean"]
            gap = p["actual_erasure_gap"]["mean"]
            p["share_of_tied_mse"] = gap / (err + gap)
            p["student_contrast_gain_along_teacher"] = (
                sum(sum(row["dot"][mask]) for row in rows) /
                sum(sum(row["teacher_energy"][mask]) for row in rows)
            )
    return result


if __name__ == "__main__":
    summary = {"native": native(), "blind": blind()}
    (HERE / "SUMMARY.json").write_text(json.dumps(summary, indent=2) + "\n")
