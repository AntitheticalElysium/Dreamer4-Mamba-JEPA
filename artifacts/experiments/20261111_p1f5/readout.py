"""Checkpoint-bound, resumable F5 evaluation queue."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

from saccade import train as TR
from .train_p1f5 import HERE

PREFIX = "artifacts.experiments"
DANGER = f"{PREFIX}.20261107_f4_followups"
HUD = f"{PREFIX}.20261109_p1_hud_change"


def steps(checkpoint, readings, smoke):
    ck = checkpoint.resolve()
    ck_arg = str(ck)
    # The established tools use the checkpoint-relative S1 reader path.
    s1 = ck.parent / "s1/results.json"
    native = readings / "native_pairs.json"
    scores = readings / "forecast_scores.json"
    consequence = readings / "consequences"
    interface = readings / "interface.json"
    recursion = readings / "recursion_4"
    thirst = readings / "natural_thirst.json"
    belief = readings / "belief_probe.json"
    path = readings / "belief_path_probe.json"
    smoke_flag = ["--smoke"] if smoke else []
    rows = [
        ("s1", ["-m", "saccade.eval.s1", ck_arg, str(s1), *(["60", "40"] if smoke else [])], s1),
        ("native_pairs", ["-m", f"{__package__}.native_pairs", ck_arg, str(native), *smoke_flag], native),
        ("forecast_scores", ["-m", f"{__package__}.forecast_scores", ck_arg, str(scores),
                             *(["--episodes", "4"] if smoke else [])], scores),
        ("consequences", ["-m", f"{HUD}.consequences", ck_arg, str(consequence), *smoke_flag],
         consequence / ("crafting_smoke" if smoke else "crafting") / "raw.npz"),
        ("interface", ["-m", f"{PREFIX}.20261030_actor_interface.interface", str(interface),
                       "--policy=random", *(["--episodes=40", "--classes=random,adjacent,risky"] if smoke else []),
                       f"f5_4steps={ck_arg}"], interface),
        ("recursion_4", ["-m", f"{PREFIX}.20261105_consequence_trace.recursion_steps", "--ckpt", ck_arg,
                         "--out", str(recursion), "--steps", "4", *(["--windows", "4"] if smoke else [])], recursion / "identity.json"),
        ("natural_thirst", ["-m", f"{HUD}.natural_pair", ck_arg, str(thirst)], thirst),
        ("belief_probe", ["-m", f"{DANGER}.belief_probe", ck_arg, str(belief)], belief),
        ("belief_path_probe", ["-m", f"{DANGER}.belief_path_probe", ck_arg, str(path)], path),
    ]
    if not smoke:
        for name, folder in (("f4", "20261106_p1f4"), ("hud", "20261109_p1_hud_change")):
            control = HERE.parent / folder / "run/ckpt_10.pt"
            out = readings / f"native_pairs_{name}.json"
            rows.append((f"native_pairs_{name}", ["-m", f"{__package__}.native_pairs", str(control), str(out)], out))
    return rows


def write(path, data):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2) + "\n")
    tmp.replace(path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("checkpoint", type=Path)
    p.add_argument("--out", type=Path, default=HERE / "readings")
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()
    assert os.environ.get("MAMBA_DETERMINISTIC") == "1"
    if args.smoke and args.out == HERE / "readings":
        p.error("provide a separate --out for smoke results")
    status = args.out / "STATUS.json"
    logs = args.out / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    done = json.loads(status.read_text()) if status.exists() else {}
    identity = {"checkpoint_sha256": TR.file_sha256(args.checkpoint), "smoke": args.smoke}
    if done.setdefault("identity", identity) != identity:
        raise RuntimeError("Evaluation results belong to another checkpoint or mode")
    incomplete = []
    for name, command, expected in steps(args.checkpoint, args.out, args.smoke):
        # The last two unchanged, full-cohort reader scripts are covered separately by the existing probe suite.
        if args.smoke and name in ("belief_probe", "belief_path_probe"):
            continue
        if done.get(name) == "complete" and expected.is_file():
            continue
        if name in ("consequences", "recursion_4") and done.get("s1") != "complete":
            done[name] = "blocked_by_s1"
            incomplete.append(name)
            write(status, done)
            continue
        done[name] = "running"
        write(status, done)
        print(name, "running", flush=True)
        try:
            with (logs / f"{name}.log").open("a") as log:
                subprocess.run([sys.executable, "-u", *command], cwd=TR.ROOT, stdout=log,
                               stderr=subprocess.STDOUT, check=True)
            if not expected.is_file():
                raise RuntimeError(f"Missing expected result {expected}")
        except (OSError, RuntimeError, subprocess.CalledProcessError) as error:
            done[name] = "failed"
            incomplete.append(name)
            write(status, done)
            print(name, str(error), file=sys.stderr, flush=True)
            continue
        done[name] = "complete"
        write(status, done)
    if incomplete:
        raise RuntimeError("Incomplete evaluation stages: " + ", ".join(incomplete))


if __name__ == "__main__":
    main()
