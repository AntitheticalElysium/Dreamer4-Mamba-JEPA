"""Attach the requested one-shot S1 to the current P1 invocation; no trainer restart or GPU allocation."""
import hashlib
import json
import os
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[4]


def ctl(*args):
    return subprocess.run(["systemctl", "--user", *args], check=True, capture_output=True, text=True).stdout


def properties(unit, *names):
    return dict(line.split("=", 1) for line in ctl("show", unit, *[f"-p{x}" for x in names]).splitlines() if "=" in line)


if __name__ == "__main__":
    parent, child = "saccade-p1-world1.service", "saccade-s1-world1.service"
    before = properties(parent, "ActiveState", "MainPID", "InvocationID", "OnSuccess")
    if before["ActiveState"] != "active" or before["OnSuccess"] not in ("", child):
        raise RuntimeError("Expected the current running P1 without a different completion hook")
    run = ROOT / "artifacts/saccade_m0/p1_world1"
    if (run / "s1/selection.json").exists():
        raise RuntimeError("A previous S1 attempt already exists")
    (run / "s1").mkdir(exist_ok=True)
    digest = hashlib.sha256((ROOT / "saccade/eval/s1.py").read_bytes()).hexdigest()
    unit = run / child
    unit.write_text(f"""[Unit]
Description=One-shot full S1 after P1 world seed 1
After={parent}

[Service]
Type=exec
WorkingDirectory={ROOT}
Environment=OMP_NUM_THREADS=6 MKL_NUM_THREADS=6
Environment=SACCADE_P1_UNIT={parent} SACCADE_P1_INVOCATION={before['InvocationID']}
Environment=SACCADE_S1_SHA256={digest}
MemoryMax=24G
Restart=no
ExecStart={ROOT}/.venv/bin/python -u -m saccade.eval.after_p1 {run} 1
""")
    ctl("link", "--runtime", str(unit))
    runtime = Path(os.environ["XDG_RUNTIME_DIR"]) / "systemd/user"
    dropdir = runtime / (parent + ".d")
    dropdir.mkdir(parents=True, exist_ok=True)
    dropin = dropdir / "50-s1.conf"
    dropin.write_text(f"[Unit]\nOnSuccess={child}\n")
    ctl("daemon-reload")
    after = properties(parent, "ActiveState", "MainPID", "InvocationID", "OnSuccess")
    assert before["MainPID"] == after["MainPID"] and before["InvocationID"] == after["InvocationID"]
    assert after["OnSuccess"] == child
    evaluator = properties(child, "LoadState", "ActiveState", "Restart", "MemoryMax", "Environment", "ExecStart")
    assert evaluator["LoadState"] == "loaded" and evaluator["ActiveState"] == "inactive"
    report = {"parent": after, "evaluator": evaluator, "unit_file": str(unit), "dropin": str(dropin),
              "evaluation_source_sha256": digest, "results": str(run / "s1/results.json"),
              "reader": str(run / "s1/results.reader.pt"), "log": str(run / "s1/eval.log")}
    (Path(__file__).resolve().parent / "installation.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
