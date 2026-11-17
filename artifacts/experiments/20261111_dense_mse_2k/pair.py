"""Serial 2,000-update comparison and readout; extending from update 384."""
import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")
import json, subprocess, sys
from pathlib import Path
import numpy as np
from saccade import train as TR
from .train import P, PARENT, atomic_write, BUDGET

MOD = "artifacts.experiments.20261111_dense_mse_2k"
OUT = P / "run"
ROOT = TR.ROOT


def summary():
    q = {}
    rates = {}
    parent = P.parent / "20261111_two_arm_audit/PHASE_hud.json"
    for k, path in [("parent", parent), *((x, P / f"PHASE_{x}.json") for x in ("endpoint", "dense"))]:
        r = json.loads(path.read_text())
        expected = PARENT if k == "parent" else OUT / k / "eval.pt"
        assert r["checkpoint_sha256"] == TR.file_sha256(expected)
        assert len(r["pairs"]) == 24 and r["K"] == 32 and r["reader"]["accuracy"] >= .99
        e = [x["e"] for x in r["pairs"]]
        if k == "parent":
            ids = e
        else:
            assert e == ids
        rates[k] = np.array([x["generated_drop"] for x in r["pairs"]])
        q[k] = {
            "due_not_due": rates[k].mean(0).tolist(),
            "contrast": float(np.diff(rates[k][:, ::-1], axis=1).mean()),
            "reader_holdout": r["reader"]["accuracy"]
        }
    for x, y in (("dense", "endpoint"), ("dense", "parent"), ("endpoint", "parent")):
        d = (rates[x][:, 0] - rates[x][:, 1]) - (rates[y][:, 0] - rates[y][:, 1])
        pick = np.random.default_rng(1112).integers(0, len(d), (5000, len(d)))
        q[x + "_minus_" + y] = {
            "contrast_change": float(d.mean()),
            "paired_episode_ci95": np.quantile(d[pick].mean(1), [.025, .975]).tolist()
        }
    atomic_write(OUT / "SUMMARY.json", q)
    print(json.dumps(q, indent=2), flush=True)


def main():
    assert os.environ.get("MAMBA_DETERMINISTIC") == "1"
    assert json.loads((P / "CHECKS.json").read_text())["passed"]
    pins = json.loads((P / "RUN_PINS.json").read_text())

    def verify_pins():
        for f, h in pins.items():
            if TR.file_sha256(ROOT / f) != h:
                raise RuntimeError(f"Pinned input changed: {f}")

    verify_pins()
    OUT.mkdir(parents=True, exist_ok=True)
    sp = OUT / "STATUS.json"
    state = json.loads(sp.read_text()) if sp.exists() else {}

    def run(name, *cmd):
        if state.get(name) == "complete":
            return
        verify_pins()
        state[name] = "running"
        atomic_write(sp, state)
        try:
            with (OUT / f"{name}.log").open("a") as log:
                subprocess.run([sys.executable, "-u", *cmd], cwd=ROOT, check=True, stdout=log, stderr=subprocess.STDOUT)
        except BaseException:
            state[name] = "failed"
            atomic_write(sp, state)
            raise
        state[name] = "complete"
        atomic_write(sp, state)

    for arm in ("endpoint", "dense"):
        run("train_" + arm, "-m", MOD + ".train", arm, str(OUT / arm), "--steps", str(BUDGET))
    for arm in ("endpoint", "dense"):
        run("phase_" + arm, "-m", MOD + ".readout", arm, str(OUT / arm / "eval.pt"))
    summary()
    run(
        "native_guards", "-m", "artifacts.experiments.20261030_actor_interface.interface", str(OUT / "guards.json"),
        "--episodes=120", "--classes=risky,adjacent,ach_COLLECT_COAL,ach_COLLECT_IRON", "--policy=random",
        f"parent={PARENT}", f"endpoint={OUT / 'endpoint/eval.pt'}", f"dense={OUT / 'dense/eval.pt'}"
    )
    state["complete"] = True
    atomic_write(sp, state)


if __name__ == "__main__":
    main()
