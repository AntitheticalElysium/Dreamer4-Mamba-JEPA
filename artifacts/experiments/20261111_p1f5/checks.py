"""Bounded real-data CUDA smoke for the two chunk phases and forecast/resume contract."""
import copy
import importlib
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile

import torch

from saccade import train as TR
from . import train_p1f5 as launch

F2 = importlib.import_module("artifacts.experiments.20261103_p1f2.checks")


def rows(path):
    return [r for line in (path / "log.jsonl").read_text().splitlines() if "g1" in (r := json.loads(line))]


def main():
    torch.set_num_threads(2)
    cfg = launch.check_recipe(launch.recipe())
    checks = launch.HERE / "checks"
    checks.mkdir(exist_ok=True)
    out = Path(tempfile.mkdtemp(dir=checks, prefix="smoke_"))
    print(out, flush=True)
    small = copy.deepcopy(cfg)
    small["p1"].update(frames=20480, checkpoint_every_fraction=.05)
    small["p1"]["monitor"]["windows"] = 4
    small["edm"]["shortcut"]["bootstrap_start"] = 2
    for name, updates in (("straight", (4,)), ("resumed", (2, 4))):
        for update in updates:
            TR.p1(small, launch.WORLD_SEED, out / name, updates=update, log_every=1)
            torch.cuda.empty_cache()
    a, b = (torch.load(out / name / "latest.pt", map_location="cpu", weights_only=False)
            for name in ("straight", "resumed"))
    exact = {k: bool(F2.equal(a[k], b[k])) for k in ("streams", "carry", "cache", "state", "rng", "fixation_controller")}
    exact["gaze_bytes"] = (out / "straight/gaze.bin").read_bytes() == (out / "resumed/gaze.bin").read_bytes()
    terms = lambda d: [{k: v for k, v in r.items() if k not in ("sec_per_update", "peak_gib")} for r in rows(d)]
    exact["logged_terms"] = terms(out / "straight") == terms(out / "resumed")
    model_diff = max(float((a["model"][k] - b["model"][k]).abs().max()) for k in a["model"])
    opt_diff = F2.optimizer_error(a["opt"], b["opt"])
    assert all(exact.values()) and model_diff < 1e-11 and opt_diff < 1e-11, (exact, model_diff, opt_diff)
    del a, b
    subprocess.run([sys.executable, "-u", "-m", f"{__package__}.train_p1f5", str(out / "launcher"),
                    "--updates", "2", "--no-checkpoints"], cwd=TR.ROOT, check=True)
    measurements = {}
    for name, steps, bootstrap_start, updates in (("early64", 64, 5000, 12), ("late64", 64, 0, 12),
                                                   ("late256", 256, 0, 6)):
        c = copy.deepcopy(cfg)
        c["edm"]["shortcut"]["bootstrap_start"] = bootstrap_start
        torch.cuda.reset_peak_memory_stats()
        TR.p1(c, launch.WORLD_SEED, out / name, updates=updates, steps=steps, checkpoints=False, log_every=1)
        log = rows(out / name)
        assert len(log) == updates
        assert all(math.isfinite(x) for r in log for x in (*r["sigma"], r["grad_norm"], r["g1"],
                   r["tsp_forecast"], r["reward_forecast"], r["cont_forecast"]))
        peak = torch.cuda.max_memory_allocated() / 2**30
        assert peak < torch.cuda.get_device_properties(0).total_memory / 2**30
        measurements[name] = {"updates": updates, "peak_gib": peak, "last": log[-1],
                              "seconds_per_update": sum(r["sec_per_update"] for r in log[updates//2:])/(updates-updates//2)}
        torch.cuda.empty_cache()
    n64, n256 = (cfg["p1"]["frames"] // (2 * cfg["data"]["streams"] * s) for s in (64, 256))
    start = cfg["edm"]["shortcut"]["bootstrap_start"]
    secs = (start * measurements["early64"]["seconds_per_update"] +
            (n64 - start) * measurements["late64"]["seconds_per_update"] +
            n256 * measurements["late256"]["seconds_per_update"])
    result = dict(passed=True, smoke_checkpoint=str(out / "straight/latest.pt"), exact_resume=exact,
                  model_max_abs=model_diff, optimizer_max_abs=opt_diff, measurements=measurements,
                  estimated_training_hours=secs / 3600)
    (launch.HERE / "checks.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
