"""P1-F1.5's pre-launch checks (PLAN.md), GPU, MAMBA_DETERMINISTIC=1:
1. The real launcher, 25 updates without checkpoints: every logged term finite; time and peak memory per update.
2. Resume, with TR.p1 itself at a shrunken frame budget (a checkpoint every 2 updates) on P1-F1.5's config: 4 straight
   updates vs 2 + resume + 2, against a second straight run (bitwise expected in one process).
3. Identity: validate_resume refuses P1-F's checkpoint under P1-F1.5's config and a P1-F1.5 checkpoint under the
   config without the factorised key; the launcher refuses a directory whose identity differs.
Usage: MAMBA_DETERMINISTIC=1 python -m artifacts.experiments.20261101_p1f15.checks SCRATCH_DIR
"""
import copy, json, os, shutil, subprocess, sys
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")
import math
import torch

from saccade import train as TR

L = __import__("importlib").import_module("artifacts.experiments.20261101_p1f15.train_p1f15")
HERE = Path(__file__).parent
P1F = TR.ROOT / "artifacts/experiments/20261030_p1_guided/run"


def main():
    assert os.environ.get("MAMBA_DETERMINISTIC") == "1"
    scratch = Path(sys.argv[1])
    res = {}
    out = scratch / "launcher25"
    shutil.rmtree(out, ignore_errors=True)
    subprocess.run([sys.executable, "-m", "artifacts.experiments.20261101_p1f15.train_p1f15", str(out), "--updates", "25",
                    "--no-checkpoints"], check=True, cwd=TR.ROOT)
    log = [json.loads(l) for l in open(out / "log.jsonl")]
    res["launcher"] = {"logged": log, "all_finite": all(math.isfinite(v) for r in log for k, v in r.items() if isinstance(v, float)),
                       "identity": json.loads((out / "identity.json").read_text())}
    print("launcher", json.dumps({k: v for k, v in res["launcher"].items() if k != "identity"}), flush=True)
    cfg = L.recipe(TR.load_cfg())
    small = copy.deepcopy(cfg); small["p1"]["frames"] = 20480
    ck = {}
    for name, plan in (("straight_a", (4,)), ("straight_b", (4,)), ("resumed", (2, 4))):
        d = scratch / name
        shutil.rmtree(d, ignore_errors=True)
        for u in plan:
            TR.p1(small, L.WORLD_SEED, d, u, None, None, True)
        ck[name] = torch.load(d / "ckpt_02.pt", weights_only=False, map_location="cpu")
    diff = lambda a, b: max(float((ck[a]["model"][k].float() - ck[b]["model"][k].float()).abs().max()) for k in ck[a]["model"])
    res["resume"] = {"straight_a_vs_b_max_abs": diff("straight_a", "straight_b"), "straight_a_vs_resumed_max_abs": diff("straight_a", "resumed"),
                     "state_equal": ck["straight_a"]["state"] == ck["resumed"]["state"]}
    joint = copy.deepcopy(small); joint["denoiser"].pop("factorised")
    prov = TR.run_provenance(small)
    for name, (c_, cf) in {"refuses_p1f15_without_factorised": (ck["resumed"], joint),
                           "refuses_p1f_under_p1f15": (torch.load(P1F / "ckpt_01.pt", weights_only=False, map_location="cpu"), cfg)}.items():
        try:
            TR.validate_resume(c_, cf, L.WORLD_SEED, prov); res["resume"][name] = False
        except ValueError as e:
            res["resume"][name] = f"refused: {e}"
    ident = out / "identity.json"
    j = json.loads(ident.read_text()); j["recipe"] = "altered"; ident.write_text(json.dumps(j))
    r = subprocess.run([sys.executable, "-m", "artifacts.experiments.20261101_p1f15.train_p1f15", str(out), "--updates", "1",
                        "--no-checkpoints"], cwd=TR.ROOT, capture_output=True, text=True)
    res["resume"]["launcher_refuses_other_identity"] = r.returncode != 0 and "different identity" in r.stderr
    print("resume", json.dumps(res["resume"]), flush=True)
    (HERE / "checks.json").write_text(json.dumps(res, indent=1, default=str))


if __name__ == "__main__":
    main()
