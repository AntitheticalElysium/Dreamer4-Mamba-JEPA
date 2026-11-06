"""P1-F3's one-step generation on its own TRAINING histories vs dev at 16 sampler steps (PLAN.md): 11-04's instrument
(../20261104_f2_train_dev/train_dev.py: its subset / compare, the same 1,009 training episodes) with F3 checkpoint 10 and
edm.sampler.steps = 16; the dev side is F3's own 16-step readout. R0: dev COLLECT_COAL recomputed here equals it.
Usage (project venv, GPU): MAMBA_DETERMINISTIC=1 python -m artifacts.experiments.20261106_f3_train_dev.train_dev_steps [--smoke]
"""
import copy, importlib, json, os, sys
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import numpy as np
import torch

from saccade import data as D, train as TR
from saccade.eval import s1 as S

TD = importlib.import_module("artifacts.experiments.20261104_f2_train_dev.train_dev")
IF, DEV, LABELS = TD.IF, TD.DEV, TD.LABELS
HERE = Path(__file__).resolve().parent
CKPT = "artifacts/experiments/20261105_p1f3/run/ckpt_10.pt"
DEV_READING = TR.ROOT / "artifacts/experiments/20261105_p1f3/readings/interface_steps.json"
EPISODES_1104 = TR.ROOT / "artifacts/experiments/20261104_f2_train_dev/readings/train_dev.json"
STEPS = 16


def run(model, cfg, L, cls, only=None):
    enc, out = S.encode(model, L, cfg), {}
    for i, (name, tuples) in enumerate(cls.items()):
        if not tuples or (only and name not in only):
            continue
        o = IF.futures(model, enc, L, tuples, 4, cfg, torch.Generator(device=DEV).manual_seed(3000 + i))
        out[name] = json.loads(json.dumps(IF.summary(o, np.array([tuples[j][0] for j in o["tuple"]])), default=str))
        print(name, "done", flush=True)
    return out


def main():
    assert os.environ.get("MAMBA_DETERMINISTIC") == "1"
    smoke = "--smoke" in sys.argv
    out = HERE / ("smoke" if smoke else "readings"); out.mkdir(parents=True, exist_ok=True)
    model, cfg = IF.load(CKPT)
    cfg = copy.deepcopy(cfg); cfg["edm"]["sampler"]["steps"] = STEPS
    ref = json.loads(DEV_READING.read_text())["checkpoints"]["f3_16steps"]
    assert ref["sha256"] == TR.file_sha256(TR.ROOT / CKPT), "dev reading is another checkpoint"
    res = {"checkpoint": CKPT, "sha256": ref["sha256"], "steps": STEPS, "dev": str(DEV_READING.relative_to(TR.ROOT))}
    if not smoke:                                                        # R0: the dev side reproduces in process
        Ld = S.first_episodes(dict(np.load(LABELS / "s0_dev_labels.npz")), 0)
        r0 = run(model, cfg, Ld, IF.classes(Ld, np.random.default_rng(2031)), only={"ach_COLLECT_COAL"})
        assert r0["ach_COLLECT_COAL"] == ref["classes"]["ach_COLLECT_COAL"], "R0: dev coal does not reproduce"
        res["r0"] = "dev COLLECT_COAL reproduced exactly"
        del Ld
    monitor = {(e.shard, e.slot) for e in D.load_split(cfg)["monitor"]}
    La = dict(np.load(LABELS / "s0_train_labels.npz"))
    ok = np.array([i for i, (s, o) in enumerate(zip(La["ep_shard"], La["ep_slot"])) if (int(s), int(o)) not in monitor])
    n = 20 if smoke else len(np.load(LABELS / "s0_dev_labels.npz")["ep_T"])
    keep = np.sort(np.random.default_rng(TD.SEED).choice(ok, n, replace=False))
    if not smoke:
        assert keep.tolist() == json.loads(EPISODES_1104.read_text())["episodes"], "not 11-04's training episodes"
    L = TD.subset(La, keep)
    cls = IF.classes(L, np.random.default_rng(2031))
    if smoke:
        cls = {k: v[:8] for k, v in cls.items()}
    res.update(train_episodes=len(ok), episodes=keep.tolist(), classes={k: len(v) for k, v in cls.items()})
    res["train"] = run(model, cfg, L, cls)
    res["train_minus_dev"] = TD.compare(res["train"], ref["classes"])
    (out / "train_dev.json").write_text(json.dumps(res, indent=1, default=str))
    for k in ("ach_COLLECT_COAL", "ach_COLLECT_IRON", "risky", "adjacent"):
        c = res["train_minus_dev"].get(k, {}).get("closure", {})
        print(k, {q: c.get(q) for q in ("rho_train", "rho_dev", "closure", "ci95", "verdict")}, flush=True)
    print("saved", out, flush=True)


if __name__ == "__main__":
    main()
