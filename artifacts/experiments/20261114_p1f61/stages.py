"""F6.1's readings beyond F5's standard queue, each an existing instrument pointed at this run (one stage per process).

  clocks       readout.py at checkpoints 2, 3 and the evaluated one: trajectory, registration (d1), trace.
  trajectory   ../20261111_p1f5/trajectory.py at all ten checkpoints (F5 RESULTS 11.2's table).
  movement, mining, sleep   ../20261111_p1f5/followups.py's stages (their "f5" key is the evaluated checkpoint).
  sleep_split  the sleep stage's draws (32 quiet NOOP sleep scenes x 8, 48 frames) split by whether WAKE_UP had already
               paid at the start (Codex 11-14: 25 never / 7 already), with early and late wakes against the scene's own
               simulator draws.
  entities     ../20261111_p1f5/entities.py with this checkpoint in F5's place, beside F4 and HUD-Change.
  compare      ../20261111_p1f5/compare.py with F6 and F6.1 columns.

    MAMBA_DETERMINISTIC=1 python -m artifacts.experiments.20261114_p1f61.stages STAGE CHECKPOINT READINGS
"""
import importlib
import json
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
F5 = "artifacts.experiments.20261111_p1f5"
F6 = HERE.parent / "20261113_p1f6"


def phase_test(gen, sim, h, perms=10000, seed=110711):
    """Codex 11-14: does the world wake THIS scene at its own time? Per scene, the 1-Wasserstein distance between the
    generated and the simulator wake-time samples (no wake = H + 1); real pairing against generated samples shuffled
    across scenes of the same cohort. p = share of shuffles at least as close as the real pairing."""
    g, s = np.sort(np.minimum(gen, h + 1), 1).astype(float), np.sort(np.minimum(sim, h + 1), 1).astype(float)
    score = lambda gg: float(np.abs(gg - s).mean())
    real = score(g)
    rng = np.random.default_rng(seed)
    null = np.array([score(g[rng.permutation(len(g))]) for _ in range(perms)])
    return {"real_w1": real, "shuffled_w1_mean": float(null.mean()), "shuffled_w1_5th": float(np.percentile(null, 5)),
            "p_shuffle_at_least_as_close": float((1 + (null <= real).sum()) / (1 + perms)), "permutations": perms}


def sleep_split(checkpoint, readings):
    import torch
    from craftax.craftax_classic.constants import Achievement
    from saccade import train as TR
    FU = importlib.import_module(f"{F5}.followups")
    AN = importlib.import_module(f"{FU.SLEEP}.analyse")
    CV = importlib.import_module("artifacts.experiments.20261106_determinant_curves.curves")
    z = torch.load(readings / "followups/sleep/run.pt", map_location="cpu", weights_only=False)
    assert z["checkpoint_sha256"] == TR.file_sha256(checkpoint)
    a, b, h = z["generated"], z["truth"], z["H"]
    active = np.concatenate([np.ones((*b.shape[:2], 1), bool), np.cumprod(~b[:, :, :-1, 3].astype(bool), 2).astype(bool)], 2)
    gen = AN.first(a[..., 3].astype(bool) & (a[..., 1] < .5) & ~a[..., 2].astype(bool), h)   # as followups.sleep's "awake"
    sim = AN.first(active & (b[..., 1] < .5) & ~b[..., 3].astype(bool), h)
    L = dict(np.load(CV.LABELS / "s0_dev_labels.npz"))
    off = np.concatenate([[0], np.cumsum(L["ep_T"] + 1)])
    paid = np.array([bool(L["s_ach"][off[r["e"]] + r["t0"], Achievement.WAKE_UP.value]) for r in z["rows"]])
    res = {"checkpoint_sha256": z["checkpoint_sha256"], "H": int(h), "K": int(z["K"])}
    ix = np.random.default_rng(110711)
    for name, m in (("all", np.ones(len(paid), bool)), ("wake_up_not_yet_paid", ~paid), ("wake_up_already_paid", paid)):
        g, s = gen[m], sim[m]
        boot = ix.integers(0, len(g), (2000, len(g)))
        gw, sw = np.minimum(g, h).mean(1), np.minimum(s, h).mean(1)
        res[name] = {"scenes": int(m.sum()),
                     "awake_by_step": {str(k): {"generated": AN.interval(g <= k), "simulator": AN.interval(s <= k)}
                                       for k in (8, 12, 16, 32, 48)},
                     "early_wake": AN.interval(g < s.min(1, keepdims=True)),
                     "late_or_no_wake": AN.interval(g > s.max(1, keepdims=True)),
                     "restricted_wait_ratio": {"mean": float(gw.mean() / sw.mean()),
                                               "ci95": np.percentile(gw[boot].mean(1) / sw[boot].mean(1), [2.5, 97.5]).tolist()},
                     "phase_test": phase_test(g, s, h)}
    (readings / "followups/sleep_split.json").write_text(json.dumps(res, indent=1) + "\n")
    print("sleep_split", {k: res[k]["restricted_wait_ratio"]["mean"] for k in ("all", "wake_up_not_yet_paid")}, flush=True)


def main(name, checkpoint, readings):
    k = int(checkpoint.stem[-2:])
    assert checkpoint == (HERE / f"run/ckpt_{k:02d}.pt").resolve()
    readings.mkdir(parents=True, exist_ok=True)
    if name == "clocks":
        RD = importlib.import_module(f"{__package__}.readout")
        RD.OUT = readings
        for j in sorted({2, 3, k}):                                      # F5 and F6 hold readings at 2 and 10; 3 is the early read
            RD.main(j, RD.STAGES)
    elif name == "trajectory":
        TJ = importlib.import_module(f"{F5}.trajectory")
        TJ.HERE, TJ.OUT = HERE, readings / "trajectory"
        TJ.OUT.mkdir(parents=True, exist_ok=True)
        for j in range(1, 11):
            if (HERE / f"run/ckpt_{j:02d}.pt").exists():
                TJ.checkpoint(j)
    elif name in ("movement", "mining", "sleep"):
        FU = importlib.import_module(f"{F5}.followups")
        FU.OUT = readings / "followups"
        FU.run_stage(name, checkpoint)
    elif name == "sleep_split":
        sleep_split(checkpoint, readings)
    elif name == "entities":
        EN = importlib.import_module(f"{F5}.entities")
        EN.OUT = readings / "dissection"
        EN.OUT.mkdir(parents=True, exist_ok=True)
        EN.DS.CKPTS = dict(EN.DS.CKPTS, f5=checkpoint)
        EN.main()
        path = EN.OUT / "entities.json"
        res = json.loads(path.read_text())
        res["f61"] = res.pop("f5")
        path.write_text(json.dumps(res, indent=1) + "\n")
    elif name == "compare":
        CP = importlib.import_module(f"{F5}.compare")
        CP.HERE = HERE
        for arm, root in (("f6", F6 / "readings"), ("f61", readings)):
            CP.CURVES[arm] = root / "consequences/curves/raw.npz"
            CP.INTERFACE[arm] = root / "interface.json"
            CP.RECURSION[arm] = root / "recursion_4/raw.npz"
        CP.main()
    else:
        raise SystemExit(f"unknown stage {name}")


if __name__ == "__main__":
    assert os.environ.get("MAMBA_DETERMINISTIC") == "1"
    main(sys.argv[1], Path(sys.argv[2]).resolve(), Path(sys.argv[3]).resolve())
