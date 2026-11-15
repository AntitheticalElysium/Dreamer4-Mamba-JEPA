"""F6.1 dissection, check 2 (PLAN.md): waking on time on sleeps where WAKE_UP has already paid. The quiet sleep bank's
own construction and generation (`../20261107_danger_adversarial/sleep.py`: asleep at energy 8, health >= 7, food and
drink >= 3, no arrow and no zombie or skeleton within 6 cells, t >= 17, one scene per episode, NOOP, 8 simulator and 8
generated draws x 48 frames), restricted to scenes where WAKE_UP had already been earned, with 48 scenes in place of
32; read by F6.1's sleep_split (`../20261114_p1f61/stages.py`: phase test, awake by step, early / late wakes, wait
ratio) for F6.1 and F6 at checkpoint 10.

    JAX_PLATFORMS=cpu python -m artifacts.experiments.20261115_f61_dissection.sleep_paid build        (CPU, simulator)
    MAMBA_DETERMINISTIC=1 python -m artifacts.experiments.20261115_f61_dissection.sleep_paid run      (GPU, both worlds)
    python -m artifacts.experiments.20261115_f61_dissection.sleep_paid read                           (CPU)
"""
import importlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

SL = importlib.import_module("artifacts.experiments.20261107_danger_adversarial.sleep")
AN = importlib.import_module("artifacts.experiments.20261107_danger_adversarial.analyse")
ST = importlib.import_module("artifacts.experiments.20261114_p1f61.stages")
HERE = Path(__file__).resolve().parent
OUT = HERE / "run/sleep_paid"
CKPTS = {"F6.1": "artifacts/experiments/20261114_p1f61/run/ckpt_10.pt", "F6": "artifacts/experiments/20261113_p1f6/run/ckpt_10.pt"}
N = 48


def choose_paid(L):
    """sleep.choose with one more condition: WAKE_UP earned by the source state."""
    T = SL.CV.transitions(L); g = T['g']; pos = L['s_pos'][g].astype(int)
    good = T['asleep'] & (T['body'][:, 3] == 8) & (T['body'][:, 0] >= 7) & (T['body'][:, 1] >= 3) & (T['body'][:, 2] >= 3) & (T['t'] >= 17)
    good &= ~L['s_arrows_m'][g].any(1)
    for k in ('zombies', 'skeletons'):
        near = L[f's_{k}_m'][g] & (abs(L[f's_{k}_p'][g].astype(int) - pos[:, None]).sum(-1) <= 6)
        good &= ~near.any(1)
    good &= L['s_ach'][g, 15].astype(bool)
    rng = np.random.default_rng(110711); eps = np.unique(T['ep'][good]); rows = []
    for e in rng.permutation(eps)[:N]:
        j = int(rng.choice(np.flatnonzero(good & (T['ep'] == e))))
        rows.append(dict(e=int(e), t0=int(T['t'][j] - 1), j=j, shard=int(L['ep_shard'][e]), slot=int(L['ep_slot'][e])))
    print("eligible episodes", len(eps), "scenes", len(rows), flush=True)
    return rows, len(eps)


def read():
    L = dict(np.load(SL.CV.LABELS / "s0_dev_labels.npz"))
    off = np.concatenate([[0], np.cumsum(L["ep_T"] + 1)])
    res = {}
    for name in CKPTS:
        z = torch.load(OUT / name / "run.pt", map_location="cpu", weights_only=False)
        assert all(L["s_ach"][off[r["e"]] + r["t0"], 15] for r in z["rows"])
        a, b, h = z["generated"], z["truth"], z["H"]
        active = np.concatenate([np.ones((*b.shape[:2], 1), bool), np.cumprod(~b[:, :, :-1, 3].astype(bool), 2).astype(bool)], 2)
        gen = AN.first(a[..., 3].astype(bool) & (a[..., 1] < .5) & ~a[..., 2].astype(bool), h)
        sim = AN.first(active & (b[..., 1] < .5) & ~b[..., 3].astype(bool), h)
        boot = np.random.default_rng(110711).integers(0, len(gen), (2000, len(gen)))
        gw, sw = np.minimum(gen, h).mean(1), np.minimum(sim, h).mean(1)
        res[name] = {"scenes": len(gen), "checkpoint_sha256": z["checkpoint_sha256"],
                     "awake_by_step": {str(k): {"generated": AN.interval(gen <= k), "simulator": AN.interval(sim <= k)} for k in (8, 12, 16, 32, 48)},
                     "early_wake": AN.interval(gen < sim.min(1, keepdims=True)),
                     "late_or_no_wake": AN.interval(gen > sim.max(1, keepdims=True)),
                     "restricted_wait_ratio": {"mean": float(gw.mean() / sw.mean()),
                                               "ci95": np.percentile(gw[boot].mean(1) / sw[boot].mean(1), [2.5, 97.5]).tolist()},
                     "phase_test": ST.phase_test(gen, sim, h)}
        print(name, json.dumps({k: res[name][k] for k in ("restricted_wait_ratio", "phase_test")}), flush=True)
    (HERE / "sleep_paid.json").write_text(json.dumps(res, indent=1) + "\n")


if __name__ == "__main__":
    mode = sys.argv[1]
    OUT.mkdir(parents=True, exist_ok=True)
    if mode == "build":
        SL.choose, SL.N, SL.HERE = choose_paid, N, OUT
        SL.build()
    elif mode == "run":
        for name, ck in CKPTS.items():
            (OUT / name).mkdir(exist_ok=True)
            if (OUT / name / "run.pt").exists():
                continue
            (OUT / name / "bank.pt").write_bytes((OUT / "bank.pt").read_bytes())
            SL.HERE, SL.CKPT = OUT / name, ck
            SL.run()
    else:
        read()
