"""F2's one-step generation on its own TRAINING histories vs dev (PLAN.md): the actor interface (classes, futures,
summaries, K = 4, its seeds) for F2 checkpoint 10 under the fixed random evaluation gaze, on 1,009 training episodes
(monitor excluded) sampled with a fixed seed; the dev side is the archived F2 random reading of the same instrument.
Requires MAMBA_DETERMINISTIC=1 (the archived reading's mode).
Usage (project venv, GPU): MAMBA_DETERMINISTIC=1 python -m artifacts.experiments.20261104_f2_train_dev.train_dev [--smoke]
"""
import importlib, json, os, sys
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import numpy as np
import torch

from saccade import data as D, train as TR
from saccade.eval import s1 as S

OG = importlib.import_module("artifacts.experiments.20261104_f2_own_gaze.own_gaze")
IF, F2, ARCHIVE, DEV = OG.IF, OG.F2, OG.ARCHIVE, OG.DEV
HERE = Path(__file__).resolve().parent
LABELS = TR.ROOT / "artifacts/eda/s0"
SEED = 612403
METRIC = {"risky": ("term_p", "term_p"), "adjacent": ("neg_p", "neg_p")}         # else ("unlock_p", "event")


def subset(L, keep):
    """The labels of episodes `keep` (sorted), in first_episodes' layout."""
    s_off, t_off = np.r_[0, np.cumsum(L["ep_T"] + 1)], np.r_[0, np.cumsum(L["ep_T"])]
    s = np.concatenate([np.arange(s_off[e], s_off[e + 1]) for e in keep])
    t = np.concatenate([np.arange(t_off[e], t_off[e + 1]) for e in keep])
    return {k: v[keep] if k.startswith("ep_") else v[s] if k.startswith("s_") else v[t] for k, v in L.items()}


def compare(train, dev):
    """Per class: train - dev of the generated, real-successor-head and recorded values (unpaired, standard errors
    combined), and the declared gap closure of the generated relative recall with its verdict (PLAN.md)."""
    out = {}
    for name in train:
        if name not in dev:
            continue
        a, b, row = train[name], dev[name], {}
        for w, q in [(w, q) for w in ("gen", "rec_head") for q in ("unlock_p", "neg_p", "term_p")] + \
                    [("recorded", q) for q in ("event", "neg_p", "term_p")]:
            d, se = a[w][q]["est"] - b[w][q]["est"], float(np.hypot(a[w][q]["se"], b[w][q]["se"]))
            row[f"{w}_{q}"] = {"train": a[w][q]["est"], "dev": b[w][q]["est"], "difference": d, "ci95": [d - 1.96 * se, d + 1.96 * se]}
        m, r = METRIC.get(name, ("unlock_p", "event"))
        g, rt, rd = row[f"gen_{m}"], a["recorded"][r]["est"], b["recorded"][r]["est"]
        if not rt or not rd or g["dev"] >= rd:                           # no recorded event, or no dev gap to close
            row["closure"] = {"metric": m, "recorded_train": rt, "recorded_dev": rd, "verdict": "no_gap" if rd else "no_events"}
            out[name] = row
            continue
        rho_d = g["dev"] / rd
        close = lambda x: (x / rt - rho_d) / (1 - rho_d)                 # x: a generated value on the training side
        lo, hi = (close(g["dev"] + c) for c in g["ci95"])
        c = close(g["train"])
        row["closure"] = {"metric": m, "recorded_train": rt, "recorded_dev": rd, "rho_train": g["train"] / rt, "rho_dev": rho_d,
                          "closure": c, "ci95": [lo, hi],
                          "verdict": "memorised" if c >= 1 / 3 and g["ci95"][0] > 0 else "not_memorised" if hi < 1 / 3 else "unresolved"}
        out[name] = row
    return out


def main():
    assert os.environ.get("MAMBA_DETERMINISTIC") == "1"
    smoke = "--smoke" in sys.argv
    out = HERE / ("smoke" if smoke else "readings"); out.mkdir(parents=True, exist_ok=True)
    model, cfg = IF.load(F2)
    monitor = {(e.shard, e.slot) for e in D.load_split(cfg)["monitor"]}
    La = dict(np.load(LABELS / "s0_train_labels.npz"))
    ok = np.array([i for i, (s, o) in enumerate(zip(La["ep_shard"], La["ep_slot"])) if (int(s), int(o)) not in monitor])
    n = 20 if smoke else len(np.load(LABELS / "s0_dev_labels.npz")["ep_T"])
    keep = np.sort(np.random.default_rng(SEED).choice(ok, n, replace=False))
    L = subset(La, keep)
    cls = IF.classes(L, np.random.default_rng(2031))
    if smoke:
        cls = {k: v[:8] for k, v in cls.items()}
    res = {"checkpoint": str(F2), "sha256": TR.file_sha256(F2), "labels_sha256": TR.file_sha256(LABELS / "s0_train_labels.npz"),
           "train_episodes": len(ok), "monitor_excluded": len(La["ep_T"]) - len(ok), "episodes": keep.tolist(),
           "classes": {k: len(v) for k, v in cls.items()}, "dev": str(ARCHIVE), "train": {}}
    enc = S.encode(model, L, cfg)
    for i, (name, tuples) in enumerate(cls.items()):
        if not tuples:
            continue
        o = IF.futures(model, enc, L, tuples, 4, cfg, torch.Generator(device=DEV).manual_seed(3000 + i))
        res["train"][name] = IF.summary(o, np.array([tuples[j][0] for j in o["tuple"]]))
        print(name, "done", flush=True)
    res["train_minus_dev"] = compare(res["train"], json.loads(ARCHIVE.read_text())["checkpoints"]["p1f2"]["classes"])
    (out / "train_dev.json").write_text(json.dumps(res, indent=1, default=str))
    print("saved", out, flush=True)


if __name__ == "__main__":
    main()
