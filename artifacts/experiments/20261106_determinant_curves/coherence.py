"""Instrument check on the determinant curves' surprising wake reading (~50% of futures from asleep histories read
awake, truth 1-2%): are the generated "awake" frames coherent, or is the asleep reader swayed by one cue? Three
single-cue asleep readers (the player cell's token; the lighting latent; the mean map token), fitted like the main one
(curves.fit_readers' protocol: train400, s1.fit, seed 0), read the same generated frames as the main reader:
asleep e8-stale histories, regenerated through the same captured interface path (seed 4300; not the curves' draws).
Writes readings/coherence.json.
Usage (project venv, GPU): MAMBA_DETERMINISTIC=1 python -m artifacts.experiments.20261106_determinant_curves.coherence
"""
import copy, json, os

import numpy as np
import torch
import torch.nn.functional as F

from saccade.eval import s1 as S

from . import curves as CV


def main():
    assert os.environ.get("MAMBA_DETERMINISTIC") == "1"
    model, cfg = CV.IF.load(CV.CKPT)
    cfg = copy.deepcopy(cfg); cfg["edm"]["sampler"]["steps"] = CV.STEPS
    L = dict(np.load(CV.LABELS / "s0_dev_labels.npz"))
    enc = S.encode(model, L, cfg)
    R, valid = CV.fit_readers(model, cfg, L, enc, 0)
    Lt = dict(np.load(CV.LABELS / "s0_train400_labels.npz"))
    et = S.encode(model, Lt, cfg)
    p, v = torch.cat([e["p"] for e in et]).float(), torch.cat([e["v"] for e in et]).float()
    cues = {"player": lambda p, v: p[:, CV.PLAYER], "light": lambda p, v: v, "map": lambda p, v: p[:, :63].mean(1)}
    y = torch.as_tensor(Lt["s_sleep"]).long()
    held = torch.as_tensor(np.repeat(np.arange(len(Lt["ep_T"])), Lt["ep_T"] + 1) % 10 == 9)
    C = {}
    for name, f in cues.items():
        x = f(p, v)
        xt, yt, xv, yv = (t.to(CV.DEV) for t in (x[~held], y[~held], x[held], y[held]))
        C[name], _ = S.fit(lambda idx, val: (xv if val else xt)[idx], lambda o, idx, val: F.cross_entropy(o["y"], (yv if val else yt)[idx]),
                           len(yt), len(yv), {"y": 2}, 0)
    T = CV.transitions(L)
    lab = CV.strata(T)
    rows = np.sort(np.random.default_rng(4301).choice(np.nonzero(lab["asleep"] == "e8_stale")[0], 300, replace=False))
    tuples = [(int(T["ep"][j]), int(T["t"][j])) for j in rows]
    with CV.captured() as got:
        o = CV.IF.futures(model, enc, L, tuples, CV.K, cfg, torch.Generator(device=CV.DEV).manual_seed(4300))
    gp, gv = (torch.cat(x) for x in zip(*got))
    j = rows[o["tuple"]]
    rp = torch.stack([enc[T["ep"][x]]["p"][T["t"][x]] for x in j]).float()
    rv = torch.stack([enc[T["ep"][x]]["v"][T["t"][x]] for x in j]).float()
    s1 = S.load_reader(CV.S1_READER, CV.TR.ROOT / CV.CKPT, CV.DEV)
    out = {"histories": len(rows), "futures": len(j)}
    with torch.no_grad():
        for w, (pp, vv) in (("generated", (gp, gv)), ("recorded", (rp, rv))):
            main = CV.read(R, s1, pp, vv, np.full(len(pp), 0))["asleep"] >= 0.5
            cue = {k: (C[k](f(pp, vv).to(CV.DEV))["y"].softmax(1)[:, 1] >= 0.5).cpu().numpy() for k, f in cues.items()}
            if w == "generated":
                out["head_unlock_given_awake_main"] = float(o["gen_unlock_p"][~main].mean())
            out[w] = {"asleep_main": float(main.mean()), **{f"asleep_{k}": float(c.mean()) for k, c in cue.items()},
                      **{f"agree_main_{k}": float((c == main).mean()) for k, c in cue.items()},
                      "all_three_cues_agree": float(((cue["player"] == cue["light"]) & (cue["light"] == cue["map"])).mean())}
    (CV.HERE / "readings/coherence.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1), flush=True)


if __name__ == "__main__":
    main()
