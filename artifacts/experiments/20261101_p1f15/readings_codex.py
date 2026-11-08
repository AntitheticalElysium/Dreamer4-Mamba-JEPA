"""Codex's readings for P1-F1.5 (PLAN.md readings, item 2), through beta-TSP's wrapper (`../20261101_beta_tsp/
readings_codex.py`: Codex's code unchanged, its output directory pointed here). panel / cpu: unchanged (read_panel
samples with M.sample, i.e. the cascade for a factorised world; its fixed-level denoising reads periphery outputs,
which never read the glimpse). sampler: sampler_followup's factorial with its Euler algorithms in cascade form
(M.sample, stage-2 cache included, at each algorithm's steps / churn / churn range / noise scale / sigma_max; Heun
has no cascade form and is not a declared reading), on the same contexts as P1-F's (asserted against beta-TSP's saved
P1-F panels); the factorial's own production check (Euler-3 / churn 1 against M.sample) is kept.
Usage (project venv): python -m artifacts.experiments.20261101_p1f15.readings_codex panel|cpu|sampler CKPT
"""
import copy, importlib, json, sys, time
from pathlib import Path

import numpy as np
import torch

from saccade import model as M, train as TR
from saccade.eval import s1 as S, gates as G

BR = importlib.import_module("artifacts.experiments.20261101_beta_tsp.readings_codex")
HERE = Path(__file__).resolve().parent
BR.OUT = HERE / "readings/codex"


def cascade_sample(den, shapes, prev_action, memory, fix, cache, alg, cfg, gen):
    """sampler_followup's algorithm `alg` in its cascade form: M.sample at the algorithm's schedule."""
    n, heun, churn, tmin, tmax, s_noise, smax = alg
    assert not heun
    c = copy.deepcopy(cfg)
    c["edm"]["sampler"].update(steps=n, s_churn=churn, s_tmin=tmin, s_tmax=tmax, s_noise=s_noise, sigma_max=smax)
    return tuple(t[:, 0] for t in M.sample(den, shapes, prev_action, memory, fix, cache, c, gen))


def sampler(path):
    """BR.sampler's loop for this world alone, the factorial's Euler algorithms in cascade form."""
    R = BR.redirect("sampler_followup.run", ())
    R.ALGS = {k: v for k, v in R.ALGS.items() if not v[1]}
    R.sample = cascade_sample
    torch.set_num_threads(4)
    start = time.time()
    L = S.first_episodes(dict(np.load(S.LABELS / "s0_dev_labels.npz")), 150)
    grids = [d["grid"] for d in G.episode_grids(L)]
    p1f = json.loads((BR.HERE / "readings/codex/sampler_results.json").read_text())   # beta-TSP's, P1-F's panels
    out = {"algorithms": {k: list(v) for k, v in R.ALGS.items()}, "source_sha256": TR.file_sha256(Path(R.__file__)),
           "wrapper_sha256": TR.file_sha256(Path(__file__)), "beta_wrapper_sha256": TR.file_sha256(Path(BR.__file__)), "models": {}}
    ck = torch.load(path, weights_only=False, map_location="cpu"); cfg = ck["cfg"]
    assert cfg["denoiser"].get("factorised") and cfg["edm"]["sampler"]["steps"] == 3 and cfg["edm"]["sampler"]["s_churn"] == 1
    assert cfg["edm"]["sampler"]["s_noise"] == 1 and cfg["edm"]["sampler"]["sigma_max"] == 5
    assert cfg["edm"]["sampler"]["s_tmin"] == 0 and cfg["edm"]["sampler"]["s_tmax"] >= 5
    model = R.M.M0(cfg).to(R.DEV).eval().requires_grad_(False); model.load_state_dict(ck["model"]); del ck
    enc = S.encode(model, L, cfg)
    reader_path = Path(path).parent / "s1/results.reader.pt"
    reader = S.load_reader(reader_path, path, R.DEV)
    rows = [(e, t, int(enc[e]["prev_action"][t])) for e in range(len(enc)) for t in range(8, len(enc[e]["p"]))]
    rng = np.random.default_rng(610370)
    count = [rows[i] for i in rng.choice(len(rows), min(256, len(rows)), replace=False)]
    old = R.CAT.N_MAX; R.CAT.N_MAX = 128
    moves = R.CAT.contexts(L, enc, np.random.default_rng(610340))[0]["move"]; R.CAT.N_MAX = old
    panels = {"counts": count, "move": moves}
    for k, v in panels.items():                                          # P1-F's contexts (beta-TSP's readings)
        assert json.loads(json.dumps(v)) == p1f["models"]["p1f10"]["panels"][k]["contexts"], k
    result, checks = R.evaluate(model, cfg, reader, enc, L, panels, grids)
    out["models"]["p1f15"] = {"checkpoint": str(path), "sha256": TR.file_sha256(path), "reader_sha256": TR.file_sha256(reader_path),
                              "panels": result, "checks": checks}
    out["elapsed_seconds"] = time.time() - start
    (BR.OUT / "sampler_results.json").write_text(json.dumps(out, indent=1))
    print("p1f15 complete", round(out["elapsed_seconds"], 1), flush=True)


def main():
    what, path = sys.argv[1], Path(sys.argv[2])
    BR.WORLDS = {"p1f15": path}
    if what == "sampler":
        return sampler(path)
    sys.argv = [sys.argv[0], what, "p1f15"]
    BR.main()


if __name__ == "__main__":
    main()
