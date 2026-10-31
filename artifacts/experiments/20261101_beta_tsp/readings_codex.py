"""Codex's declared readings for this run (OVERNIGHT_PROPOSAL.md readings 1-4; PLAN.md "Readings"), with Codex's code
unchanged: each tool's module-level output directory is pointed at readings/codex/ (its fixed inputs linked there,
read-only), so nothing in Codex's folders is written. Worlds: P1-F checkpoint 10 (`p1f10`, the control) and this run's
checkpoint 10 (`beta10`); the same panels, contexts and seeds for both.
- panel NAME: read_panel NAME CKPT --generation --movement: the fixed-world re-render noise panel by light (token
  variance under renderer static: the manipulation check), with TSP / reconstruction pixel error, real reward NLL,
  generation decomposition and denoising at three noise levels; movement; counter consequences at the rule's crop and
  at another crop containing the whole counter, DO and NOOP (generated glimpse decoded by the model's own pixel head).
- cpu NAME: lighting.read (illumination sweep) and real_counters (real-crop digit reads at both crops), CPU float32.
- sampler: sampler_followup's factorial, its loop with these two worlds and each world's own S1 reader (P1-F's
  run/s1 reader is the same fit as the own-gaze one Codex used, every tensor equal): entity counts and movement shift;
  every algorithm is computed, Euler-3 / churn 1 is read as primary, Euler-8 / churn 0 and Euler-32 / churn 1 as
  secondary (PLAN.md).
Usage (project venv): python -m artifacts.experiments.20261101_beta_tsp.readings_codex panel|cpu NAME | sampler
"""
import importlib, json, os, sys, time
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")
import numpy as np
import torch

from saccade import train as TR
from saccade.eval import s1 as S, gates as G

EXP = TR.ROOT / "artifacts/experiments"
HERE = Path(__file__).resolve().parent
OUT = HERE / "readings/codex"
FOLLOW = EXP / "20261031_grounding_loss_review/followup"
WORLDS = {"p1f10": EXP / "20261030_p1_guided/run/ckpt_10.pt", "beta10": HERE / "run/ckpt_10.pt"}
READERS = {"p1f10": EXP / "20261030_p1_guided/run/s1/results.reader.pt", "beta10": HERE / "run/s1/results.reader.pt"}
TOOL = "artifacts.experiments.20261031_grounding_loss_review."


def redirect(name, inputs):
    """Import Codex's tool and point its module-level HERE at OUT, with its fixed inputs linked there."""
    OUT.mkdir(parents=True, exist_ok=True)
    for f in inputs:
        if not (OUT / f).exists():
            (OUT / f).symlink_to(FOLLOW / f)
    module = importlib.import_module(TOOL + name)
    module.HERE = OUT
    return module


def sampler():
    """sampler_followup.run.main's loop, unchanged apart from the worlds and their readers."""
    R = redirect("sampler_followup.run", ())
    torch.set_num_threads(4)
    start = time.time()
    L = S.first_episodes(dict(np.load(S.LABELS / "s0_dev_labels.npz")), 150)
    grids = [d["grid"] for d in G.episode_grids(L)]
    out = {"algorithms": {k: list(v) for k, v in R.ALGS.items()}, "source_sha256": TR.file_sha256(Path(R.__file__)),
           "wrapper_sha256": TR.file_sha256(Path(__file__)), "models": {}}
    shared = None
    for name, path in WORLDS.items():
        ck = torch.load(path, weights_only=False, map_location="cpu"); cfg = ck["cfg"]
        assert cfg["edm"]["sampler"]["steps"] == 3 and cfg["edm"]["sampler"]["s_churn"] == 1
        assert cfg["edm"]["sampler"]["s_noise"] == 1 and cfg["edm"]["sampler"]["sigma_max"] == 5
        assert cfg["edm"]["sampler"]["s_tmin"] == 0 and cfg["edm"]["sampler"]["s_tmax"] >= 5
        model = R.M.M0(cfg).to(R.DEV).eval().requires_grad_(False); model.load_state_dict(ck["model"]); del ck
        enc = S.encode(model, L, cfg)
        reader = S.load_reader(READERS[name], path, R.DEV)
        rows = [(e, t, int(enc[e]["prev_action"][t])) for e in range(len(enc)) for t in range(8, len(enc[e]["p"]))]
        rng = np.random.default_rng(610370)
        count = [rows[i] for i in rng.choice(len(rows), min(256, len(rows)), replace=False)]
        old = R.CAT.N_MAX; R.CAT.N_MAX = 128
        moves = R.CAT.contexts(L, enc, np.random.default_rng(610340))[0]["move"]; R.CAT.N_MAX = old
        panels = {"counts": count, "move": moves}
        if shared is None: shared = panels
        else: assert panels == shared
        result, checks = R.evaluate(model, cfg, reader, enc, L, panels, grids)
        out["models"][name] = {"checkpoint": str(path), "sha256": TR.file_sha256(path),
                               "reader_sha256": TR.file_sha256(READERS[name]), "panels": result, "checks": checks}
        out["elapsed_seconds"] = time.time() - start
        (OUT / "sampler_results.json").write_text(json.dumps(out, indent=1))
        print(name, "complete", round(out["elapsed_seconds"], 1), flush=True)
        del model, enc, reader
        torch.cuda.empty_cache()


def main():
    what = sys.argv[1]
    if what == "sampler":
        return sampler()
    name = sys.argv[2]; path = WORLDS[name]
    if what == "panel":
        RP = redirect("followup.read_panel", ("panel.json", "panel.npz"))
        sys.argv = [RP.__file__, name, str(path), "--generation", "--movement"]
        RP.main()
    elif what == "cpu":
        torch.set_num_threads(4)
        redirect("followup.lighting", ("lighting.npz",)).read(name, path)
        redirect("followup.real_counters", ()).main(name, path)
    else:
        raise SystemExit(__doc__)


if __name__ == "__main__":
    main()
