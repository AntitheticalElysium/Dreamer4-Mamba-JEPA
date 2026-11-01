"""Where does P1-F's one-step C2ST regression sit (RESULTS.md: 0.761 -> 0.960)? User 10-31: run now, Codex reviews post
hoc. Per world (P1 world 1 ckpt 10, P1-F ckpt 10): S1.3's own path and seeds (saccade.eval.s1.encode with the fixed
random-prior fixations, one_step on rng 0, conditioning_features, c2st on rng 1), so `all_outcomes/all` reproduces
S1.3's c2st_generated; then the same classifier with the full conditioning and
- ONE outcome group at a time (all tuples): periphery map cells (0-62), periphery HUD cells (63-80), glimpse,
  lighting, reward + termination;
- all outcome groups on the tuples whose deciding state is in a guided-rule context (`attempt`: the frames P1-F was
  always trained with the rule's glimpse) and on the rest (`non_attempt`), each fitted within its own subset (same
  rows for both worlds, so the attempt subset's smaller size limits both worlds alike).
Reading (declared before the run; descriptive, where not why): the regression sits where P1-F's accuracy exceeds
world 1's. In periphery map / lighting on non-attempt tuples: the encoder's grounding balance (the G2 harmoniser
coupling) is the suspect. Confined to attempt tuples / glimpse: guidance is. Both can hold; neither is attributed.
Usage (project venv, GPU): python -m artifacts.experiments.20261030_p1_guided.localize_c2st OUT_JSON [--smoke]
"""
import importlib, json, os, sys
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")
import numpy as np
import torch

from saccade import model as M, train as TR
from saccade.eval import s1 as S

PM = importlib.import_module("artifacts.experiments.20261030_fovea_placement.placement_map")
HERE = Path(__file__).resolve().parent
WORLDS = {"w1": TR.ROOT / "artifacts/saccade_m0/p1_world1/ckpt_10.pt", "p1f": HERE / "run/ckpt_10.pt"}


def main():
    out_path = Path(sys.argv[1])
    L = S.first_episodes(dict(np.load(S.LABELS / "s0_dev_labels.npz")), 20 if "--smoke" in sys.argv else 0)
    rule = [int(k) for k in json.loads((PM.HERE / "map.json").read_text())["rule"]]
    ids, s_off = PM.context_ids(L), np.r_[0, np.cumsum(L["ep_T"] + 1)]
    half = lambda x: x.reshape(len(x), -1).half().numpy()
    res = {}
    for name, path in WORLDS.items():
        ck = torch.load(path, weights_only=False, map_location="cpu")
        cfg = ck["cfg"]
        model = M.M0(cfg).to(S.DEV); model.load_state_dict(ck["model"]); model.eval()
        res[name] = {"checkpoint": str(path), "sha256": TR.file_sha256(path), "c2st": {}}
        del ck
        enc = S.encode(model, L, cfg)
        n_tuples = min(30000, int(sum(len(r["p"]) - 1 for r in enc)))
        eps_, ts, gen = S.one_step(model, enc, cfg, n_tuples, np.random.default_rng(0))
        del model
        torch.cuda.empty_cache()
        real = {k: torch.stack([enc[e][k][t].float() for e, t in zip(eps_, ts)]) for k in ("p", "g", "v")}
        real["r"] = torch.tensor([float(enc[e]["reward"][t]) for e, t in zip(eps_, ts)])
        real["term"] = torch.tensor([bool(enc[e]["terminated"][t]) for e, t in zip(eps_, ts)])
        context = S.conditioning_features(enc, eps_, ts, cfg["data"]["context_prefix"])
        width = cfg["belief"]["width"]
        blocks = [context.shape[1] - width - 17 - 49, width, 17, 49]
        attempt = np.isin(ids[s_off[eps_] + ts - 1], rule)
        res[name]["attempt_share"] = float(attempt.mean())
        Xr, Xg = S.outcome_features(real), S.outcome_features(gen)
        fits = [(f"all_outcomes/{s}", m, Xr, Xg) for s, m in (("all", np.ones(len(eps_), bool)), ("attempt", attempt),
                                                               ("non_attempt", ~attempt))]
        for g, f in (("periphery_map", lambda d: d["p"][:, :63]), ("periphery_hud", lambda d: d["p"][:, 63:]),
                     ("glimpse", lambda d: d["g"]), ("lighting", lambda d: d["v"]),
                     ("reward_termination", lambda d: torch.stack([M.SYMLOG(d["r"]), d["term"].float()], 1))):
            fits.append((f"{g}/all", np.ones(len(eps_), bool), half(f(real)), half(f(gen))))
        for key, m, xr, xg in fits:
            r = S.c2st(context[m], xr[m], xg[m], eps_[m], np.random.default_rng(1), context_blocks=blocks)
            res[name]["c2st"][key] = {k: r[k] for k in ("accuracy", "n_test", "outcome_features", "heldout_loss_by_epoch")}
            print(name, key, round(r["accuracy"], 4), r["n_test"], flush=True)
            out_path.write_text(json.dumps(res, indent=1))
        del enc, context, Xr, Xg, real, gen
    print("saved", out_path, flush=True)


if __name__ == "__main__":
    main()
