"""Preflight for P1-F6.1 (PLAN.md, preflight). CUDA, bounded, real data.

  init      same seed: every F5 parameter is bitwise F5's (TSP's forecast mask aside, dropped with tsp: false), the only
            new parameters are the reward forecast's seven output layers, and the global RNG after construction is F5's.
  f5_path   the F5 recipe under this code reproduces F5's logged first update exactly.
  f6_path   the F6 recipe under this code reproduces F6's logged first update exactly (token forecast path untouched).
  matched   this arm's first update: every term F5 also has equals F5's logged value; the distance-1 reward forecast
            equals F5's reward forecast; no visual forecast term.
  known     each distance d = 2..8 reads exactly the belief predicted before frame t - d + 1 and frame t's reward, on
            exactly the pairs with no episode start in [t - d + 1, t] (recomputed from episode ids, not from first
            flags); every distance's term reaches Mamba.
  resume    2 + 2 updates equal 4 straight (model, optimizer, streams, carry, cache, RNG, gaze).
  smoke     12 updates through the launcher's recipe: finite terms, peak memory, seconds per update.

    MAMBA_DETERMINISTIC=1 python -m artifacts.experiments.20261114_p1f61.checks
"""
import copy
import importlib
import json
import math
import os
from pathlib import Path
import tempfile

import numpy as np
import torch

from saccade import chooser as C, data as D, model as M, train as TR
from . import train as launch

F2 = importlib.import_module("artifacts.experiments.20261103_p1f2.checks")
F5, F6 = launch.F5, importlib.import_module("artifacts.experiments.20261113_p1f6.train")
EXP = launch.HERE.parent


def rows(path):
    return [r for line in (path / "log.jsonl").read_text().splitlines() if "g1" in (r := json.loads(line))]


def first_update_matches(cfg, run, out):
    """Train one logged update under cfg and compare every logged term with the archived run's first row."""
    first = rows(run)[0]
    assert first["update"] == 1
    TR.p1(cfg, launch.WORLD_SEED, out, updates=1, checkpoints=False, log_every=1)
    torch.cuda.empty_cache()
    r = rows(out)[0]
    keys = [k for k in first if k not in ("sec_per_update", "peak_gib")]
    diff = {k: (r[k], first[k]) for k in keys if r[k] != first[k]}
    assert not diff, diff
    return keys


def known_answer(cfg):
    dev = torch.device("cuda")
    torch.manual_seed(launch.WORLD_SEED)
    split = D.load_split(cfg)
    d, P = cfg["data"], cfg["data"]["context_prefix"]
    streams = D.Streams(split["train"], d["streams"], P, seed=launch.WORLD_SEED, targets_offsets=D.frame_offsets(split["train"]))
    model = M.M0(cfg).to(dev)
    TR.set_reward_values(model, split["train"])
    out = Path(tempfile.mkdtemp(dir=launch.HERE / "checks", prefix="known_"))
    controller = C.Controller(model, cfg, out, dev)
    chunk = streams.next(64)
    controller.prepare(chunk, streams)
    r_ = np.arange(4)                                                    # 4 streams: the graph is kept for the reach test
    carry = model.belief.initial(d["streams"], dev)
    cache = torch.zeros(d["streams"], P + 1, cfg["belief"]["width"], device=dev)
    boot = TR.bootstrap_rows(cfg, launch.WORLD_SEED, 0)
    sub = D.select(chunk, r_)
    full, seen = [], {}
    features, ahead = model.outcome.features, model.outcome.ahead_logp

    def record_features(belief, embed):
        h = features(belief, embed)
        if embed is None and belief.shape[1] == sub["first"].shape[1]:   # the every-frame call (distances >= 2)
            full.append(h)
        return h

    def record_ahead(h, r, dist):
        seen[dist] = (h, r)
        return ahead(h, r, dist)

    model.outcome.features, model.outcome.ahead_logp = record_features, record_ahead
    terms, _, _ = TR.p1_losses(model, sub, M.carry_rows(carry, r_), cache[r_], None, cfg, boot[r_], 0, None)
    model.outcome.features, model.outcome.ahead_logp = features, ahead
    t = TR.tensors(sub, dev)
    ep, first, valid, rew = (t[k].cpu().numpy() for k in ("episode", "first", "valid", "reward"))
    assert len(full) == 1 and "tsp_forecast" not in terms and "token_forecast" not in terms
    ha = full[0]
    res, belief_params = {}, [p for p in model.belief.parameters() if p.requires_grad]
    for dist in range(2, cfg["heads"]["forecast"]["reward_distances"] + 1):
        pairs = [(i, j, j - dist + 1) for i in range(ep.shape[0]) for j in range(P, ep.shape[1])
                 if valid[i, j] and not first[i, j] and valid[i, j - dist + 1] and not first[i, j - dist + 1]
                 and (ep[i, j - dist + 1:j + 1] == ep[i, j]).all() and not first[i, j - dist + 2:j + 1].any()]
        h, r = seen[dist]
        want_h = torch.stack([ha[i, s] for i, j, s in pairs])
        want_r = torch.stack([t["reward"][i, j] for i, j, s in pairs])
        assert h.shape[0] == len(pairs) and torch.equal(h, want_h) and torch.equal(r, want_r), dist
        with TR.autocast(dev, cfg):
            want = float(-ahead(want_h, want_r, dist).mean())
        got = float(terms[f"reward_forecast_{dist}"])
        assert abs(got - want) <= 1e-6 * max(1., abs(want)), (dist, got, want)
        g = torch.autograd.grad(terms[f"reward_forecast_{dist}"], belief_params, retain_graph=True, allow_unused=True)
        reach = float(sum(x.float().square().sum() for x in g if x is not None).sqrt())
        assert reach > 0, dist
        res[dist] = {"pairs": len(pairs), "excluded_episode_start": int(sum(
            bool(valid[i, j] and not first[i, j]) for i in range(ep.shape[0]) for j in range(P, ep.shape[1]))) - len(pairs),
            "term": got, "belief_grad_norm": reach}
    total = terms["reward_forecast_1"]
    for k in res:                                                        # the trainer's order of addition
        total = total + terms[f"reward_forecast_{k}"]
    assert torch.equal(total, terms["reward_forecast"]), "the grouped reward forecast is not the unweighted sum"
    return res


def main():
    assert os.environ.get("MAMBA_DETERMINISTIC") == "1"
    torch.set_num_threads(2)
    cfg, f5, f6 = launch.recipe(), F5.check_recipe(F5.recipe()), F6.check_recipe(F6.recipe())
    checks = launch.HERE / "checks"
    checks.mkdir(exist_ok=True)
    out = Path(tempfile.mkdtemp(dir=checks, prefix="smoke_"))
    print(out, flush=True)
    res = {}
    torch.manual_seed(launch.WORLD_SEED); a = M.M0(f5); ra = torch.get_rng_state()
    torch.manual_seed(launch.WORLD_SEED); b = M.M0(cfg); rb = torch.get_rng_state()
    sa, sb = a.state_dict(), b.state_dict()
    new = sorted(set(sb) - set(sa))
    assert set(sa) - set(sb) == {"tsp.mask"} and new and all(k.startswith("outcome.ahead.") for k in new), new
    assert all(torch.equal(sa[k], sb[k]) for k in sa if k != "tsp.mask") and torch.equal(ra, rb)
    res["init"] = {"shared_tensors_bitwise_equal": len(sa) - 1, "new": new,
                   "new_parameters": int(sum(sb[k].numel() for k in new)), "global_rng_equal": True}
    del a, b
    res["f5_path"] = {"first_update_terms_equal_f5_log": first_update_matches(f5, F5.HERE / "run", out / "f5")}
    res["f6_path"] = {"first_update_terms_equal_f6_log": first_update_matches(f6, F6.HERE / "run", out / "f6")}
    first = rows(F5.HERE / "run")[0]
    TR.p1(cfg, launch.WORLD_SEED, out / "arm", updates=1, checkpoints=False, log_every=1)
    torch.cuda.empty_cache()
    r = rows(out / "arm")[0]
    keys = [k for k in first if k in r and k not in ("sec_per_update", "peak_gib", "grad_norm", "sigma", "reward_forecast")]
    diff = {k: (r[k], first[k]) for k in keys if r[k] != first[k]}
    assert "tsp_forecast" not in r and "token_forecast" not in r and not diff, diff
    assert r["reward_forecast_1"] == first["reward_forecast"], (r["reward_forecast_1"], first["reward_forecast"])
    res["matched"] = {"first_update_terms_equal_f5_log": keys, "distance_1_equals_f5_reward_forecast": r["reward_forecast_1"],
                      "by_distance": [r[f"reward_forecast_{k}"] for k in range(1, 9)], "reward_forecast_sum": r["reward_forecast"],
                      "grad_norm": [r["grad_norm"], first["grad_norm"]]}
    res["known"] = known_answer(cfg)
    torch.cuda.empty_cache()
    small = copy.deepcopy(cfg)
    small["p1"].update(frames=20480, checkpoint_every_fraction=.05)
    small["p1"]["monitor"]["windows"] = 4
    small["edm"]["shortcut"]["bootstrap_start"] = 2
    for name, updates in (("straight", (4,)), ("resumed", (2, 4))):
        for update in updates:
            TR.p1(small, launch.WORLD_SEED, out / name, updates=update, log_every=1)
            torch.cuda.empty_cache()
    x, y = (torch.load(out / name / "latest.pt", map_location="cpu", weights_only=False) for name in ("straight", "resumed"))
    exact = {k: bool(F2.equal(x[k], y[k])) for k in ("streams", "carry", "cache", "state", "rng", "fixation_controller")}
    exact["gaze_bytes"] = (out / "straight/gaze.bin").read_bytes() == (out / "resumed/gaze.bin").read_bytes()
    model_diff = max(float((x["model"][k] - y["model"][k]).abs().max()) for k in x["model"])
    opt_diff = F2.optimizer_error(x["opt"], y["opt"])
    assert all(exact.values()) and model_diff < 1e-11 and opt_diff < 1e-11, (exact, model_diff, opt_diff)
    res["resume"] = {"exact": exact, "model_max_abs": model_diff, "optimizer_max_abs": opt_diff}
    del x, y
    torch.cuda.reset_peak_memory_stats()
    TR.p1(cfg, launch.WORLD_SEED, out / "early64", updates=12, checkpoints=False, log_every=1)
    log = rows(out / "early64")
    names = ["g1", "reward_forecast", "cont_forecast", *(f"reward_forecast_{k}" for k in range(1, 9))]
    assert len(log) == 12 and all(math.isfinite(v) for r in log for v in (*r["sigma"], r["grad_norm"], *(r[n] for n in names)))
    res["smoke"] = {"updates": 12, "peak_gib": torch.cuda.max_memory_allocated() / 2 ** 30,
                    "reward_forecast_by_update": [r["reward_forecast"] for r in log],
                    "seconds_per_update": sum(r["sec_per_update"] for r in log[6:]) / 6, "last": log[-1]}
    res["passed"] = True
    (launch.HERE / "checks.json").write_text(json.dumps(res, indent=2, allow_nan=False) + "\n")
    print(json.dumps(res), flush=True)


if __name__ == "__main__":
    main()
