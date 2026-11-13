"""Preflight for the token-forecast arm (PLAN.md, preflight). CUDA, bounded, real data.

  init      same seed: every F5 parameter is bitwise F5's (TSP's unused forecast mask aside), the only new parameters
            are the token head's, and the global RNG after construction is F5's.
  f5_path   the F5 recipe under this code reproduces F5's logged first update exactly.
  matched   this arm's first update: every term F5 also has equals F5's logged value (same weights, data, draws).
  known     the token forecast term with the head's output layer zeroed equals copying the previous frame's tokens,
            computed independently from the chunk.
  resume    2 + 2 updates equal 4 straight (model, optimizer, streams, carry, cache, RNG, gaze).
  smoke     12 updates through the launcher's recipe: finite terms, peak memory, seconds per update.

    MAMBA_DETERMINISTIC=1 python -m artifacts.experiments.20261113_p1f6.checks
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
F5 = launch.F5


def rows(path):
    return [r for line in (path / "log.jsonl").read_text().splitlines() if "g1" in (r := json.loads(line))]


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
    last = model.token_forecast.net[-1]
    with torch.no_grad():
        last.weight.zero_(), last.bias.zero_()
    r = np.arange(8)
    carry = model.belief.initial(d["streams"], dev)
    cache = torch.zeros(d["streams"], P + 1, cfg["belief"]["width"], device=dev)
    boot = TR.bootstrap_rows(cfg, launch.WORLD_SEED, 0)
    with torch.no_grad():
        terms, _, _ = TR.p1_losses(model, D.select(chunk, r), M.carry_rows(carry, r), cache[r], None, cfg, boot[r], 0, None)
        t = TR.tensors(D.select(chunk, r), dev)
        with TR.autocast(dev, cfg):
            p = model.perception(t["frames"], t["fixation"]).p.float()
        nf = ~t["first"][:, P:] & t["valid"][:, P:]
        copy_ = (p[:, P:] - p[:, P - 1:-1]).square()[nf].mean()
    got, want = float(terms["token_forecast"]), float(copy_)
    assert "tsp_forecast" not in terms and want > 0 and abs(got - want) <= 1e-7 * max(1., want), (got, want)
    return {"token_forecast_zeroed_head": got, "copy_previous": want, "frames": int(nf.sum())}


def main():
    assert os.environ.get("MAMBA_DETERMINISTIC") == "1"
    torch.set_num_threads(2)
    cfg, f5 = launch.recipe(), F5.check_recipe(F5.recipe())
    checks = launch.HERE / "checks"
    checks.mkdir(exist_ok=True)
    out = Path(tempfile.mkdtemp(dir=checks, prefix="smoke_"))
    print(out, flush=True)
    res = {}
    torch.manual_seed(launch.WORLD_SEED); a = M.M0(f5); ra = torch.get_rng_state()
    torch.manual_seed(launch.WORLD_SEED); b = M.M0(cfg); rb = torch.get_rng_state()
    sa, sb = a.state_dict(), b.state_dict()
    new = sorted(set(sb) - set(sa))
    assert set(sa) - set(sb) == {"tsp.mask"} and all(k.startswith("token_forecast.") for k in new) and new
    assert all(torch.equal(sa[k], sb[k]) for k in sa if k != "tsp.mask") and torch.equal(ra, rb)
    res["init"] = {"shared_tensors_bitwise_equal": len(sa) - 1, "new": new,
                   "token_head_parameters": int(sum(sb[k].numel() for k in new)), "global_rng_equal": True}
    del a, b
    first = rows(F5.HERE / "run")[0]
    assert first["update"] == 1
    same = lambda r, keys: {k: (r[k], first[k]) for k in keys if r[k] != first[k]}
    TR.p1(f5, launch.WORLD_SEED, out / "f5", updates=1, checkpoints=False, log_every=1)
    torch.cuda.empty_cache()
    r = rows(out / "f5")[0]
    keys = [k for k in first if k not in ("sec_per_update", "peak_gib")]
    assert not same(r, keys), same(r, keys)
    res["f5_path"] = {"first_update_terms_equal_f5_log": keys}
    TR.p1(cfg, launch.WORLD_SEED, out / "arm", updates=1, checkpoints=False, log_every=1)
    torch.cuda.empty_cache()
    r = rows(out / "arm")[0]
    keys = [k for k in first if k in r and k not in ("sec_per_update", "peak_gib", "grad_norm", "sigma")]
    assert "token_forecast" in r and "tsp_forecast" not in r and not same(r, keys), same(r, keys)
    res["matched"] = {"first_update_terms_equal_f5_log": keys, "token_forecast": r["token_forecast"],
                      "f5_tsp_forecast": first["tsp_forecast"], "grad_norm": [r["grad_norm"], first["grad_norm"]]}
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
    assert len(log) == 12 and all(math.isfinite(v) for r in log for v in (*r["sigma"], r["grad_norm"], r["g1"],
                                  r["token_forecast"], r["reward_forecast"], r["cont_forecast"]))
    res["smoke"] = {"updates": 12, "peak_gib": torch.cuda.max_memory_allocated() / 2 ** 30,
                    "token_forecast_by_update": [r["token_forecast"] for r in log],
                    "seconds_per_update": sum(r["sec_per_update"] for r in log[6:]) / 6, "last": log[-1]}
    res["passed"] = True
    (launch.HERE / "checks.json").write_text(json.dumps(res, indent=2, allow_nan=False) + "\n")
    print(json.dumps(res), flush=True)


if __name__ == "__main__":
    main()
