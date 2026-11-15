"""F6.1 dissection, check 10 (PLAN.md): does the denoiser's rise follow memory on clean prefixes, and does its cache hold
it back? Asleep exact-history pairs (F5's clock-pair builder, `../20261111_p1f5/dissect.py` build_clock, with an asleep
start: fatigue set to two values 2 steps apart, NOOP; the last 8 frames, actions and outcomes identical; the previous
rise 9-10 steps back in one branch, 11-12 in the other, so one branch's rise is due at the next frame). On frozen F6.1,
one generated frame per cell with common random numbers (F5's dissect.generate, the cache built over the 8 identical
frames): cached memory from branch c, current memory from branch m, the cache's context signal at the trained level
(7) or stressed (4); the energy digit read by the HUD reader.

    JAX_PLATFORMS=cpu python -m artifacts.experiments.20261115_f61_dissection.swap build      (CPU, simulator)
    MAMBA_DETERMINISTIC=1 python -m artifacts.experiments.20261115_f61_dissection.swap run    (GPU)
"""
import copy
import importlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

from saccade import data as D, model as M, train as TR

DS = importlib.import_module("artifacts.experiments.20261111_p1f5.dissect")
TJ = importlib.import_module("artifacts.experiments.20261111_p1f5.trajectory")
HERE = Path(__file__).resolve().parent
BANK = HERE / "run/swap/onset_bank.pt"                                  # check 10c; the fatigue-edit bank: asleep_bank.pt
CKPT = (HERE.parent / "20261114_p1f61/run/ckpt_10.pt").resolve()
SPEC = dict(actions=([6] + [0] * 39, [0, 0, 6] + [0] * 37), col=3, sign=+1, steps=40, search=(14, 36), seed=111515,
            select=lambda b: (b[:, 3] <= 6) & (b[:, 0] >= 5) & (b[:, 1] >= 3) & (b[:, 2] >= 3), margin=45)
# Check 10c: natural histories. A mid-sleep fatigue edit (check 10) leaves the sleep onset shared, and in natural play the
# phase is the steps since onset (mod 11), so the belief rightly saw both branches as one scene (check 10b). Here one
# branch takes SLEEP at step 0, the other at step 2: onsets and every rise 2 steps apart, both physically consistent.
CELLS = [(c, m, s) for s in (7, 4) for c in (0, 1) for m in (0, 1)]       # branch 0 is due


def build(n=32):
    """dissect.build_clock's loop with an asleep start (it selects awake, bright starts)."""
    import jax, jax.numpy as jnp
    assert jax.default_backend() == "cpu"
    k = SPEC
    L = dict(np.load(D.ROOT / "artifacts/eda/s0/s0_dev_labels.npz"))
    off = np.r_[0, np.cumsum(L["ep_T"] + 1)]
    _, _, _, step, render = D.RP.env_and_render()
    bank = []
    order = [e for e in range(len(L["ep_T"])) if e % 5 == 4] + [e for e in range(len(L["ep_T"])) if e % 5 != 4]
    for e in order:
        T = int(L["ep_T"][e])
        lo, hi = off[e:e + 2]
        b = L["s_body"][lo:hi]
        ts = np.arange(T + 1)
        idx = np.flatnonzero((ts >= 12) & (ts <= T - k["margin"]) & k["select"](b) & ~L["s_sleep"][lo:hi].astype(bool))
        if not len(idx):
            continue
        t = int(idx[0])
        sh, sl = int(L["ep_shard"][e]), int(L["ep_slot"][e])
        state = D.RP.advance_to(sh, sl, t)
        orig = D.RP.episode_fields(sh, sl)
        assert np.array_equal(np.asarray(render(state)), orig["observations"][t].numpy())
        states = [state, state]
        frames, rewards, term, body, asleep = ([[], []] for _ in range(5))
        keys = jax.random.split(jax.random.PRNGKey(k["seed"] + e), 50)
        for s in range(k["steps"]):
            for br in range(2):
                st = states[br]
                frames[br].append(np.asarray(render(st)))
                body[br].append([int(st.player_health), int(st.player_food), int(st.player_drink), int(st.player_energy)])
                asleep[br].append(bool(st.is_sleeping))
                _, nxt, r, d, _ = step(keys[s], st, jnp.asarray(k["actions"][br][s], jnp.int32))
                states[br] = nxt
                rewards[br].append(float(r))
                term[br].append(bool(d))
        frames, rewards, term, body, asleep = map(np.asarray, (frames, rewards, term, body, asleep))
        digit, health = body[..., k["col"]], body[..., 0]
        found = None
        for s in range(*k["search"]):
            if (np.sign(digit[:, s + 1] - digit[:, s]) == k["sign"]).tolist() != [True, False]:
                continue
            if not asleep[:, 3:s + 2].all() or not np.array_equal(frames[0, s - 7:s + 1], frames[1, s - 7:s + 1]):
                continue
            if not np.array_equal(rewards[0, s - 8:s], rewards[1, s - 8:s]) or not np.array_equal(term[0, s - 8:s], term[1, s - 8:s]):
                continue
            if term[:, :s + 1].any() or (np.diff(health[:, :s + 2], axis=1) < 0).any():
                continue
            events = [np.flatnonzero(np.sign(np.diff(digit[j, :s + 1])) == k["sign"]) + 1 for j in (0, 1)]
            if all(len(x) for x in events):
                found = (s, [int(x[-1]) for x in events])
                break
        if found is None:
            continue
        s, resets = found
        bank.append({"e": e, "t0": t, "end": s, "frames": frames[:, :s + 2], "reward": rewards[:, :s + 1], "term": term[:, :s + 1],
                     "digit": digit[:, :s + 2], "col": k["col"], "last_event": resets, "actions": np.asarray(k["actions"])[:, :s + 1],
                     "prefix_frames": orig["observations"][:t].numpy(), "prefix_actions": orig["actions_taken"][:t].numpy(),
                     "prefix_reward": orig["rewards"][:t].numpy(), "prefix_term": orig["terminated"][:t].numpy()})
        print("pair", len(bank), "episode", e, "t0", t, "step", s, "last rises", resets, "digit", digit[:, s:s + 2].tolist(), flush=True)
        if len(bank) >= n:
            break
    BANK.parent.mkdir(parents=True, exist_ok=True)
    torch.save(bank, BANK)


@torch.no_grad()
def cell_rises(model, cfg, row, i, reader):
    """dissect.generate (one frame, K draws, its seeds) with the cached and the current memory taken separately."""
    q, prefix, frames, rewards, terms = row["end"], (row["prefix_frames"], row["prefix_actions"], row["prefix_reward"], row["prefix_term"]), row["frames"], row["reward"], row["term"]
    old_f, old_a, old_r, old_d = prefix
    n = len(old_f)
    f = np.concatenate([np.repeat(old_f[None], 2, 0), frames[:, :q + 1]], 1)
    a = np.concatenate([np.repeat(old_a[None], 2, 0), row["actions"][:, :q + 1]], 1)    # each branch's own actions
    rw = np.column_stack([np.repeat(np.r_[0., old_r][None], 2, 0), rewards[:, :q]])
    td = np.column_stack([np.repeat(np.r_[False, old_d][None], 2, 0), terms[:, :q]])
    dev = DS.DEV
    fix = torch.as_tensor(np.repeat(D.fixation_prior(np.random.default_rng(12000 + i), (len(f[0]),))[None], 2, 0), device=dev)
    query = torch.as_tensor(D.fixation_prior(np.random.default_rng(15000 + i), ()), device=dev)
    act = torch.as_tensor(a, device=dev)
    assert torch.equal(act[0, -9:], act[1, -9:])
    prev = torch.cat([act.new_full((2, 1), -1), act[:, :-1]], 1)
    out = {}
    with TR.autocast(dev, cfg):
        p = model.perception(torch.as_tensor(f, device=dev), fix)
        y, _ = model.belief(model.belief.tokens(model.belief.percept_embed(p), torch.as_tensor(rw, device=dev, dtype=torch.float32),
                                                torch.as_tensor(td, device=dev), act), model.belief.initial(2, dev))
        bp = torch.cat([torch.zeros_like(y[:, :1]), y[:, 1::2][:, :-1]], 1)
        assert all(torch.equal(z[0, -8:], z[1, -8:]) for z in p.parts())
        for c, m, sig in CELLS:
            cf = copy.deepcopy(cfg)
            cf["edm"]["shortcut"]["context_signal"] = sig
            rng = torch.Generator(device=dev).manual_seed(14000 + i)
            cache = model.denoiser.new_cache()
            for t in range(n + q - 7, n + q + 1):
                cache = M.context_pass(model.denoiser, tuple(z[0:1, t:t + 1] for z in p.parts()), prev[0:1, t:t + 1],
                                       model.belief.memory(bp[c:c + 1, t:t + 1]), fix[0:1, t:t + 1], cache, cf, rng)
            cache = {"t": cache["t"], "kv": [tuple(z.expand(DS.DRAWS, *z.shape[1:]) for z in kv) for kv in cache["kv"]]}
            z = M.sample(model.denoiser, [tuple(x.shape[2:]) for x in p.parts()], act.new_zeros(DS.DRAWS, 1),
                         model.belief.memory(y[m:m + 1, -1:]).expand(DS.DRAWS, -1, -1, -1),
                         query[None, None].expand(DS.DRAWS, 1, -1), cache, cf, torch.Generator(device=dev).manual_seed(16000 + i))
            digits = reader(torch.cat([z[0][:, 0, 66], z[2][:, 0]], 1).float())["y"].argmax(-1).cpu().numpy()
            out[f"c{c}m{m}s{sig}"] = float((digits > row["digit"][0, q]).mean())
    return out


def run():
    model, cfg = DS.NP.IF.load(CKPT)
    assert cfg["edm"]["shortcut"]["context_signal"] == 7
    reader = TJ.readers(model, cfg, slots=(66,))[66][0]
    bank = torch.load(BANK, map_location="cpu", weights_only=False)
    rows = [cell_rises(model, cfg, r, i, reader) for i, r in enumerate(bank)]
    keys = list(rows[0])
    rate = {k: np.array([r[k] for r in rows]) for k in keys}
    ix = np.random.default_rng(1111).integers(len(rows), size=(10000, len(rows)))
    ci = lambda d: [float(d.mean()), *np.quantile(d[ix].mean(1), [.025, .975]).tolist()]
    res = {"pairs": len(rows), "draws": DS.DRAWS, "checkpoint_sha256": TR.file_sha256(CKPT), "rate": {k: float(v.mean()) for k, v in rate.items()}}
    for s in (7, 4):
        g = lambda c, m: rate[f"c{c}m{m}s{s}"]
        res[f"signal_{s}"] = {"native_due_minus_delayed": ci(g(0, 0) - g(1, 1)),
                              "current_memory_effect": ci((g(0, 0) + g(1, 0) - g(0, 1) - g(1, 1)) / 2),
                              "cached_memory_effect": ci((g(0, 0) + g(0, 1) - g(1, 0) - g(1, 1)) / 2)}
    res["stress_effect_on_due"] = ci(rate["c0m0s4"] - rate["c0m0s7"])
    res["per_pair"] = rows
    (HERE / "swap.json").write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps({k: v for k, v in res.items() if k != "per_pair"}, indent=1), flush=True)


if __name__ == "__main__":
    {"build": build, "run": run}[sys.argv[1]]()
