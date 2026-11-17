"""F6.1 dissection, check 17 (PLAN.md): on-schedule exact-history pairs. dissect.build_clock with the due branch keeping
the recorded state (its drop falls where the unedited episode drops) and the delayed branch's counter lowered by 8;
read like check 16 (drawn rate, first-step expectation) on F6.1 and the recency fine-tune, beside the old pairs.

    JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= python -m artifacts.experiments.20261115_f61_dissection.schedule_pairs build
    MAMBA_DETERMINISTIC=1 python -m artifacts.experiments.20261115_f61_dissection.schedule_pairs read
"""
import importlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
RUN = HERE / "run/schedule"
RF = importlib.import_module("artifacts.experiments.20261115_f61_dissection.recency_ft")
TF, UC = RF.TF, RF.UC
DS, D, M, C, TR, TJ = TF.DS, TF.D, TF.M, TF.C, TF.TR, TF.TJ
SPEC = {"energy": dict(field="player_fatigue", col=3, slot=66, seed=111302), "hunger": dict(field="player_hunger", col=1, slot=64, seed=111303)}
STEPS, SEARCH, MARGIN, SHIFT, N_PAIRS = 64, (16, 62), 66, 8, 32


def build(name):
    """dissect.build_clock(name) with values (recorded, recorded - SHIFT), STEPS / SEARCH / MARGIN widened to reach the
    second drop; the same episode order, selection, keys and checks."""
    import jax, jax.numpy as jnp
    assert jax.default_backend() == "cpu"
    k, select = SPEC[name], DS.CLOCKS["energy"]["select"]
    L = dict(np.load(D.ROOT / "artifacts/eda/s0/s0_dev_labels.npz"))
    off = np.r_[0, np.cumsum(L["ep_T"] + 1)]
    _, _, _, step, render = D.RP.env_and_render()
    bank, excluded = [], 0
    order = [e for e in range(len(L["ep_T"])) if e % 5 == 4] + [e for e in range(len(L["ep_T"])) if e % 5 != 4]
    for e in order:
        T = int(L["ep_T"][e])
        lo, hi = off[e:e + 2]
        b = L["s_body"][lo:hi]
        ts = np.arange(T + 1)
        ok = (ts >= 12) & (ts <= T - MARGIN) & select(b) & ~L["s_sleep"][lo:hi] & (L["s_light"][lo:hi] >= .9)
        idx = np.flatnonzero(ok)
        if not len(idx):
            continue
        t = int(idx[0])
        sh, sl = int(L["ep_shard"][e]), int(L["ep_slot"][e])
        state = D.RP.advance_to(sh, sl, t)
        orig = D.RP.episode_fields(sh, sl)
        assert np.array_equal(np.asarray(render(state)), orig["observations"][t].numpy())
        v0 = float(getattr(state, k["field"]))
        dtype = getattr(state, k["field"]).dtype
        states = [state, state.replace(**{k["field"]: jnp.asarray(v0 - SHIFT, dtype)})]
        frames, rewards, term, body = ([[], []] for _ in range(4))
        keys = jax.random.split(jax.random.PRNGKey(k["seed"] + e), STEPS + 3)
        for s in range(STEPS):
            for br in range(2):
                st = states[br]
                frames[br].append(np.asarray(render(st)))
                body[br].append([int(st.player_health), int(st.player_food), int(st.player_drink), int(st.player_energy)])
                _, nxt, r, d, _ = step(keys[s], st, jnp.asarray(0, jnp.int32))
                states[br] = nxt
                rewards[br].append(float(r))
                term[br].append(bool(d))
        frames, rewards, term, body = map(np.asarray, (frames, rewards, term, body))
        digit, health = body[..., k["col"]], body[..., 0]
        found = []
        for s in range(*SEARCH):
            if (np.sign(digit[:, s + 1] - digit[:, s]) == -1).tolist() != [True, False]:
                continue
            if not np.array_equal(frames[0, s - 7:s + 1], frames[1, s - 7:s + 1]):
                continue
            if not np.array_equal(rewards[0, s - 8:s], rewards[1, s - 8:s]) or not np.array_equal(term[0, s - 8:s], term[1, s - 8:s]):
                continue
            if term[:, :s + 1].any() or (np.diff(health[:, :s + 2], axis=1) < 0).any():
                continue
            events = [np.flatnonzero(np.diff(digit[j, :s + 1]) < 0) + 1 for j in (0, 1)]
            if not all(len(x) for x in events):
                continue
            found.append((s, [int(x[-1]) for x in events]))
        if not found:
            excluded += 1
            continue
        s, resets = found[0]
        bank.append({"e": e, "t0": t, "end": s, "frames": frames[:, :s + 2], "reward": rewards[:, :s + 1], "term": term[:, :s + 1],
                     "digit": digit[:, :s + 2], "last_event": resets, "values": [v0, v0 - SHIFT], "due_episode_step": t + s + 1,
                     "prefix_frames": orig["observations"][:t].numpy(), "prefix_actions": orig["actions_taken"][:t].numpy(),
                     "prefix_reward": orig["rewards"][:t].numpy(), "prefix_term": orig["terminated"][:t].numpy()})
        print("pair", len(bank), "episode", e, "t0", t, "step", s, "due at episode step", t + s + 1, "last events", resets,
              "digit", digit[:, s:s + 2].tolist(), "values", [v0, v0 - SHIFT], flush=True)
        if len(bank) >= N_PAIRS:
            break
    RUN.mkdir(parents=True, exist_ok=True)
    torch.save(bank, RUN / f"{name}_bank.pt")
    (RUN / f"{name}_build.json").write_text(json.dumps({"pairs": len(bank), "excluded": excluded, "rows": [
        {k_: r[k_] for k_ in ("e", "t0", "end", "last_event", "values", "due_episode_step")} for r in bank]}, indent=1) + "\n")


def banks():
    """(name, rows in traces_ft.pairs' format, slot, sign): the old energy / hunger pairs and the on-schedule ones."""
    old = {n: (n, rows, slot, sign) for n, rows, slot, sign in TF.pairs() if n in ("energy", "hunger")}
    out = [old["energy"], old["hunger"]]
    for name in ("energy", "hunger"):
        bank = torch.load(RUN / f"{name}_bank.pt", map_location="cpu", weights_only=False)
        rows = [((r["prefix_frames"], r["prefix_actions"], r["prefix_reward"], r["prefix_term"]), r["frames"], r["reward"], r["term"],
                 r["end"], np.zeros((2, r["end"] + 1), np.int64), r["digit"][:, r["end"]:r["end"] + 2]) for r in bank]
        out.append((name + "_on_schedule", rows, SPEC[name]["slot"], -1))
    return out


@torch.no_grad()
def read_bank(model, cfg, rows, slot, sign, reader):
    """use_check.expect's per-pair reading (traces_ft.pair_read's setup and seeds; the physical stage of model.sample)."""
    den, dev = model.denoiser, DS.DEV
    cells = []
    for i, (prefix, frames, rewards, terms, q, acts, truth) in enumerate(rows):
        old_f, old_a, old_r, old_d = prefix
        n = len(old_f)
        f = np.concatenate([np.repeat(old_f[None], 2, 0), frames[:, :q + 1]], 1)
        a = np.concatenate([np.repeat(old_a[None], 2, 0), acts], 1)
        rw = np.column_stack([np.repeat(np.r_[0., old_r][None], 2, 0), rewards[:, :q]])
        td = np.column_stack([np.repeat(np.r_[False, old_d][None], 2, 0), terms[:, :q]])
        fix = torch.as_tensor(np.repeat(D.fixation_prior(np.random.default_rng(12000 + i), (len(f[0]),))[None], 2, 0), device=dev)
        query = torch.as_tensor(D.fixation_prior(np.random.default_rng(15000 + i), ()), device=dev)
        act = torch.as_tensor(a, device=dev)
        prev = torch.cat([act.new_full((2, 1), -1), act[:, :-1]], 1)
        ft = torch.as_tensor(f, device=dev)
        pc = torch.stack([C.coarse(model.perception, ft[b])[0] for b in (0, 1)])
        first = torch.zeros(pc.shape[:2], dtype=torch.bool, device=dev); first[:, 0] = True
        tin, tst = TF.traces(pc, first, torch.zeros(2, TF.N, 2, device=dev))
        cell = {}
        with TR.autocast(dev, cfg):
            p = model.perception(ft, fix)
            y, _ = model.belief(model.belief.tokens(model.belief.percept_embed(p), torch.as_tensor(rw, device=dev, dtype=torch.float32),
                                                    torch.as_tensor(td, device=dev), act), model.belief.initial(2, dev))
            bp = torch.cat([torch.zeros_like(y[:, :1]), y[:, 1::2][:, :-1]], 1)
            nxt = model.perception(torch.as_tensor(frames[0:1, q + 1:q + 2], device=dev), query[None, None])
            last, delta = p.p[0, -1, slot].float(), nxt.p[0, 0, slot].float() - p.p[0, -1, slot].float()
            for m, c in ((0, 0), (1, 1), (0, 1), (1, 0)):
                rng = torch.Generator(device=dev).manual_seed(14000 + i)
                cache = den.new_cache()
                for t in range(n + q - 7, n + q + 1):
                    den.trace_input = tin[c:c + 1, t:t + 1]
                    cache = M.context_pass(den, tuple(z[0:1, t:t + 1] for z in p.parts()), prev[0:1, t:t + 1],
                                           model.belief.memory(bp[m:m + 1, t:t + 1]), fix[0:1, t:t + 1], cache, cfg, rng)
                cache = {"t": cache["t"], "kv": [tuple(z.expand(DS.DRAWS, *z.shape[1:]) for z in kv) for kv in cache["kv"]]}
                den.trace_input = tst[c:c + 1, -1:].expand(DS.DRAWS, -1, -1, -1)
                pz, vz, xs = UC.physical(den, [tuple(x.shape[2:]) for x in p.parts()], act.new_zeros(DS.DRAWS, 1),
                                         model.belief.memory(y[m:m + 1, -1:]).expand(DS.DRAWS, -1, -1, -1),
                                         query[None, None].expand(DS.DRAWS, 1, -1), cache, cfg,
                                         torch.Generator(device=dev).manual_seed(16000 + i))
                digits = reader(torch.cat([pz[:, 0, slot], vz[:, 0]], 1).float())["y"].argmax(-1).cpu().numpy()
                cell[f"m{m}c{c}"] = (float(((digits < truth[0, 0]) if sign < 0 else (digits > truth[0, 0])).mean()),
                                     float(UC.kappa(xs[0][:, 0, slot], last, delta).mean()))
        cells.append(cell)
    r = {k: np.array([c[k][0] for c in cells]) for k in cells[0]}
    k0 = {k: np.array([c[k][1] for c in cells]) for k in cells[0]}
    return {"pairs": len(cells), "rate": {k: float(v.mean()) for k, v in r.items()},
            "first_step": {k: float(v.mean()) for k, v in k0.items()},
            "native_due_minus_delayed": UC.ci(r["m0c0"] - r["m1c1"]),
            "recency_effect": UC.ci((r["m0c0"] + r["m1c0"] - r["m0c1"] - r["m1c1"]) / 2),
            "memory_effect": UC.ci((r["m0c0"] + r["m0c1"] - r["m1c0"] - r["m1c1"]) / 2),
            "first_step_due_minus_delayed": UC.ci(k0["m0c0"] - k0["m1c1"])}


def read():
    TF.traces, TF.install = RF.recency, RF.install
    model, cfg = DS.NP.IF.load(TF.CKPT.resolve())
    for p in model.parameters():
        p.requires_grad_(False)
    TF.install(model.denoiser)
    model.denoiser.trace_proj.requires_grad_(False)                     # use_check.load: no autocast weight cache
    model.eval()
    base = {k: v.clone() for k, v in model.denoiser.state_dict().items()}
    readers = TJ.readers(model, cfg, slots=(64, 66))
    arms = [("f61", base)] + [(f"recency_{u}", path) for u, path in (
        (2000, HERE / "run/recency/denoiser_2000.pt"), (4000, HERE / "run/recency_long/denoiser_4000.pt"),
        (6000, HERE / "run/recency_long/denoiser_6000.pt")) if path.exists()]
    bk = banks()
    res = {"due_episode_step": {}}
    for name, rows, slot, sign in bk:
        res["due_episode_step"][name] = float(np.median([len(r[0][0]) + r[4] + 1 for r in rows]))
    for arm, sd in arms:
        model.denoiser.load_state_dict(sd if isinstance(sd, dict) else torch.load(sd, map_location=DS.DEV, weights_only=False)["denoiser"])
        res[arm] = {}
        for name, rows, slot, sign in bk:
            res[arm][name] = read_bank(model, cfg, rows, slot, sign, readers[slot][0])
            print(arm, name, json.dumps(res[arm][name]), flush=True)
    (HERE / "schedule_pairs.json").write_text(json.dumps(res, indent=1) + "\n")


if __name__ == "__main__":
    if sys.argv[1] == "build":
        for name in ("energy", "hunger"):
            build(name)
    else:
        read()
