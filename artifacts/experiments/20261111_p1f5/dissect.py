"""F5 dissection (RESULTS.md sections 3, 4, 6, 7): three readings on trained checkpoints, no training.

registration  Is a digit change visible in the belief's input? Corrected M4 protocol (20261108_denoiser_teaching_continuation
              visibility: 40k DEV transitions, seed 2120, |delta percept_embed| logistic, 5 episode folds, AP) on F4 (reproduction
              control: archived health .99865 food .26796 drink .19100 energy .06281), HUD-Change and F5. S1 encode.
swap          Cooldown pairs (native_pairs): is F5's due/cooling contrast carried by the belief or by the visual cache, which
              holds the earlier hit? Cache percepts from member v, belief stream (cache memory tokens + generation memory)
              from member b, all four (v, b); same draws.
recovery      Exact-history recovery pairs (built like 20261108_full_frame_change/phase_pairs.py with player_recover 20 vs
              12): identical last 8 frames, rewards and terminations; the earlier rise (reward +0.1) is 18 vs 26 steps back.
              Native generation, due vs not-due rise rates, on F5, HUD-Change (belief lacks the clock: control) and F4.

    python -m artifacts.experiments.20261111_p1f5.dissect build_recovery        (CPU)
    MAMBA_DETERMINISTIC=1 python -m artifacts.experiments.20261111_p1f5.dissect registration|swap|recovery
"""
import importlib
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import numpy as np
import torch

from saccade import data as D, model as M, train as TR
from saccade.eval import s1 as S

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
OUT = HERE / "readings/dissection"
NP = importlib.import_module("artifacts.experiments.20261111_p1f5.native_pairs")
CKPTS = {"f4": EXP / "20261106_p1f4/run/ckpt_10.pt", "hud": EXP / "20261109_p1_hud_change/run/ckpt_10.pt",
         "f5": HERE / "run/ckpt_10.pt"}
DEV = NP.DEV
DRAWS = NP.DRAWS
ARMS_TAG = os.environ.get("DISSECT_ARMS", "")          # registration only: a comma list restricts the arms (split rerun)


# ---------- registration ----------

@torch.no_grad()
def registration():
    CV = importlib.import_module("artifacts.experiments.20261106_determinant_curves.curves")
    RT = importlib.import_module("artifacts.experiments.20261106_determinant_routes.routes")
    L = dict(np.load(CV.LABELS / "s0_dev_labels.npz"))
    T = CV.transitions(L)
    rows = np.sort(np.random.default_rng(2120).choice(np.flatnonzero(T["t"] >= CV.P), 40000, replace=False))
    ep = T["ep"][rows]
    fold = RT.folds_of(ep)
    res = {"rows": len(rows), "archived_f4": {"health": .99865, "food": .26796, "drink": .19100, "energy": .06281}}
    for arm, ck in CKPTS.items():
        if ARMS_TAG and arm not in ARMS_TAG.split(","):
            continue
        model, cfg = CV.IF.load(str(ck))
        enc = S.encode(model, L, cfg)
        e = {}
        for a in range(0, len(rows), 2048):
            j = rows[a:a + 2048]
            for dt in (-1, 0):
                at = lambda key: torch.stack([enc[int(T["ep"][x])][key][int(T["t"][x]) + dt] for x in j])
                p, g, v, fx = (at(k).to(DEV) for k in ("p", "g", "v", "fix"))
                e.setdefault(dt, []).append(model.belief.percept_embed(M.Percept(p.float(), g.float(), v.float(), fx)).float().cpu())
        delta = (torch.cat(e[0]) - torch.cat(e[-1])).double().abs()
        res[arm] = {"sha256": TR.file_sha256(ck)}
        for i, name in enumerate(("health", "food", "drink", "energy")):
            y = (T["true_body"][rows][:, i] != T["body"][rows][:, i]).astype(float)
            s = np.zeros(len(y))
            for f in range(5):
                m = fold == f
                s[m] = RT.logistic(delta[~m], y[~m], delta[m])
            res[arm][name] = {"prevalence": float(y.mean()), "ap": RT.ap_boot(y, s, ep)}
            for tag, m in (("awake", ~T["asleep"][rows]), ("asleep", T["asleep"][rows])):
                if y[m].sum() >= 20:
                    res[arm][name][tag] = {"prevalence": float(y[m].mean()), "events": int(y[m].sum()), "ap": RT.ap_boot(y[m], s[m], ep[m])}
            print(arm, name, res[arm][name], flush=True)
        del model, enc
        torch.cuda.empty_cache()
    write("registration_split.json" if ARMS_TAG else "registration.json", res)


# ---------- generation with route control ----------

@torch.no_grad()
def generate(model, cfg, row, slot, reader, routes, event):
    """native_pairs.generate with the cache's percepts from member v and the belief stream from member b, per (v, b)."""
    i, e, prefix, frames, rewards, terms, q, truth = row
    old_f, old_a, old_r, old_d = prefix
    n = len(old_f)
    f = np.concatenate([np.repeat(old_f[None], 2, 0), frames[:, :q + 1]], 1)
    a = np.r_[old_a, np.zeros(q + 1, dtype=np.int64)]
    rw = np.column_stack([np.repeat(np.r_[0., old_r][None], 2, 0), rewards[:, :q]])
    td = np.column_stack([np.repeat(np.r_[False, old_d][None], 2, 0), terms[:, :q]])
    fix = D.fixation_prior(np.random.default_rng(12000 + i), (len(f[0]),))
    query = torch.as_tensor(D.fixation_prior(np.random.default_rng(15000 + i), ()), device=DEV)
    fix = torch.as_tensor(np.repeat(fix[None], 2, 0), device=DEV)
    act = torch.as_tensor(np.repeat(a[None], 2, 0), device=DEV)
    prev = torch.cat([act.new_full((2, 1), -1), act[:, :-1]], 1)
    out = {}
    with TR.autocast(DEV, cfg):
        p = model.perception(torch.as_tensor(f, device=DEV), fix)
        inputs = model.belief.tokens(model.belief.percept_embed(p), torch.as_tensor(rw, device=DEV, dtype=torch.float32),
                                     torch.as_tensor(td, device=DEV), act)
        y, _ = model.belief(inputs, model.belief.initial(2, DEV))
        bp = torch.cat([torch.zeros_like(y[:, :1]), y[:, 1::2][:, :-1]], 1)
        common = 8 if slot == 64 or event == "rise" else 4
        assert all(torch.equal(z[0, -common:], z[1, -common:]) for z in p.parts())
        for v, b in routes:
            rng = torch.Generator(device=DEV).manual_seed(14000 + i)
            cache = model.denoiser.new_cache()
            for t in range(n + q - 7, n + q + 1):
                cache = M.context_pass(model.denoiser, tuple(z[v:v + 1, t:t + 1] for z in p.parts()),
                                       prev[v:v + 1, t:t + 1], model.belief.memory(bp[b:b + 1, t:t + 1]),
                                       fix[v:v + 1, t:t + 1], cache, cfg, rng)
            cache = {"t": cache["t"], "kv": [tuple(z.expand(DRAWS, *z.shape[1:]) for z in kv) for kv in cache["kv"]]}
            z = M.sample(model.denoiser, [tuple(x.shape[2:]) for x in p.parts()], act.new_zeros(DRAWS, 1),
                         model.belief.memory(y[b:b + 1, -1:]).expand(DRAWS, -1, -1, -1),
                         query[None, None].expand(DRAWS, 1, -1), cache, cfg,
                         torch.Generator(device=DEV).manual_seed(16000 + i))
            digits = reader(torch.cat([z[0][:, 0, slot], z[2][:, 0]], 1).float())["y"].argmax(-1).cpu().numpy()
            base = truth[0, 0]
            assert truth[0, 0] == truth[1, 0]
            hits = int((digits < base).sum() if event == "drop" else (digits > base).sum())
            out[f"{v}{b}"] = {"events": hits, "digits": np.bincount(digits, minlength=10).tolist()}
        real = model.perception(torch.as_tensor(frames[:, q:q + 2], device=DEV), query[None, None].expand(2, 2, -1))
        rp = torch.cat([real.p[:, :, slot], real.v], -1).float()
    assert np.array_equal(reader(rp)["y"].argmax(-1).cpu().numpy(), truth)
    return {"episode": int(e), "true": truth.tolist(), "routes": out}


def summarise(rows, keys):
    rate = {k: np.array([r["routes"][k]["events"] for r in rows], float) / DRAWS for k in keys}
    rng = np.random.default_rng(1111)
    ix = rng.integers(len(rows), size=(5000, len(rows)))
    s = {k: float(v.mean()) for k, v in rate.items()}
    if "00" in rate and "11" in rate:
        d = rate["00"] - rate["11"]
        s["contrast_native"] = float(d.mean())
        s["contrast_native_ci95"] = np.quantile(d[ix].mean(1), [.025, .975]).tolist()
    if len(keys) == 4:
        belief = (rate["00"] + rate["10"] - rate["01"] - rate["11"]) / 2
        visual = (rate["00"] + rate["01"] - rate["10"] - rate["11"]) / 2
        for name, d in (("belief_effect", belief), ("visual_effect", visual)):
            s[name] = float(d.mean())
            s[name + "_ci95"] = np.quantile(d[ix].mean(1), [.025, .975]).tolist()
    return s


def swap():
    res = {}
    keys = ("00", "01", "10", "11")
    for arm in ("f5", "hud"):
        model, cfg = NP.IF.load(CKPTS[arm].resolve())
        reader = NP.readers(model, cfg)[63][0]
        rows = [generate(model, cfg, row, 63, reader, [(int(k[0]), int(k[1])) for k in keys], "drop")
                for row in NP.cooldown_pairs(None)]
        res[arm] = {"sha256": TR.file_sha256(CKPTS[arm]), "summary": summarise(rows, keys), "pairs": rows}
        print(arm, json.dumps(res[arm]["summary"]), flush=True)
        del model
        torch.cuda.empty_cache()
    write("swap.json", res)


# ---------- recovery pairs ----------

def build_recovery(n=32):
    import jax, jax.numpy as jnp
    assert jax.default_backend() == "cpu"
    L = dict(np.load(D.ROOT / "artifacts/eda/s0/s0_dev_labels.npz"))
    off = np.r_[0, np.cumsum(L["ep_T"] + 1)]
    _, _, _, step, render = D.RP.env_and_render()
    bank, excluded = [], []
    order = [e for e in range(len(L["ep_T"])) if e % 5 == 4] + [e for e in range(len(L["ep_T"])) if e % 5 != 4]
    for e in order:
        T = int(L["ep_T"][e])
        lo, hi = off[e:e + 2]
        b = L["s_body"][lo:hi]
        ts = np.arange(T + 1)
        ok = (ts >= 12) & (ts <= T - 45) & (b[:, 0] <= 7) & (b[:, 1:].min(1) >= 3) & ~L["s_sleep"][lo:hi] & (L["s_light"][lo:hi] >= .9)
        idx = np.flatnonzero(ok)
        if not len(idx):
            continue
        t = int(idx[0])
        sh, sl = int(L["ep_shard"][e]), int(L["ep_slot"][e])
        state = D.RP.advance_to(sh, sl, t)
        orig = D.RP.episode_fields(sh, sl)
        assert np.array_equal(np.asarray(render(state)), orig["observations"][t].numpy())
        states = [state.replace(player_recover=jnp.asarray(v, state.player_recover.dtype)) for v in (20., 12.)]
        frames, rewards, term, health = ([[], []] for _ in range(4))
        keys = jax.random.split(jax.random.PRNGKey(111301 + e), 50)
        for s in range(42):
            for br in range(2):
                st = states[br]
                frames[br].append(np.asarray(render(st)))
                health[br].append(int(st.player_health))
                _, nxt, r, d, _ = step(keys[s], st, jnp.asarray(0, jnp.int32))
                states[br] = nxt
                rewards[br].append(float(r))
                term[br].append(bool(d))
        frames, rewards, term, health = map(np.asarray, (frames, rewards, term, health))
        found = []
        for s in range(16, 40):
            if (health[:, s + 1] > health[:, s]).tolist() != [True, False]:
                continue
            if not np.array_equal(frames[0, s - 7:s + 1], frames[1, s - 7:s + 1]):
                continue
            if not np.array_equal(rewards[0, s - 8:s], rewards[1, s - 8:s]) or not np.array_equal(term[0, s - 8:s], term[1, s - 8:s]):
                continue
            if term[:, :s + 1].any() or (np.diff(health[:, :s + 2], axis=1) < 0).any():
                continue
            rises = [np.flatnonzero(np.diff(health[k, :s + 1]) > 0) + 1 for k in (0, 1)]
            if not all(len(x) for x in rises):
                continue
            found.append((s, [int(x[-1]) for x in rises]))
        if not found:
            excluded.append({"e": e, "reason": "no qualified pair"})
            continue
        s, resets = found[0]
        bank.append({"e": e, "t0": t, "end": s, "frames": frames[:, :s + 2], "reward": rewards[:, :s + 1], "term": term[:, :s + 1],
                     "health": health[:, :s + 2], "last_rise": resets,
                     "prefix_frames": orig["observations"][:t].numpy(), "prefix_actions": orig["actions_taken"][:t].numpy(),
                     "prefix_reward": orig["rewards"][:t].numpy(), "prefix_term": orig["terminated"][:t].numpy()})
        print("pair", len(bank), "episode", e, "t0", t, "step", s, "last rises", resets, "health", health[:, s:s + 2].tolist(), flush=True)
        if len(bank) >= n:
            break
    OUT.mkdir(parents=True, exist_ok=True)
    torch.save(bank, OUT / "recovery_bank.pt")
    write("recovery_build.json", {"pairs": len(bank), "excluded": len(excluded),
                                  "rows": [{k: (r[k].tolist() if hasattr(r[k], "tolist") and k in ("last_rise",) else r[k])
                                            for k in ("e", "t0", "end", "last_rise")} for r in bank]})


def recovery_pairs():
    bank = torch.load(OUT / "recovery_bank.pt", map_location="cpu", weights_only=False)
    for i, row in enumerate(bank):
        q = row["end"]
        prefix = (row["prefix_frames"], row["prefix_actions"], row["prefix_reward"], row["prefix_term"])
        yield i, row["e"], prefix, row["frames"], row["reward"], row["term"], q, row["health"][:, q:q + 2]


def recovery():
    res = {}
    for arm in ("f5", "hud", "f4"):
        model, cfg = NP.IF.load(CKPTS[arm].resolve())
        reader = NP.readers(model, cfg)[63][0]
        rows = [generate(model, cfg, row, 63, reader, [(0, 0), (1, 1)], "rise") for row in recovery_pairs()]
        res[arm] = {"sha256": TR.file_sha256(CKPTS[arm]), "summary": summarise(rows, ("00", "11")), "pairs": rows}
        print(arm, json.dumps(res[arm]["summary"]), flush=True)
        del model
        torch.cuda.empty_cache()
    write("recovery.json", res)


@torch.no_grad()
def pricing(episodes=120):
    """F5's own TSP heads at fixed HUD windows: per-cell squared error of the forecast (bp only) and the reactive mode
    (sees the frame), frames split by whether that digit changes; digit-blind reference = the cell's pixel variance."""
    model, cfg = NP.IF.load(CKPTS["f5"].resolve())
    L = dict(np.load(S.LABELS / "s0_dev_labels.npz"))
    off = np.r_[0, np.cumsum(L["ep_T"] + 1)]
    chosen = np.sort(np.random.default_rng(55112).choice(len(L["ep_T"]), episodes, replace=False))
    elist = S.episodes(L)
    # window centre -> {digit name: (position in the window, body column)}
    plan = {(7, 1): {"health": (3, 0), "food": (4, 1), "drink": (5, 2)}, (7, 2): {"food": (3, 1), "drink": (4, 2), "energy": (5, 3)}}
    acc = {}
    for i in chosen:
        ep = elist[i]
        history = D.fixation_prior(np.random.default_rng([0, ep.shard, ep.slot]), (ep.T + 1,))
        chunk = TR.tensors(D.gather([ep], [[(0, 1, t, *f) for t, f in enumerate(history)]]), DEV)
        body = L["s_body"][off[i]:off[i + 1]]
        assert len(body) == ep.T + 1
        changed = body[1:] != body[:-1]                                         # [T, 4] digit changes at frame t
        with TR.autocast(DEV, cfg):
            parts = [model.perception(chunk["frames"][:, a:a + 512], chunk["fixation"][:, a:a + 512]) for a in range(0, ep.T + 1, 512)]
            percept = M.Percept(*(torch.cat(z, 1) for z in zip(*(p.parts() for p in parts))), chunk["fixation"])
            embed = model.belief.percept_embed(percept)
            y, _ = model.belief(model.belief.tokens(embed, chunk["reward"], chunk["terminated"], chunk["action"]), model.belief.initial(1, DEV))
            bp = y[:, 1::2][:, :-1]
            for centre, cells in plan.items():
                q = torch.tensor(centre, device=DEV).expand(1, ep.T, 2)
                win = M.window(q)
                target = M.cell_pixels(M.crop(chunk["frames"][:, 1:], q)).float() / 127.5 - 1
                fc = model.pixels.mean(model.tsp.predict(bp, None, None, win)).float()
                re = model.pixels.mean(model.tsp.predict(bp, percept.p[:, 1:], percept.v[:, 1:], win)).float()
                ef, er = ((fc - target) ** 2).mean(-1)[0].cpu().numpy(), ((re - target) ** 2).mean(-1)[0].cpu().numpy()   # [T, 9]
                for name, (k, col) in cells.items():
                    a = acc.setdefault(f"{centre}/{name}", {"f": [], "r": [], "c": [], "px": [], "win_f": []})
                    a["f"].append(ef[:, k]); a["r"].append(er[:, k]); a["c"].append(changed[:, col])
                    a["px"].append(target[0, :, k].cpu().numpy()); a["win_f"].append(ef.mean(1))
    res = {"episodes": int(len(chosen)), "sha256": TR.file_sha256(CKPTS["f5"])}
    for key, a in acc.items():
        f, r, c, px, wf = (np.concatenate(a[k]) for k in ("f", "r", "c", "px", "win_f"))
        res[key] = {"frames": int(len(f)), "change_frames": int(c.sum()),
                    "forecast_err_unchanged": float(f[~c].mean()), "forecast_err_changed": float(f[c].mean()),
                    "reactive_err_unchanged": float(r[~c].mean()), "reactive_err_changed": float(r[c].mean()),
                    "digit_blind_reference": float(px.var(0).mean()),
                    "share_of_window_forecast_error": float(f.mean() / (9 * wf.mean()))}
        print(key, json.dumps(res[key]), flush=True)
    write("pricing.json", res)


def write(name, obj):
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / name).write_text(json.dumps(obj, indent=1, default=float) + "\n")


if __name__ == "__main__":
    if sys.argv[1] != "build_recovery":
        assert os.environ.get("MAMBA_DETERMINISTIC") == "1"
    {"registration": registration, "swap": swap, "build_recovery": build_recovery, "recovery": recovery, "pricing": pricing}[sys.argv[1]]()
