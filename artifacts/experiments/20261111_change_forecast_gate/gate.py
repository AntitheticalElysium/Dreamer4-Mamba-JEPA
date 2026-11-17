"""Oracle objective gate for the change map's forecast mode (PLAN.md). CPU only; frozen F4 encoder; no training.

    python -m artifacts.experiments.20261111_change_forecast_gate.gate stage0     # foil + camera shift, scene 9
    python -m artifacts.experiments.20261111_change_forecast_gate.gate checks     # C1, C2, C3, noise floors, R
    python -m artifacts.experiments.20261111_change_forecast_gate.gate motion     # only if Stage 0 needs the candidate
"""
import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("OMP_NUM_THREADS", "4")
import json, math, sys
from pathlib import Path
import numpy as np
import torch
import lightgbm as lgb
from saccade import data as D, model as M, train as TR

P = Path(__file__).resolve().parent
E = P.parent
TEACH = D.ROOT / "artifacts/saccade_full_frame_change/targets/change_teacher.txt"
BANK = E / "20261108_full_frame_change/phase_bank.pt"
CKPT = E / "20261106_p1f4/run/ckpt_10.pt"
GRID, CELL = D.GRID, D.CELL
CENTRES = [(r, c) for r in range(1, 8) for c in range(1, 8)]          # m0.yaml fixation_prior: uniform_49_centres


# ---------- canonical change target with the temporal-variability support as an argument ----------

def cells(frames):
    return frames.astype(np.float32).reshape(-1, GRID, CELL, GRID, CELL, 3).transpose(0, 1, 3, 2, 4, 5)


def support(frames):
    """Exactly the w of D.change_correspondence, computed over the frames given (the canonical store: a whole episode)."""
    return (np.abs(np.diff(cells(frames), axis=0)) > 0).mean((0, 3, 4, 5)) + 0.01


def correspond(prev, cur, w, photometric=False):
    """D.change_correspondence for transitions prev[i] -> cur[i] (cell arrays), support w given. `photometric` is the
    single pre-declared candidate (PLAN Stage 0): subtract the per-channel median difference before scoring shifts."""
    rr, qq = np.meshgrid(np.arange(GRID), np.arange(GRID), indexing="ij")
    shifts = np.asarray(D._a9b().SHIFTS)
    cost, valid, sup = [], [], []
    for dr, dc in shifts:
        r, q = rr - dr, qq - dc
        ok = (r >= 0) & (r < GRID) & (q >= 0) & (q < GRID)
        d = cur - prev[:, r.clip(0, GRID - 1), q.clip(0, GRID - 1)]
        if photometric:
            b = np.median(d[:, ok].reshape(len(d), -1, 3), axis=1)
            d = d - b[:, None, None, None, None, :]
        cost.append(np.abs(d).mean((-1, -2, -3)))
        valid.append(ok)
        sup.append(w * w[r.clip(0, GRID - 1), q.clip(0, GRID - 1)] * ok)
    cost, valid, sup = np.stack(cost, 1), np.stack(valid), np.stack(sup)
    glob = ((cost * sup).sum((-1, -2)) / sup.sum((-1, -2))).argmin(1)
    shift = shifts[glob]
    sr, sq = rr[None] - shift[:, 0, None, None], qq[None] - shift[:, 1, None, None]
    source = prev[np.arange(len(prev))[:, None, None], sr.clip(0, GRID - 1), sq.clip(0, GRID - 1)]
    return np.stack([prev, source], 3), np.stack([np.ones_like(valid[glob]), valid[glob]], 3), glob


def targets(prev_frames, cur_frames, w, teacher, photometric=False):
    prev, cur = cells(prev_frames), cells(cur_frames)
    source, valid, glob = correspond(prev, cur, w, photometric)
    exact = (np.abs(cur[:, :, :, None] - source).max((-1, -2, -3)) == 0) & valid
    p = np.full(valid.shape, np.nan, np.float32)
    p[exact] = 0
    pi, ri, qi, mode = np.nonzero(valid & ~exact)
    if len(pi):
        p[pi, ri, qi, mode] = teacher.predict(D._a9b().feats(cur[pi, ri, qi], source[pi, ri, qi, mode]))
    return p, glob


def episode_targets(frames, teacher, w=None, photometric=False):
    """Targets for every transition of a frame sequence, support from `w` or from the sequence itself."""
    w = support(frames) if w is None else w
    return targets(frames[:-1], frames[1:], w, teacher, photometric)


# ---------- costs of forgetting ----------

def ent(p):
    z = np.stack([p, 1 - p], -1)
    return -(z * np.log(np.clip(z, 1e-300, 1))).sum(-1)


def jensen(p):
    """Per-entry excess BCE of the best tied forecast over members/keys on axis 0 (entries valid in all)."""
    common = np.isfinite(p).all(0)
    gap = np.zeros(p.shape[1:])
    gap[common] = np.maximum(ent(p[:, common].astype(float).mean(0)) - ent(p[:, common].astype(float)).mean(0), 0)
    return gap, float(gap.sum() / (np.isfinite(p).sum() / len(p)))     # the G2 reduction: mean over valid entries


def g1_cost(y):
    """y: [members_or_keys, n] percept numbers. Excess squared error of the best tied mean, per number."""
    return float(((y - y.mean(0)) ** 2).mean())


class Teacher:
    def __init__(self):
        self.m = lgb.Booster(model_file=str(TEACH))

    def predict(self, x):
        return self.m.predict(x, num_threads=4)


def encoder():
    ck = torch.load(CKPT, map_location="cpu", weights_only=False, mmap=True)
    enc = M.Perception(ck["cfg"]).eval()
    enc.load_state_dict({k.removeprefix("perception."): v for k, v in ck["model"].items() if k.startswith("perception.")})
    return enc


def percept(enc, frames, fix):
    """frames [n,63,63,3] uint8, fix [n,2] -> [n,1880] (periphery, glimpse, lighting)."""
    with torch.no_grad():
        p = enc(torch.as_tensor(np.ascontiguousarray(frames)), torch.as_tensor(np.asarray(fix), dtype=torch.long))
    return torch.cat([x.reshape(len(frames), -1).float() for x in p.parts()], 1).numpy()


def g1_pair(enc, a, b):
    """Pair cost averaged over the 49 fixation centres of the training prior."""
    fr = np.stack([a, b] * len(CENTRES))
    fx = np.repeat(np.asarray(CENTRES), 2, 0)
    y = percept(enc, fr, fx).reshape(len(CENTRES), 2, -1)
    return float(np.mean([g1_cost(y[i]) for i in range(len(CENTRES))]))


# ---------- simulator ----------

def sim():
    import jax, jax.numpy as jnp
    assert jax.default_backend() == "cpu"
    _, _, _, step, render = D.RP.env_and_render()

    def roll(st, keys, acts):
        fr, sts, r, d = [np.asarray(render(st))], [st], [], []
        for k, a in zip(keys, acts):
            _, st, re, te, _ = step(k, st, jnp.asarray(a, jnp.int32))
            fr.append(np.asarray(render(st))); sts.append(st); r.append(float(re)); d.append(bool(te))
        return np.array(fr), sts, np.array(r), np.array(d)
    return jax, jnp, step, render, roll


def dev_labels():
    L = dict(np.load(D.ROOT / "artifacts/eda/s0/s0_dev_labels.npz"))
    L["off"] = np.concatenate([[0], np.cumsum(L["ep_T"] + 1)])
    return L


def recorded(L, i):
    return D.RP.episode_fields(int(L["ep_shard"][i]), int(L["ep_slot"][i]))


# ---------- C1: the hunger pairs, rebuilt bitwise ----------

def hunger_pairs(L, jax, jnp, roll):
    bank = torch.load(BANK, map_location="cpu", weights_only=False)
    out = []
    for rec in bank:
        i, t0 = rec["e"], rec["t0"]
        st = D.RP.advance_to(int(L["ep_shard"][i]), int(L["ep_slot"][i]), t0)
        kk = jax.random.split(jax.random.PRNGKey(110812 + i), 50)
        runs = [roll(st.replace(player_hunger=jnp.asarray(v, st.player_hunger.dtype)), kk, [0] * 40) for v in (20., 12.)]
        f = np.stack([x[0] for x in runs]); d = np.stack([x[3] for x in runs])
        assert np.array_equal(f[:, :rec["frames"].shape[1]], rec["frames"])
        food = np.array([[int(s.player_food) for s in x[1]] for x in runs])
        if d.any():
            out.append({"e": i, "excluded": "termination"}); continue
        assert np.array_equal(f[0, 24:32], f[1, 24:32])
        assert food[0, 32] == food[0, 31] - 1 and food[1, 32] == food[1, 31]
        out.append({"e": i, "prefix": rec["prefix_frames"], "branch": f, "q": 31, "controls": list(range(24, 31)),
                    "reward": np.stack([x[2] for x in runs]), "term": d})
    return out


# ---------- C2: zombie readiness pairs ----------

def zombie_pairs(L, jax, jnp, step, render, roll, want=32, tries=64, seed=20261111):
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(L["ep_T"]))
    out, scanned = [], 0
    for i in order:
        if len(out) >= want:
            break
        T = int(L["ep_T"][i]); o = L["off"][i]
        cand = []
        for t in range(8, T - 12):
            s = o + t
            if L["s_sleep"][s] or L["s_body"][s, 0] < 6:
                continue
            m = L["s_zombies_m"][s].astype(bool)
            dist = np.abs(L["s_zombies_p"][s] - L["s_pos"][s]).sum(-1)
            near = m & (dist <= 4)
            if near.sum() != 1 or dist[near][0] != 1:
                continue
            if np.any(np.diff(L["s_body"][s - 6:s + 1, 0]) < 0):
                continue
            cand.append(t)
        if not cand:
            continue
        t0 = int(rng.choice(cand)); scanned += 1
        st = D.RP.advance_to(int(L["ep_shard"][i]), int(L["ep_slot"][i]), t0)
        zi = int(np.flatnonzero(np.asarray(st.zombies.mask) & (np.abs(np.asarray(st.zombies.position) - np.asarray(st.player_position)).sum(-1) == 1))[0])
        h0 = int(st.player_health)
        mk = lambda c: st.replace(zombies=st.zombies.replace(attack_cooldown=st.zombies.attack_cooldown.at[zi].set(c)))
        for k in range(tries):
            kk = jax.random.split(jax.random.PRNGKey(seed + 1000 * int(i) + k), 9)
            runs = [roll(mk(c), kk, [0] * 9) for c in (0, 2)]
            hp = np.array([[int(s.player_health) for s in x[1]] for x in runs])
            d = np.stack([x[3] for x in runs])
            sl = np.array([[bool(s.is_sleeping) for s in x[1]] for x in runs])
            ok = (not d.any() and not sl.any() and hp[0, 1] == h0 - 2 and hp[1, 1] == h0 and hp[1, 3] == h0 - 2
                  and hp[0, 6] == hp[1, 6] and hp[0, 7] == hp[0, 6] - 2 and hp[1, 7] == hp[1, 6])
            f = np.stack([x[0] for x in runs])
            if ok and np.array_equal(f[0, 3:7], f[1, 3:7]) and not np.array_equal(f[0, 1], f[1, 1]):
                fields = recorded(L, i)
                assert np.array_equal(f[0, 0], fields["observations"][t0].numpy())
                out.append({"e": int(i), "t0": t0, "key": k, "health0": h0, "zombie": zi,
                            "prefix": fields["observations"][:t0].numpy(), "branch": f, "q": 6, "controls": [3, 4, 5],
                            "reward": np.stack([x[2] for x in runs]), "term": d})
                print("zombie pair", i, t0, k, flush=True)
                break
    return out, scanned


# ---------- natural noise floors ----------

def noise_floor(L, jax, jnp, step, render, teacher, enc, n_ep=120, per=2, keys=8, seed=7):
    rng = np.random.default_rng(seed)
    eps = rng.choice(np.flatnonzero(L["ep_T"] >= 20), n_ep, replace=False)
    rows = []
    for i in eps:
        fields = recorded(L, i)
        frames = fields["observations"].numpy(); acts = fields["actions_taken"].numpy()
        w = support(frames)
        for t in sorted(rng.choice(np.arange(1, len(acts)), per, replace=False)):
            st = D.RP.advance_to(int(L["ep_shard"][i]), int(L["ep_slot"][i]), int(t))
            assert np.array_equal(np.asarray(render(st)), frames[t])
            kk = jax.random.split(jax.random.PRNGKey(seed * 100003 + int(i) * 1009 + int(t)), keys)
            nxt = np.stack([np.asarray(render(step(k, st, jnp.asarray(int(acts[t]), jnp.int32))[1])) for k in kk])
            p, _ = targets(np.repeat(frames[t:t + 1], keys, 0), nxt, w, teacher)
            fix = CENTRES[rng.integers(len(CENTRES))]
            y = percept(enc, nxt, np.repeat([fix], keys, 0))
            rows.append({"e": int(i), "t": int(t), "night": bool(L["s_light"][L["off"][i] + t] < 0.5),
                         "change": jensen(p)[1], "g1": g1_cost(y)})
    return rows


# ---------- per-scene scoring ----------

def scene_costs(sc, teacher, enc, foils=(1, 3, 6), photometric=False):
    pre, br, q = sc["prefix"], sc["branch"], sc["q"]
    tg = [episode_targets(np.concatenate([pre, br[m]]), teacher, photometric=photometric) for m in (0, 1)]
    off = len(pre)
    x = np.stack([tg[m][0][off:] for m in (0, 1)]); glob = np.stack([tg[m][1][off:] for m in (0, 1)])
    gap, cost = jensen(x[:, q:q + 1])
    row = {"e": sc["e"], "change": cost, "gap_cells": np.argwhere(gap > 1e-9).tolist(),
           "controls": [jensen(x[:, c:c + 1])[1] for c in sc["controls"]],
           "g1": g1_pair(enc, br[0, q + 1], br[1, q + 1]), "foil": {}, "shift_query": glob[:, q].tolist()}
    for a in foils:
        ff = br[1].copy()
        ff[q + 1:] = np.minimum(ff[q + 1:].astype(np.int16) + a, 255).astype(np.uint8)
        xf, gf = episode_targets(np.concatenate([pre, ff]), teacher, photometric=photometric)
        pair = np.stack([x[1, q], xf[off + q]])[:, None]
        fgap, fcost = jensen(pair)
        row["foil"][str(a)] = {"change": fcost, "shift": [int(glob[1, q]), int(gf[off + q])],
                               "gap_by_plane": fgap.sum((0, 1, 2)).tolist(), "g1": g1_pair(enc, br[1, q + 1], ff[q + 1])}
    return row


def reproduction_control(L, teacher):
    fields = recorded(L, 3)
    fr = fields["observations"].numpy()
    a = D.change_probabilities(fr, teacher)
    b, _ = episode_targets(fr, teacher)
    assert a.shape == b.shape and np.array_equal(np.isnan(a), np.isnan(b)) and np.array_equal(a[~np.isnan(a)], b[~np.isnan(b)])
    return {"episode": 3, "transitions": int(len(a)), "identical": True}


# ---------- stages ----------

def stage0():
    jax, jnp, step, render, roll = sim()
    L, teacher = dev_labels(), Teacher()
    out = {"reproduction": reproduction_control(L, teacher)}
    sc = next(s for s in hunger_pairs(L, jax, jnp, roll) if s["e"] == 9)
    br, pre = sc["branch"], sc["prefix"]
    # 0a: the 11-11 gate's window (support over branch frames 24..40, the foil from frame 32), its +3 foil.
    win = br[:, 24:41]
    xw = np.stack([episode_targets(win[m], teacher)[0] for m in (0, 1)])
    foil = win[1].copy(); foil[8:] = np.minimum(foil[8:].astype(np.int16) + 3, 255).astype(np.uint8)
    xf, gfw = episode_targets(foil, teacher)
    _, gw0 = episode_targets(win[1], teacher)
    out["window_17"] = {"note": "support over the short branch window, as the 11-11 gate",
                        "hunger_one_step": jensen(xw[:, 7:8])[1], "foil3_one_step": jensen(np.stack([xw[1, 7], xf[7]])[:, None])[1],
                        "shift_plain": gw0.tolist(), "shift_foil": gfw.tolist()}
    for photometric in (False, True):
        r = scene_costs(sc, teacher, encoder(), photometric=photometric)
        out["episode_support" + ("_photometric" if photometric else "")] = {k: v for k, v in r.items()}
    (P / "STAGE0.json").write_text(json.dumps(out, indent=1, default=float))
    print(json.dumps(out, indent=1, default=float)[:3000])


def checks(photometric=False):
    jax, jnp, step, render, roll = sim()
    L, teacher, enc = dev_labels(), Teacher(), encoder()
    torch.set_num_threads(4)
    rep = reproduction_control(L, teacher)
    c1 = hunger_pairs(L, jax, jnp, roll)
    c2, scanned = zombie_pairs(L, jax, jnp, step, render, roll)
    rows = {"C1": [], "C2": []}
    for name, bank in (("C1", c1), ("C2", c2)):
        for sc in bank:
            if "excluded" in sc:
                rows[name].append(sc); continue
            r = scene_costs(sc, teacher, enc, photometric=photometric)
            r.update({k: sc[k] for k in ("t0", "key", "health0") if k in sc})
            rows[name].append(r)
            print(name, sc["e"], round(r["change"], 6), round(r["g1"], 6), {a: round(v["change"], 6) for a, v in r["foil"].items()}, flush=True)
    noise = noise_floor(L, jax, jnp, step, render, teacher, enc)
    report = {"plan": "PLAN.md", "photometric_candidate": photometric, "reproduction": rep,
              "teacher_sha256": TR.file_sha256(TEACH), "bank_sha256": TR.file_sha256(BANK), "encoder_sha256": TR.file_sha256(CKPT),
              "zombie_states_scanned": scanned, "rows": rows, "noise": noise}
    name = "CHECKS" + ("_photometric" if photometric else "") + ".json"
    (P / name).write_text(json.dumps(report, indent=1, default=float))


def motion(n=2000, seed=11):
    """Camera-shift agreement with the simulator's player displacement, canonical vs candidate (only if needed)."""
    L = dev_labels(); rng = np.random.default_rng(seed)
    eps = rng.choice(np.flatnonzero(L["ep_T"] >= 20), 200, replace=False)
    shifts = np.asarray(D._a9b().SHIFTS); res = []
    for i in eps:
        fr = recorded(L, i)["observations"].numpy(); w = support(fr); o = L["off"][i]
        ts = np.sort(rng.choice(len(fr) - 1, n // 200, replace=False))
        c = cells(fr)
        for ph in (False, True):
            _, _, g = correspond(c[ts], c[ts + 1], w, ph)
            for t, gg in zip(ts, g):
                dp = L["s_pos"][o + t + 1] - L["s_pos"][o + t]
                res.append({"photometric": ph, "night": bool(L["s_light"][o + t] < 0.5), "moved": bool(np.abs(dp).sum() > 0),
                            "shift": shifts[gg].tolist(), "dpos": dp.tolist()})
    (P / "MOTION.json").write_text(json.dumps(res))


if __name__ == "__main__":
    {"stage0": stage0, "checks": lambda: checks("--photometric" in sys.argv), "motion": motion}[sys.argv[1]]()
