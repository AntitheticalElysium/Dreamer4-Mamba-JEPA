"""Danger M1 / M2 (PLAN.md 5-6) on P1-F4's checkpoint under the canonical code (df702338).
  m1       (GPU): health changes by cause on dev strata, exp 2's generation and readers; R0 sham = exp 2's zombie
           population regenerated equals F4's saved futures -> readings/danger/m1.json
  m2build  (CPU, JAX): hit pairs (a sleeping agent, a ready adjacent zombie at t - 1 vs the adjacent zombies removed)
           -> readings/danger/m2_bank.pt
  m2run    (GPU): exp 3's 2 x 2 (cache x memory) on the pairs, F4's sampler -> readings/danger/m2.json
Usage: MAMBA_DETERMINISTIC=1 python -m artifacts.experiments.20261107_f4_followups.danger_m m1|m2run;
  JAX_PLATFORMS=cpu python -m artifacts.experiments.20261107_f4_followups.danger_m m2build"""
import importlib, json, os, sys
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import numpy as np
import torch

from saccade import data as D, train as TR
from saccade.eval import s1 as S

CV = importlib.import_module("artifacts.experiments.20261106_determinant_curves.curves")
RT = importlib.import_module("artifacts.experiments.20261106_determinant_routes.routes")
RF = importlib.import_module("artifacts.experiments.20261107_f4_followups.routes_f4")
HERE = Path(__file__).resolve().parent
OUT = HERE / "readings/danger"
CKPT = RF.CKPT
F4RAW = HERE.parent / "20261106_p1f4/readings/consequences/curves/raw.npz"
CAP = 1000


def ratio(a, b, ep, n=2000, seed=1107):
    """mean(a) / mean(b) with an episode bootstrap (a, b per future)."""
    ids, inv = np.unique(ep, return_inverse=True)
    sa, sb = np.bincount(inv, a, len(ids)), np.bincount(inv, b, len(ids))
    w = np.stack([np.bincount(r, minlength=len(ids)) for r in np.random.default_rng(seed).integers(0, len(ids), (n, len(ids)))])
    bt = (w @ sa) / np.maximum(w @ sb, 1e-12)
    return {"est": float(a.mean() / max(b.mean(), 1e-12)), "ci95": [float(np.percentile(bt, 2.5)), float(np.percentile(bt, 97.5))]}


def m1_strata(L, T):
    """PLAN 5: the situation at t - 1 (curves.transitions' g); first match wins."""
    g, pos = T["g"], L["s_pos"][T["g"]].astype(int)
    dist = lambda k: np.where(L[f"s_{k}_m"][g], np.abs(L[f"s_{k}_p"][g].astype(int) - pos[:, None]).sum(-1), 99).min(1)
    zd, sd, ad = dist("zombies"), dist("skeletons"), dist("arrows")
    quiet = (zd > 3) & (sd > 3) & ~L["s_arrows_m"][g].any(1)
    health, food, drink, energy = (T["body"][:, i] for i in range(4))
    asleep = T["asleep"]
    need_awake, need_asleep = (food == 0) | (drink == 0) | (energy == 0), (food == 0) | (drink == 0)   # game_logic necessities
    lab = np.full(len(g), "", object)
    for name, m in (("arrow_awake", (ad == 1) & (zd != 1) & ~asleep), ("arrow_asleep", (ad == 1) & (zd != 1) & asleep),
                    ("need_damage_awake", quiet & ~asleep & need_awake), ("need_damage_asleep", quiet & asleep & need_asleep),
                    ("recovery_awake", quiet & ~asleep & ~need_awake & (health < 9)),
                    ("recovery_asleep", quiet & asleep & ~need_asleep & (health < 9))):
        lab[(T["t"] >= CV.P) & m & (lab == "")] = name
    return lab


def m1():
    assert os.environ.get("MAMBA_DETERMINISTIC") == "1"
    model, cfg = CV.IF.load(CKPT)
    L = dict(np.load(CV.LABELS / "s0_dev_labels.npz"))
    enc = S.encode(model, L, cfg)
    T = CV.transitions(L)
    with torch.enable_grad():
        R, valid = CV.fit_readers(model, cfg, L, enc, 0)
    s1 = S.load_reader(TR.ROOT / Path(CKPT).parent / "s1/results.reader.pt", TR.ROOT / CKPT, CV.DEV)
    raw = np.load(F4RAW, allow_pickle=True)                             # R0: exp 2's zombie population, its seed
    rows = CV.select(CV.strata(T), np.random.default_rng(2106), 1.0)["zombie"]
    o = CV.generate(model, cfg, enc, L, T, rows, 4100 + CV.POPS.index("zombie"), R, s1)
    sham = {k: bool(np.array_equal(o[k], raw[f"zombie/{k}"])) for k in ("j", "gen_hud63", "gen_asleep", "gen_term_p")}
    print("R0", sham, flush=True)
    lab = m1_strata(L, T)
    rng, out = np.random.default_rng(2111), {"checkpoint": CKPT, "sham_r0": sham, "readers": valid, "strata": {}}
    for i, name in enumerate(sorted(set(lab) - {""})):
        idx = np.nonzero(lab == name)[0]
        idx = np.sort(rng.choice(idx, CAP, replace=False)) if len(idx) > CAP else idx
        o = CV.generate(model, cfg, enc, L, T, idx, 4300 + i, R, s1)
        j = o["j"]; h0, h1 = T["body"][j, 0], T["true_body"][j, 0]; ep = T["ep"][j]
        gd, td = (o["gen_hud63"] < h0).astype(float), (h1 < h0).astype(float)
        gr, tr = (o["gen_hud63"] > h0).astype(float), (h1 > h0).astype(float)
        out["strata"][name] = {"rows": int(len(idx)), "episodes": int(len(np.unique(ep))), "futures": int(len(j)),
                               "drop": {"gen": RT.boot(gd, ep), "true": RT.boot(td, ep), "gen_over_true": ratio(gd, td, ep)},
                               "rise": {"gen": RT.boot(gr, ep), "true": RT.boot(tr, ep), "gen_over_true": ratio(gr, tr, ep)}}
        print(name, json.dumps({k: (round(v["gen"]["est"], 4), round(v["true"]["est"], 4)) for k, v in out["strata"][name].items()
                                if isinstance(v, dict)}), flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "m1.json").write_text(json.dumps(out, indent=1))


def m2build():
    import jax, jax.numpy as jnp
    assert jax.default_backend() == "cpu"
    L = dict(np.load(CV.LABELS / "s0_dev_labels.npz"))
    T = CV.transitions(L)
    g, s_off = T["g"], np.r_[0, np.cumsum(L["ep_T"] + 1)]
    prev = np.maximum(g - 1, 0)
    adj_prev = (L["s_zombies_m"][prev] & (np.abs(L["s_zombies_p"][prev].astype(int) - L["s_pos"][prev].astype(int)[:, None]).sum(-1) == 1)).any(1) & ~np.isin(g, s_off)
    idx = np.nonzero((T["t"] >= CV.P) & T["asleep"] & T["ready"] & ~L["t_terminated"].astype(bool))[0]
    j = RT.choose(T, idx, 240, 1)
    _, _, _, step, render = D.RP.env_and_render()
    rows, excluded = [], []
    for jj in j:
        e, t = int(T["ep"][jj]), int(T["t"][jj])
        shard, slot = int(L["ep_shard"][e]), int(L["ep_slot"][e])
        st = D.RP.advance_to(shard, slot, t - 1)
        obs = D.RP.episode_fields(shard, slot)["observations"]
        assert np.array_equal(np.asarray(render(st)), obs[t - 1].numpy()), ("replay mismatch", e, t)
        z = st.zombies
        adj = (jnp.abs(z.position - st.player_position).sum(-1) == 1) & z.mask
        edit = st.replace(zombies=z.replace(mask=z.mask & ~adj))
        a = int(L["t_action"][jj])
        key = D.RP._slot_keys(shard, slot)[1][t - 1]
        nxt = [step(key, s, jnp.asarray(a, jnp.int32))[1] for s in (st, edit)]
        succ = [np.asarray(render(s)) for s in nxt]
        assert np.array_equal(succ[0], obs[t].numpy()), ("natural successor differs from the stored frame", e, t)
        h0, h, sl = int(st.player_health), [int(s.player_health) for s in nxt], [bool(s.is_sleeping) for s in nxt]
        truth = {"health": [h0, *h], "asleep": sl}
        if not (h[0] < h0 and not sl[0] and h[1] >= h0 and sl[1]):
            excluded.append({"e": e, "t": t, "truth": truth})
            continue
        rows.append({"kind": "hit", "e": e, "t": t, "j": int(jj), "stratum": "beside" if adj_prev[jj] else "fresh",
                     "recorded_action": a, "actions": (a, a), "frame": np.stack([np.asarray(render(st)), np.asarray(render(edit))]),
                     "successor": np.stack(succ), "truth": truth})
        D.RP._shard.cache_clear()
    OUT.mkdir(parents=True, exist_ok=True)
    torch.save(rows, OUT / "m2_bank.pt")
    (OUT / "m2_bank.json").write_text(json.dumps({"pairs": len(rows), "episodes": len({r["e"] for r in rows}),
                                                  "strata": {s: sum(r["stratum"] == s for r in rows) for s in ("fresh", "beside")},
                                                  "excluded": excluded, "bank_sha256": TR.file_sha256(OUT / "m2_bank.pt")}, indent=1))
    print("pairs", len(rows), "excluded", len(excluded), flush=True)


def setup():
    """F4's checkpoint (the canonical Denoiser refuses any other recipe at construction), dev labels, S1 encode."""
    model, cfg = CV.IF.load(CKPT)
    assert cfg["edm"]["sampler"]["steps"] == 4
    L = dict(np.load(CV.LABELS / "s0_dev_labels.npz"))
    return model, cfg, L, S.encode(model, L, cfg)


@torch.no_grad()
def m2run():
    model, cfg, L, enc = setup()
    with torch.enable_grad():
        R, valid = CV.fit_readers(model, cfg, L, enc, 0)
    s1 = S.load_reader(TR.ROOT / Path(CKPT).parent / "s1/results.reader.pt", TR.ROOT / CKPT, CV.DEV)
    shapes, lo = [tuple(enc[0][k].shape[1:]) for k in ("p", "g", "v")], cfg["edm"]["inference_context_sigma"]
    res = []
    for i, r in enumerate(torch.load(OUT / "m2_bank.pt", weights_only=False)):
        P2, Ps, b, bp, caches, fx = RT.pair_state(model, cfg, enc, r)
        sham = {"tokens_max_abs": float((P2.p[0].float() - enc[r["e"]]["p"][r["t"] - 1].to(CV.DEV).float()).abs().max()),
                "belief_rel": float((bp[2] - enc[r["e"]]["bp"][r["t"]].to(CV.DEV).float()).norm() / bp[2].norm())}
        assert sham["tokens_max_abs"] < 0.04, ("natural member's tokens differ from S1 encode", sham)
        h0, gen = r["truth"]["health"][0], []
        for ck, mm in RT.COND:
            xs = RF.sample(model, cfg, shapes, r["actions"][ck], bp[mm:mm + 1], fx, caches[lo][ck], RT.K, r["j"])
            if (ck, mm) == (0, 0) and i % 25 == 0:
                again = RF.sample(model, cfg, shapes, r["actions"][ck], bp[mm:mm + 1], fx, caches[lo][ck], RT.K, r["j"])
                assert all(torch.equal(u, v) for u, v in zip(xs, again)), "generation sham differs"
                sham["generation_repeat"] = "bitwise equal"
            rd = CV.read(R, s1, xs[0][:, 0].float().cpu(), xs[2][:, 0].float().cpu(), np.full(RT.K, CV.PLAYER))
            gen.append({"drop": (rd["hud63"] < h0).astype(float), "woken": (rd["asleep"] < 0.5).astype(float)})
        rec = CV.read(R, s1, Ps.p.float().cpu(), Ps.v.float().cpu(), np.full(2, CV.PLAYER))
        res.append({"e": r["e"], "stratum": r["stratum"], "sham": sham, "gen": gen,
                    "rec_check": bool((rec["hud63"] < h0).tolist() == [True, False] and (rec["asleep"] < 0.5).tolist() == [True, False])})
    readings = {"pairs": len(res), "readers": valid, "recorded_reader_check": float(np.mean([x["rec_check"] for x in res])),
                "shams": {"tokens_max_abs": max(x["sham"]["tokens_max_abs"] for x in res), "belief_rel": max(x["sham"]["belief_rel"] for x in res),
                          "generation_repeats_bitwise": sum("generation_repeat" in x["sham"] for x in res)}}
    ep = np.array([x["e"] for x in res])
    for ev in ("drop", "woken"):
        E = np.array([[x["gen"][c][ev].mean() for c in range(4)] for x in res])      # COND: 00 both zombie, 10 memory-only,
        tr = np.full(len(res), -1.0)                                                 # 01 cache-only, 11 neither
        rd = {"factorial": RT.factorial(E, tr, ep),
              "share_cache_only": ratio(E[:, 2] - E[:, 3], E[:, 0] - E[:, 3], ep),
              "share_memory_only": ratio(E[:, 1] - E[:, 3], E[:, 0] - E[:, 3], ep)}
        for s in ("fresh", "beside"):
            m = np.array([x["stratum"] == s for x in res])
            if m.sum() >= 10:
                rd[f"cells_{s}"] = {c: float(E[m, i].mean()) for i, c in enumerate(("00", "10", "01", "11"))}
        readings[ev] = rd
        print(ev, {c: round(v["est"], 3) for c, v in rd["factorial"]["cells"].items()}, "cache-only share",
              round(rd["share_cache_only"]["est"], 3), "memory-only share", round(rd["share_memory_only"]["est"], 3), flush=True)
    (OUT / "m2.json").write_text(json.dumps(readings, indent=1, default=str))


if __name__ == "__main__":
    {"m1": m1, "m2build": m2build, "m2run": m2run}[sys.argv[1]]()
