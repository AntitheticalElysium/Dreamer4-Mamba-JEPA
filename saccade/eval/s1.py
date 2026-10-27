"""S1 readings of M0 after P1 (TEST_SPEC S1; written before P1 ends, 2026-10-27). Dev split for readings; train-400 for
probe fitting; simulator labels for evaluation only.

1. Entity retention in percept tokens (S1.1, and the S3.2 perception blocker): the S0 per-cell probe definition
   (saccade/eval/probe.py) with the cell's periphery token in place of its pixels and the lighting latent in place of
   the frame mean; trained encoder and its untrained copy (the same initialisation), 2 probe seeds each; read by light
   and ring beside the pixel reference at the periphery's acuity (acuity_precision.json, r = 3).
2. Belief contents (S1.2): features b_t (Mamba after the percept, reward and termination of t); controls (i) the current
   percept with its reward and termination, (ii) the last 8 steps with everything the belief saw in them. Targets:
   achievements (22), HUD quantities (16, capped at 9), recent sightings by kind-level absence stratum and last-seen nearest edge.
3. One-step generation (S1.3): next percepts sampled from real histories (8 real context frames at the inference
   context level, the real predicted belief, action and fixation); classifier two-sample test on matched tuples with
   its power on corrupted real tuples; entity counts through the frozen reader (the S1.1 probe, seed 0); reward and
   continuation calibration on real percepts beside the S0 references (s0_calib.json).
4. TSP (S1.4): glimpse likelihood, nats per pixel, beside copy-previous and per-position-mean baselines.
Fixations: the random prior with a fixed seed per episode. Probe protocol (all probes): MLP 256-256 GELU, AdamW 1e-3
(wd 1e-4), batch 4096, held-out 10% of train-400 episodes, stop after 3 epochs without held-out improvement (<= 30).
Usage (project venv, GPU): python -m saccade.eval.s1 CHECKPOINT OUT_JSON
  [MAX_DEV_EPISODES MAX_TRAIN_EPISODES] (limits for smoke tests only)
"""
import json, math, sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from saccade import data as D, model as M, train as TR
from saccade.eval import gates as G
from saccade.eval.probe import KEEP, RING

DEV = torch.device("cuda")
ROOT = Path(__file__).resolve().parents[2]
LABELS = ROOT / "artifacts/eda/s0"
STRATA = ((1, 4), (5, 8), (9, 16), (17, 64))
SIDES = ("top", "bottom", "left", "right")


def episodes(L):
    return [D.Episode(int(s), int(o), int(T), "") for s, o, T in zip(L["ep_shard"], L["ep_slot"], L["ep_T"])]


@torch.no_grad()
def encode(model, L, cfg, untrained=None):
    """Per episode of a labels file: periphery / glimpse / lighting tokens, fixation, corrected and predicted beliefs,
    the outcome heads on the real next percept (P of the recorded reward bin, P(unlock | observed health change),
    P(continue)) and the TSP
    glimpse NLL per frame. With `untrained`, also that encoder's periphery tokens and lighting latent."""
    out = []
    offsets = np.r_[0, np.cumsum(L["ep_T"] + 1)]
    for e, ep in enumerate(episodes(L)):
        rng = np.random.default_rng([cfg["p1"]["monitor"]["fixed_fixation_seed"], ep.shard, ep.slot])
        recs = [[(0, 1, t, *D.fixation_prior(rng, ())) for t in range(ep.T + 1)]]
        c = TR.tensors(D.gather([ep], recs), DEV)
        with TR.autocast(DEV, cfg):
            parts = [model.perception(c["frames"][:, a:a + 512], c["fixation"][:, a:a + 512]) for a in range(0, ep.T + 1, 512)]
            P = M.Percept(*(torch.cat(z, 1) for z in zip(*[p.parts() for p in parts])), c["fixation"])
            emb = model.belief.percept_embed(P)
            y, _ = model.belief(model.belief.tokens(emb, c["reward"], c["terminated"], c["action"]), model.belief.initial(1, DEV))
            b, bp = y[:, 0::2], torch.cat([torch.zeros_like(y[:, :1]), y[:, 1::2][:, :-1]], 1)
            h = model.outcome.features(bp, emb)
            lp = model.outcome.bin_logp(h)
            p_true = lp.gather(-1, model.outcome.bin_of(c["reward"])[..., None])[..., 0].exp()
            health = torch.as_tensor(L["s_body"][offsets[e]:offsets[e + 1], 0], device=DEV).float()
            dh = torch.cat([health.new_zeros(1), health[1:] - health[:-1]])
            p_unlock = unlock_probability(lp, model.outcome.bin_value, dh[None])
            p_cont = torch.sigmoid(model.outcome.cont_logit(h, c["reward"]).float())
            cells = M.window(c["fixation"])
            pix = M.cell_pixels(M.crop(c["frames"], c["fixation"]))
            tsp = M.pixel_nll(*model.pixels(model.tsp.predict(bp, P.p, P.v, cells), P.v, cells), pix).mean((-1, -2))
        rec = {"p": P.p[0].half().cpu(), "g": P.g[0].half().cpu(), "v": P.v[0].half().cpu(), "fix": c["fixation"][0].cpu(),
               "b": b[0].float().half().cpu(), "bp": bp[0].float().half().cpu(), "p_true": p_true[0].float().cpu(),
               "p_unlock": p_unlock[0].float().cpu(), "p_cont": p_cont[0].cpu(), "tsp": tsp[0].float().cpu(),
               "action": c["action"][0].cpu(), "prev_action": c["prev_action"][0].cpu(), "reward": c["reward"][0].cpu(),
               "terminated": c["terminated"][0].cpu()}
        if untrained is not None:
            with TR.autocast(DEV, cfg):
                U = [untrained(c["frames"][:, a:a + 512], c["fixation"][:, a:a + 512]) for a in range(0, ep.T + 1, 512)]
            rec["p_u"], rec["v_u"] = torch.cat([u.p for u in U], 1)[0].half().cpu(), torch.cat([u.v for u in U], 1)[0].half().cpu()
        out.append(rec)
        if e % 200 == 0:
            print(f"encoded {e}/{len(L['ep_T'])}", flush=True)
    return out


class MLP(nn.Module):
    def __init__(self, n_in, heads: dict):
        super().__init__()
        self.body = nn.Sequential(nn.Linear(n_in, 256), nn.GELU(), nn.Linear(256, 256), nn.GELU())
        self.heads = nn.ModuleDict({k: nn.Linear(256, v) for k, v in heads.items()})

    def forward(self, x):
        h = self.body(x.float())
        return {k: m(h) for k, m in self.heads.items()}


def fit(make_x, loss, n, n_val, heads, seed, device=None):
    """The S0 probe protocol on index sets: make_x(idx, val) -> inputs, loss(out, idx, val) -> scalar."""
    device = DEV if device is None else torch.device(device)
    torch.manual_seed(seed)
    model = MLP(make_x(torch.arange(2, device=device), False).shape[1], heads).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    g = torch.Generator(device=device).manual_seed(seed)
    best, state, best_ep, log = 1e9, None, 0, []
    for epoch in range(30):
        model.train()
        for idx in torch.randperm(n, device=device, generator=g).split(4096):
            lo = loss(model(make_x(idx, False)), idx, False)
            opt.zero_grad(), lo.backward(), opt.step()
        model.eval()
        with torch.no_grad():
            vl = float(np.mean([loss(model(make_x(i, True)), i, True).item() for i in torch.arange(n_val, device=device).split(65536)]))
        log.append(vl)
        if vl < best:
            best, state, best_ep = vl, {k: v.clone() for k, v in model.state_dict().items()}, epoch
        elif epoch - best_ep >= 3:
            break
    model.load_state_dict(state)
    return model.eval(), log


ap = G.average_precision


def unlock_probability(log_bin_probability, bin_values, health_change):
    """Evaluation only: the same unlock event as round(reward - .1 * health change) == 1.

    Health labels define the event; they are never fed into the trained world model.
    """
    unlock = (bin_values - 0.1 * health_change[..., None]).round() == 1
    return (log_bin_probability.exp() * unlock).sum(-1)


def save_reader(reader, path, checkpoint, world_seed, cfg):
    """Persist the exact fitted S1 reader for S2, bound to its encoder checkpoint."""
    schema = {"token_dim": cfg["perception"]["periphery"]["token_dim"],
              "lighting_dim": cfg["perception"]["lighting"]["dim"],
              "hidden": [256, 256], "heads": {"kind": 5, "terr": 17},
              "kind_labels": ["background", *G.KINDS], "probe_seed": 0, "storage_precision": "float16",
              "train_labels_sha256": TR.file_sha256(LABELS / "s0_train400_labels.npz")}
    torch.save({"state_dict": {k: v.detach().cpu() for k, v in reader.state_dict().items()}, "schema": schema,
                "checkpoint_sha256": TR.file_sha256(checkpoint), "world_seed": world_seed}, path)
    return {"path": str(path), "sha256": TR.file_sha256(path), "checkpoint_sha256": TR.file_sha256(checkpoint),
            "schema": schema}


def load_reader(path, checkpoint, device):
    """Load a frozen reader only for the encoder checkpoint on which it was fitted."""
    saved = torch.load(path, weights_only=False, map_location="cpu")
    if saved["checkpoint_sha256"] != TR.file_sha256(checkpoint):
        raise ValueError("Reader belongs to a different encoder checkpoint")
    schema = saved["schema"]
    reader = MLP(schema["token_dim"] + schema["lighting_dim"], schema["heads"])
    reader.load_state_dict(saved["state_dict"])
    return reader.to(device).eval().requires_grad_(False)


# ---- S1.1 token probe ----

def cell_table(enc, L, key_p="p", key_v="v"):
    """Map cells but the player's: token [n*62, 16], lighting [n*62, 8], kind, terrain (-1 if covered), ring, night, ep."""
    s_off = np.r_[0, np.cumsum(L["ep_T"] + 1)]
    cols = {k: [] for k in ("tok", "v", "kind", "terr", "ring", "night", "ep")}
    for e, ep in enumerate(G.episode_grids(L)):
        n = len(ep["grid"])
        cols["tok"].append(enc[e][key_p][:, :63][:, KEEP].reshape(-1, enc[e][key_p].shape[-1]))
        cols["v"].append(enc[e][key_v].repeat_interleave(int(KEEP.sum()), 0))
        kind = ep["grid"].reshape(n, 63)[:, KEEP]
        cols["kind"].append(torch.as_tensor(kind.ravel().astype(np.int64)))
        cols["terr"].append(torch.as_tensor(np.where(kind == 0, ep["view"].reshape(n, 63)[:, KEEP], -1).ravel().astype(np.int64)))
        cols["ring"].append(np.tile(RING[KEEP], n)); cols["night"].append(np.repeat(L["s_light"][s_off[e]:s_off[e + 1]] < 0.5, KEEP.sum()))
        cols["ep"].append(np.full(n * int(KEEP.sum()), e))
    return {k: (torch.cat(v) if isinstance(v[0], torch.Tensor) else np.concatenate(v)) for k, v in cols.items()}


def token_probe(tr, dv, seed):
    held = torch.as_tensor(tr["ep"] % 10 == 9)
    T = {k: tr[k][~held].to(DEV) for k in ("tok", "v", "kind", "terr")}; V = {k: tr[k][held].to(DEV) for k in ("tok", "v", "kind", "terr")}
    mx = lambda idx, val: torch.cat([(V if val else T)["tok"][idx], (V if val else T)["v"][idx]], 1)
    lf = lambda o, idx, val: F.cross_entropy(o["kind"], (V if val else T)["kind"][idx]) + \
        F.cross_entropy(o["terr"], (V if val else T)["terr"][idx], ignore_index=-1)
    model, log = fit(mx, lf, len(T["kind"]), len(V["kind"]), {"kind": 5, "terr": 17}, seed)
    with torch.no_grad():
        kp = torch.cat([model(torch.cat([dv["tok"][i].to(DEV), dv["v"][i].to(DEV)], 1))["kind"].argmax(1).cpu()
                        for i in torch.arange(len(dv["kind"])).split(262144)]).numpy()
    return model, kp, log


def retention(dv, kp):
    kind, ep = dv["kind"].numpy(), dv["ep"]
    pres, pp = kind > 0, kp > 0
    res = {}
    for lname, lm in (("day", ~dv["night"]), ("night", dv["night"])):
        for rname, rm in (("r1_2", dv["ring"] <= 2), ("r1_4", dv["ring"] >= 1)):
            cnd = lm & rm
            res[f"{lname}_{rname}"] = {"presence_recall": G.estimate(pp, pres & cnd, ep), "presence_precision": G.estimate(pres, pp & cnd, ep),
                                       **{f"recall_{kn}": G.estimate(kp == k, (kind == k) & cnd, ep) for k, kn in enumerate(G.KINDS, 1)},
                                       **{f"precision_{kn}": G.estimate(kind == k, (kp == k) & cnd, ep) for k, kn in enumerate(G.KINDS, 1)}}
    return res


def perception_blocker(trained, untrained, reference=None):
    """TEST_SPEC S3.2 (perception), per kind on rings 1-2, day and night: a kind's recall or precision more than 0.10
    below the pixel reference at acuity 3, or below the untrained copy by > 0.02 beyond 3 SE (means of the probe seeds)."""
    ref = reference if reference is not None else json.loads(
        (ROOT / "artifacts/eda/system_design_v1/acuity/acuity_precision.json").read_text())["results"]["3"]
    fired, unmeasured = [], []
    for light in ("day", "night"):
        for kn in G.KINDS:
            for m in ("recall", "precision"):
                key = f"{light}_r1_2"
                metric = f"{m}_{kn}"
                rr = [ref[s][key][metric] for s in ref]
                t = [x[key][metric] for x in trained]; u = [x[key][metric] for x in untrained]
                measured = {}
                for source, values in (("pixel_reference", rr), ("trained", t), ("untrained", u)):
                    measured[source] = bool(values) and all(v is not None and v["n_cond"] > 0 for v in values)
                    if not measured[source]:
                        unmeasured.append({"kind": kn, "light": light, "metric": m, "source": source})
                if not measured["trained"]:
                    continue
                te, ts = np.mean([x["est"] for x in t]), np.mean([x["se"] for x in t])
                if measured["pixel_reference"]:
                    r = np.mean([x["est"] for x in rr])
                    if te < r - 0.10:
                        fired.append({"kind": kn, "light": light, "metric": m, "rule": "pixel_reference", "est": te, "ref": r})
                if measured["untrained"]:
                    ue, us = np.mean([x["est"] for x in u]), np.mean([x["se"] for x in u])
                    if te < ue - 0.02 - 3 * math.hypot(ts, us):
                        fired.append({"kind": kn, "light": light, "metric": m, "rule": "untrained_copy", "est": te, "ref": ue})
    return {"status": "blocked" if fired else "unmeasured" if unmeasured else "passed",
            "fired": fired, "unmeasured": unmeasured}


# ---- S1.2 belief contents ----

def departed_episode(grid):
    """Observable history only: time since a kind was last visible while it is absent now.

    Side means the nearest screen edge at that last sighting, not an unseen exit position.
    With multiple sightings choose the closest edge, ties by row-major cell then top/bottom/left/right.
    """
    cls = np.zeros((len(grid), len(G.KINDS)), np.int64)
    side = np.full_like(cls, -1)
    for k in range(len(G.KINDS)):
        last, seen_side = -10**9, -1
        for t, frame in enumerate(grid):
            visible = frame == k + 1
            visible[frame.shape[0] // 2, frame.shape[1] // 2] = False  # player sprite occludes this cell (S0 KEEP)
            cells = np.argwhere(visible)
            if len(cells):
                r, c = cells.T
                distances = np.stack([r, frame.shape[0] - 1 - r, c, frame.shape[1] - 1 - c], 1)
                cell = int(np.argmin(distances.min(1)))
                last, seen_side = t, int(np.argmin(distances[cell]))
            elif 1 <= t - last <= STRATA[-1][1]:
                cls[t, k] = 1 + next(i for i, (lo, hi) in enumerate(STRATA) if lo <= t - last <= hi)
                side[t, k] = seen_side
    return cls, side


def departed(L):
    """Concatenate per-episode history targets; hidden deaths/identities never clear a sighting."""
    targets = [departed_episode(ep["grid"]) for ep in G.episode_grids(L)]
    return tuple(np.concatenate(x) for x in zip(*targets))


def frame_features(enc):
    """Per frame (current-percept control): periphery, glimpse, fixation one-hot, lighting, symlog reward, termination,
    previous action one-hot (used by the 8-step control only)."""
    rows = []
    for r in enc:
        n = len(r["p"])
        fix = F.one_hot((r["fix"][:, 0] - 1) * 7 + r["fix"][:, 1] - 1, 49).half()
        pa = F.one_hot(r["prev_action"].clamp(min=0), 17).half() * (r["prev_action"] >= 0)[:, None]
        rows.append(torch.cat([r["p"].reshape(n, -1), r["g"].reshape(n, -1), fix, r["v"], M.SYMLOG(r["reward"])[:, None].half(),
                               r["terminated"][:, None].half(), pa], 1))
    return torch.cat(rows)


def belief_targets(L):
    cls, side = departed(L)
    hud = np.minimum(np.concatenate([L["s_body"], L["s_inv"]], 1), 9).astype(np.int64)
    return {"ach": torch.as_tensor(L["s_ach"].astype(np.float32)), "hud": torch.as_tensor(hud),
            "dep": torch.as_tensor(cls), "side": torch.as_tensor(side)}


def belief_probe(Xtr, Ytr, ep_tr, Xdv, Ydv, ep_dv, seed, window=None, first_tr=None, first_dv=None):
    """Multi-task probe on per-frame features X [N, d]; window = 8 stacks frames t-7..t of the same episode (zeros
    before the episode start)."""
    held = ep_tr % 10 == 9
    tr_idx, va_idx = np.nonzero(~held)[0], np.nonzero(held)[0]
    Xg, Yg = Xtr.to(DEV), {k: v.to(DEV) for k, v in Ytr.items()}
    start_tr = torch.as_tensor(np.maximum.accumulate(np.where(first_tr, np.arange(len(first_tr)), 0)), device=DEV)
    idx_of = {False: torch.as_tensor(tr_idx, device=DEV), True: torch.as_tensor(va_idx, device=DEV)}

    def gather(X, start, rows):
        if window is None:
            return X[rows]
        lag = torch.arange(window - 1, -1, -1, device=rows.device)
        src = rows[:, None] - lag
        ok = src >= start[rows][:, None]
        return (X[src.clamp(min=0)] * ok[..., None]).flatten(1)

    mx = lambda i, val: gather(Xg, start_tr, idx_of[val][i])

    def lf(o, i, val):
        r = idx_of[val][i]
        side = Yg["side"][r]
        l = F.binary_cross_entropy_with_logits(o["ach"], Yg["ach"][r]) + \
            F.cross_entropy(o["hud"].reshape(-1, 10), Yg["hud"][r].reshape(-1)) + \
            F.cross_entropy(o["dep"].reshape(-1, 5), Yg["dep"][r].reshape(-1))
        if (side >= 0).any():
            l = l + F.cross_entropy(o["side"].reshape(-1, 4)[side.reshape(-1) >= 0], side.reshape(-1)[side.reshape(-1) >= 0])
        return l
    heads = {"ach": 22, "hud": 160, "dep": 4 * 5, "side": 4 * 4}
    model, log = fit(mx, lf, len(tr_idx), len(va_idx), heads, seed)
    del Xg
    torch.cuda.empty_cache()
    Xd = Xdv.to(DEV)
    start_dv = torch.as_tensor(np.maximum.accumulate(np.where(first_dv, np.arange(len(first_dv)), 0)), device=DEV)
    outs = {k: [] for k in heads}
    with torch.no_grad():
        for rows in torch.arange(len(Xdv), device=DEV).split(16384):
            o = model(gather(Xd, start_dv, rows))
            for k in heads:
                outs[k].append(o[k].float().cpu())
    o = {k: torch.cat(v) for k, v in outs.items()}
    del Xd
    torch.cuda.empty_cache()
    res = {"heldout_loss_by_epoch": log}
    ach_p = torch.sigmoid(o["ach"]).numpy(); ya = Ydv["ach"].numpy()
    res["achievements"] = [{"ap": ap(ya[:, i], ach_p[:, i]),
                            "balanced_accuracy": float(np.mean([((ach_p[:, i] > 0.5) == ya[:, i])[ya[:, i] == c].mean()
                                                               for c in (0, 1) if (ya[:, i] == c).any()]))} for i in range(22)]
    hp = o["hud"].reshape(-1, 16, 10).argmax(-1).numpy(); yh = Ydv["hud"].numpy()
    res["hud_exact"] = [G.estimate(hp[:, i] == yh[:, i], np.ones(len(yh), bool), ep_dv) for i in range(16)]
    dp = o["dep"].reshape(-1, 4, 5).softmax(-1).numpy(); yd = Ydv["dep"].numpy()
    sp = o["side"].reshape(-1, 4, 4).argmax(-1).numpy(); ys = Ydv["side"].numpy()
    res["departed"] = {kn: {"ap_by_stratum": {f"{lo}-{hi}": ap((yd[:, k] == s + 1).astype(int), dp[:, k, s + 1])
                                              for s, (lo, hi) in enumerate(STRATA)},
                            "last_seen_side_accuracy": G.estimate(sp[:, k] == ys[:, k], ys[:, k] >= 0, ep_dv)}
                       for k, kn in enumerate(G.KINDS)}
    return res


# ---- S1.3 one-step generation ----

@torch.no_grad()
def one_step(model, enc, cfg, n_tuples, rng):
    """Sample next percepts (and outcomes) for random (episode, t >= 1) tuples of dev, from their real histories."""
    e_cfg, s = cfg["edm"], cfg["edm"]["sampler"]
    P = cfg["data"]["context_prefix"]
    lens = np.array([len(r["p"]) for r in enc])
    eps_ = rng.choice(len(enc), n_tuples, p=(lens - 1) / (lens - 1).sum())
    ts = np.array([rng.integers(1, lens[e]) for e in eps_])
    dims = [tuple(enc[0][k].shape[1:]) for k in ("p", "g", "v")]
    clamp = cfg["perception"]["periphery"]["tanh_clamp"]
    n = s["steps"]
    sig = [(s["sigma_max"] ** (1 / s["rho"]) + i / (n - 1) * (s["sigma_min"] ** (1 / s["rho"]) - s["sigma_max"] ** (1 / s["rho"]))) ** s["rho"]
           for i in range(n)] + [0.0]
    gamma = min(s["s_churn"] / n, math.sqrt(2) - 1)
    gen = torch.Generator(device=DEV).manual_seed(0)
    out = {k: [] for k in ("p", "g", "v", "r", "term")}
    for chunk in np.array_split(np.arange(n_tuples), max(1, n_tuples // 64)):
        B = len(chunk)
        pad = lambda a: (torch.ones_like if a.dtype == torch.long and a.ndim == 2 else torch.zeros_like)(a[:1])   # fixations: a valid cell
        win = lambda key: torch.stack([torch.cat([pad(enc[eps_[i]][key]).expand(max(0, P - ts[i]), *enc[eps_[i]][key].shape[1:]),
                                                           enc[eps_[i]][key][max(0, ts[i] - P):ts[i] + 1]]) for i in chunk]).to(DEV)
        x0 = [win(k).float() for k in ("p", "g", "v")]
        valid = torch.stack([torch.as_tensor(np.r_[np.zeros(max(0, P - ts[i]), bool), np.ones(min(P, ts[i]) + 1, bool)]) for i in chunk]).to(DEV)
        prev_a, fix, bp = win("prev_action"), win("fix"), win("bp").float()
        episode = torch.zeros_like(prev_a)
        ctx = [a[:, :-1] + e_cfg["inference_context_sigma"] * torch.randn(a[:, :-1].shape, device=DEV, generator=gen) for a in x0]
        x = [sig[0] * torch.randn((B, 1, *d), device=DEV, generator=gen) for d in dims]
        mem = model.belief.memory(bp)
        for k in range(n):
            hat = sig[k] * (1 + gamma)
            x = [a + math.sqrt(hat ** 2 - sig[k] ** 2) * torch.randn(a.shape, device=DEV, generator=gen) for a in x]
            sg = torch.full((B, P + 1), e_cfg["inference_context_sigma"], device=DEV); sg[:, -1] = hat
            c_skip, c_out, c_in, c_noise = M.precondition(sg, e_cfg["sigma_data"])
            seq = [torch.cat([cx, a], 1) for cx, a in zip(ctx, x)]
            with TR.autocast(DEV, cfg):
                Fo = model.denoiser(tuple(M._bcast(c_in, a) * a for a in seq), c_noise, prev_a, mem, fix, episode, valid)
            Dn = M.clean_estimate(x, tuple(f[:, -1:] for f in Fo), c_skip[:, -1:], c_out[:, -1:], clamp)
            x = [a + (sig[k + 1] - hat) * (a - d) / hat for a, d in zip(x, Dn)]
        Pg = M.Percept(x[0][:, 0], x[1][:, 0], x[2][:, 0], fix[:, -1])
        with TR.autocast(DEV, cfg):
            h = model.outcome.features(bp[:, -1], model.belief.percept_embed(Pg))
            r = model.outcome.sample_reward(h.float(), gen)
            term = torch.rand(B, device=DEV, generator=gen) > torch.sigmoid(model.outcome.cont_logit(h, r).float())
        for k, v in zip(("p", "g", "v", "r", "term"), (Pg.p, Pg.g, Pg.v, r, term)):
            out[k].append(v.float().cpu())
    return eps_, ts, {k: torch.cat(v) for k, v in out.items()}


def conditioning_features(enc, eps_, ts, history):
    """Full generator conditioning: history percepts/signals/actions/fixations and every predicted belief.

    Keep the common context once in FP16; only classifier training/prediction batches expand to FP32.
    This covers the memory tokens via their pre-projection beliefs, including the current predicted belief.
    """
    features = [torch.cat([frame_features([r]), r["bp"]], 1).numpy() for r in enc]
    width = features[0].shape[1]
    rows = np.zeros((len(eps_), history * (width + 1) + enc[0]["bp"].shape[1] + 17 + 49), np.float16)
    for i, (e, t) in enumerate(zip(eps_, ts)):
        start = max(0, t - history)
        ctx = np.zeros((history, width + 1), np.float16)
        n = t - start
        if n:
            ctx[-n:, :width], ctx[-n:, -1] = features[e][start:t], 1
        act, fix = np.zeros(17, np.float16), np.zeros(49, np.float16)
        previous_action = int(enc[e]["prev_action"][t])
        if previous_action >= 0:
            act[previous_action] = 1
        fix[(int(enc[e]["fix"][t, 0]) - 1) * 7 + int(enc[e]["fix"][t, 1]) - 1] = 1
        rows[i] = np.concatenate([ctx.ravel(), enc[e]["bp"][t].numpy(), act, fix])
    return rows


def outcome_features(nxt):
    """Identical FP16 observation precision for real and generated classifier examples."""
    return torch.cat([nxt[k].reshape(len(nxt[k]), -1).half() for k in ("p", "g", "v")] +
                     [M.SYMLOG(nxt["r"]).half()[:, None], nxt["term"].half()[:, None]], 1).numpy()


def c2st(context, real, generated, ep, rng, corruption=None, context_blocks=None):
    """Episode-held-out neural C2ST; same MLP/fit schedule as the other probes.

    Train-only standardisation and equal expected energy per input group avoid scalar-count domination. Inputs
    are assembled in batches, never as a duplicated full FP32 matrix. Corruption donors stay within train/val/test.
    Trees on raw coordinates could not detect joint differences with exactly matched empirical marginals.
    """
    device = DEV if torch.cuda.is_available() else torch.device("cpu")
    blocks = list(context_blocks or [context.shape[1]])
    if sum(blocks) != context.shape[1] or min(blocks) <= 0:
        raise ValueError("Classifier input groups must cover the conditioning features")
    blocks += [real.shape[1]]
    te = G.episode_split(ep, ep, rng)[:len(ep)]
    train_episodes = np.unique(ep[~te])
    if len(train_episodes) < 2 or not te.any():
        raise ValueError("Classifier requires train, validation and test episodes")
    validation = rng.choice(train_episodes, max(1, len(train_episodes) // 10), replace=False)
    va = ~te & np.isin(ep, validation)
    partitions = [np.flatnonzero(~te & ~va), np.flatnonzero(va), np.flatnonzero(te)]
    donors = np.arange(len(ep))
    for rows in partitions:
        donors[rows] = rng.permutation(rows)

    def inputs(rows, synthetic):
        ctx, outcome = context[rows].copy(), real[rows].copy()
        if corruption == "shuffled_history":
            ctx[synthetic] = context[donors[rows[synthetic]]]
        elif corruption == "mismatched_actions":
            ctx[synthetic, -66:-49] = context[donors[rows[synthetic]], -66:-49]
        outcome[synthetic] = generated[rows[synthetic]]
        return np.concatenate([ctx, outcome], 1).astype(np.float32)

    n, sums = 0, np.zeros(context.shape[1] + real.shape[1])
    squares = np.zeros_like(sums)
    for rows in np.array_split(partitions[0], max(1, len(partitions[0]) // 512 + 1)):
        for synthetic in (False, True):
            x = inputs(rows, np.full(len(rows), synthetic))
            sums += x.sum(0, dtype=np.float64)
            squares += np.square(x).sum(0, dtype=np.float64)
            n += len(x)
    mean = torch.as_tensor(sums / n, dtype=torch.float32, device=device)
    scale = torch.as_tensor(np.maximum(squares / n - (sums / n)**2, 0)**.5,
                            dtype=torch.float32, device=device).clamp(min=1e-3)
    offset = 0
    for width in blocks:
        scale[offset:offset + width] *= math.sqrt(width * len(blocks) / len(scale))
        offset += width

    def mx(index, val):
        rows, idx = partitions[int(val)], index.cpu().numpy()
        x = inputs(rows[idx % len(rows)], idx >= len(rows))
        return (torch.as_tensor(x, device=device) - mean) / scale

    def loss(output, index, val):
        return F.cross_entropy(output["domain"], (index >= len(partitions[int(val)])).long())

    model, log = fit(mx, loss, len(partitions[0]) * 2, len(partitions[1]) * 2, {"domain": 2}, 0, device)
    correct, count = 0, 0
    with torch.no_grad():
        for rows in np.array_split(partitions[2], max(1, len(partitions[2]) // 512 + 1)):
            for synthetic in (False, True):
                x = torch.as_tensor(inputs(rows, np.full(len(rows), synthetic)), device=device)
                prediction = model((x - mean) / scale)["domain"].argmax(1)
                correct += int((prediction == int(synthetic)).sum())
                count += len(rows)
    return {"accuracy": correct / count, "n_test": count, "conditioning_features": context.shape[1],
            "outcome_features": real.shape[1], "precision": "float16_both_domains", "classifier": "MLP256x2",
            "feature_group_widths": blocks, "heldout_loss_by_epoch": log}


def first_episodes(L, n):
    if not n:
        return L
    s_end, t_end = int((L["ep_T"][:n] + 1).sum()), int(L["ep_T"][:n].sum())
    return {k: (v[:n] if k.startswith("ep_") else v[:s_end] if k.startswith("s_") else v[:t_end]) for k, v in L.items()}


def main():
    ckpt, out_path = Path(sys.argv[1]), Path(sys.argv[2])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    max_dev = int(sys.argv[3]) if len(sys.argv) > 3 else None
    max_tr = int(sys.argv[4]) if len(sys.argv) > 4 else None             # smoke tests only
    ck = torch.load(ckpt, weights_only=False, map_location="cpu")
    cfg, seed = ck["cfg"], ck["world_seed"]
    model = M.M0(cfg).to(DEV); model.load_state_dict(ck["model"]); model.eval()
    torch.manual_seed(seed)
    untrained = M.M0(cfg).perception.to(DEV).eval()                     # the same initialisation as the run
    Ltr = dict(np.load(LABELS / "s0_train400_labels.npz"))
    Ldv = dict(np.load(LABELS / "s0_dev_labels.npz"))
    Ldv, Ltr = first_episodes(Ldv, max_dev), first_episodes(Ltr, max_tr)
    res = {"checkpoint": str(ckpt), "checkpoint_sha256": TR.file_sha256(ckpt), "world_seed": seed,
           "frames": ck["state"]["frames"], "evaluation_version": 3,
           "departed_target": "visible_kind_history_v2", "side_target": "nearest_edge_at_last_visible_sighting"}
    del ck
    enc_tr, enc_dv = encode(model, Ltr, cfg, untrained), encode(model, Ldv, cfg, untrained)
    ep_tr = np.concatenate([np.full(len(r["p"]), e) for e, r in enumerate(enc_tr)])
    ep_dv = np.concatenate([np.full(len(r["p"]), e) for e, r in enumerate(enc_dv)])

    # S1.1
    res["s1_1"] = {}
    for name, kp_, kv_ in (("trained", "p", "v"), ("untrained", "p_u", "v_u")):
        tr, dv = cell_table(enc_tr, Ltr, kp_, kv_), cell_table(enc_dv, Ldv, kp_, kv_)
        res["s1_1"][name] = []
        for s in (0, 1):
            probe, kp, log = token_probe(tr, dv, s)
            res["s1_1"][name].append({"heldout_loss_by_epoch": log, **retention(dv, kp)})
            if name == "trained" and s == 0:
                reader = probe
                res["reader"] = save_reader(reader, out_path.with_suffix(".reader.pt"), ckpt, seed, cfg)
        print("s1.1", name, json.dumps({k: v for k, v in res["s1_1"][name][0].items() if k == "night_r1_2"}), flush=True)
    res["s3_perception_blocker"] = perception_blocker(res["s1_1"]["trained"], res["s1_1"]["untrained"])

    # S1.2
    Ytr, Ydv = belief_targets(Ltr), belief_targets(Ldv)
    first_tr = np.concatenate([np.r_[True, np.zeros(len(r["p"]) - 1, bool)] for r in enc_tr])
    first_dv = np.concatenate([np.r_[True, np.zeros(len(r["p"]) - 1, bool)] for r in enc_dv])
    Ftr, Fdv = frame_features(enc_tr), frame_features(enc_dv)
    cur = Ftr.shape[1] - 17                                             # the current-percept control has no action
    res["s1_2"] = {}
    for name, Xt, Xd, w in (("belief", torch.cat([r["b"] for r in enc_tr]), torch.cat([r["b"] for r in enc_dv]), None),
                            ("control_current_percept", Ftr[:, :cur], Fdv[:, :cur], None),
                            ("control_last_8_steps", Ftr, Fdv, cfg["data"]["context_prefix"])):
        res["s1_2"][name] = [belief_probe(Xt, Ytr, ep_tr, Xd, Ydv, ep_dv, s, w, first_tr, first_dv) for s in (0, 1)]
        print("s1.2", name, json.dumps(res["s1_2"][name][0]["departed"]), flush=True)
    del Ftr, Fdv

    # S1.3
    rng = np.random.default_rng(0)
    n_tuples = min(30000, int(sum(len(r["p"]) - 1 for r in enc_dv)))
    eps_, ts, gen = one_step(model, enc_dv, cfg, n_tuples, rng)
    real = {k: torch.stack([enc_dv[e][kk][t].float() for e, t in zip(eps_, ts)]) for k, kk in (("p", "p"), ("g", "g"), ("v", "v"))}
    real["r"] = torch.tensor([float(enc_dv[e]["reward"][t]) for e, t in zip(eps_, ts)])
    real["term"] = torch.tensor([bool(enc_dv[e]["terminated"][t]) for e, t in zip(eps_, ts)])
    context = conditioning_features(enc_dv, eps_, ts, cfg["data"]["context_prefix"])
    width = cfg["belief"]["width"]
    blocks = [context.shape[1] - width - 17 - 49, width, 17, 49]
    Xr, Xg = outcome_features(real), outcome_features(gen)
    res["s1_3"] = {"c2st_generated": c2st(context, Xr, Xg, eps_, np.random.default_rng(1), context_blocks=blocks), "power": {}}
    prev = {k: torch.stack([enc_dv[e][k][t - 1].float() for e, t in zip(eps_, ts)]) for k in ("p", "g", "v")}
    for kind, nxt in (("shuffled_history", real), ("frozen_time", {**real, **prev}), ("mismatched_actions", real)):
        res["s1_3"]["power"][kind] = c2st(context, Xr, outcome_features(nxt), eps_, np.random.default_rng(1),
                                          corruption=kind, context_blocks=blocks)
    del context, Xr, Xg
    with torch.no_grad():
        read = lambda p, v: torch.cat([reader(torch.cat([p[i][:, :63].reshape(-1, p.shape[-1]).half().to(DEV),
                                                         v[i].repeat_interleave(63, 0).half().to(DEV)], 1))["kind"].argmax(1).reshape(len(i), 63).cpu()
                                       for i in torch.arange(len(p)).split(2048)])
        kr, kg = read(real["p"], real["v"]).numpy()[:, KEEP[:63]], read(gen["p"], gen["v"]).numpy()[:, KEEP[:63]]
    one = np.ones(len(eps_), bool)
    res["s1_3"]["reader_counts"] = {kn: {"real": G.estimate((kr == k).sum(1), one, eps_), "generated": G.estimate((kg == k).sum(1), one, eps_)}
                                    for k, kn in enumerate(G.KINDS, 1)}
    from saccade.eval import calib as C
    y_r = np.concatenate([r["reward"][1:].numpy() for r in enc_dv])
    dh = np.concatenate([np.diff(Ldv["s_body"][a:b, 0].astype(float)) for a, b in zip(np.r_[0, np.cumsum(Ldv["ep_T"] + 1)][:-1], np.cumsum(Ldv["ep_T"] + 1))])
    unlock = np.round(y_r - 0.1 * dh).astype(int)
    ept = np.concatenate([np.full(len(r["p"]) - 1, e) for e, r in enumerate(enc_dv)])
    p_true = np.concatenate([r["p_true"][1:].numpy() for r in enc_dv]); p_unlock = np.concatenate([r["p_unlock"][1:].numpy() for r in enc_dv])
    p_cont = np.concatenate([r["p_cont"][1:].numpy() for r in enc_dv]); term = np.concatenate([r["terminated"][1:].numpy() for r in enc_dv])
    res["s1_3"]["calibration"] = {"reward": C.score(p_true, p_unlock, unlock.astype(float), ept),
                                  "unlock_event": "round(reward - 0.1 * observed_health_change) == 1",
                                  "continuation": C.score(np.where(term, 1 - p_cont, p_cont), 1 - p_cont, term.astype(float), ept),
                                  "references": json.loads((LABELS / "s0_calib.json").read_text())}

    # S1.4
    tsp = np.concatenate([r["tsp"][1:].numpy() for r in enc_dv])
    night = np.concatenate([Ldv["s_light"][a + 1:b] < 0.5 for a, b in zip(np.r_[0, np.cumsum(Ldv["ep_T"] + 1)][:-1], np.cumsum(Ldv["ep_T"] + 1))])
    res["s1_4"] = {"tsp_nats_per_pixel": {"all": G.estimate(tsp, np.ones(len(tsp), bool), ept), "day": G.estimate(tsp, ~night, ept),
                                          "night": G.estimate(tsp, night, ept)},
                   "baselines": tsp_baselines(enc_tr, enc_dv, Ltr, Ldv)}
    out_path.write_text(json.dumps(res, indent=1))
    print("saved", out_path, flush=True)


@torch.no_grad()
def tsp_baselines(enc_tr, enc_dv, Ltr, Ldv):
    """Copy-previous (logistic at the previous frame's pixels at the same crop) and per-position mean (logistic at the
    train-400 mean pixel of each screen position); each scale fitted by maximum likelihood on train-400 (grid on log s)."""
    mean = torch.zeros(63, 63, 3, dtype=torch.float64); n = 0
    for ep in episodes(Ltr):
        fr = torch.as_tensor(D.RP.episode_fields(ep.shard, ep.slot)["observations"].numpy()).double()
        mean += fr.sum(0); n += len(fr)
    mean = (mean / n).round().clamp(0, 255).to(torch.uint8)

    def nll(L, enc, log_s, prev):
        tot, cnt = np.zeros(len(log_s)), 0
        for e, ep in enumerate(episodes(L)):
            fr = torch.as_tensor(D.RP.episode_fields(ep.shard, ep.slot)["observations"].numpy())
            fix = enc[e]["fix"][1:]
            cur = M.cell_pixels(M.crop(fr[1:], fix))
            ref = M.cell_pixels(M.crop(fr[:-1] if prev else mean.expand(len(fix), 63, 63, 3), fix))
            mu = ref.float() / 127.5 - 1
            for i, ls in enumerate(log_s):
                tot[i] += float(M.pixel_nll(mu, torch.full_like(mu, ls), cur).sum())
            cnt += cur.numel()
        return tot / cnt
    grid = np.linspace(-6, 0, 25)
    out = {}
    for name, prev in (("copy_previous", True), ("position_mean", False)):
        tr = nll(Ltr, enc_tr, grid, prev)
        best = float(grid[int(np.argmin(tr))])
        out[name] = {"log_scale": best, "train_nats_per_pixel": float(tr.min()),
                     "dev_nats_per_pixel": float(nll(Ldv, enc_dv, [best], prev)[0])}
    return out


if __name__ == "__main__":
    main()
