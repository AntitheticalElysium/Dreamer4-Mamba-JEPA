"""Can the gaze forecast teach Mamba the pixel clocks? (PLAN.md) The 11-14 learnability instrument
(`../20261114_learnability/learnable.py`: F6 checkpoint 10, perception frozen, 128-step windows from the zero state,
the same window and fixation draws as its arms) with the gaze forecast as the only loss: the predicted belief before
frame t, beside the chooser's own 8-frame inputs, forecasts F6's frozen final chooser's 49-window distribution at t.
Simulator labels are used only to read the result (probes, cohorts, exact-history banks), never in a loss.

    MAMBA_DETERMINISTIC=1 python -m artifacts.experiments.20261115_gaze_learnability.gaze gpu      (references, trained)
    MAMBA_DETERMINISTIC=1 GAZE_CPU=1 python -m artifacts.experiments.20261115_gaze_learnability.gaze summary
    (each step is skipped when its output exists)
"""
import copy
import importlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from saccade import chooser as C, data as D, model as M, train as TR
from saccade.eval import gates as GT, s1 as S

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
OUT = Path(os.environ.get("GAZE_OUT", HERE))                           # the smoke writes elsewhere
RUN = OUT / "run"
LB = importlib.import_module("artifacts.experiments.20261114_learnability.learnable")
FR = importlib.import_module("artifacts.experiments.20261115_gaze_forecast_eda.fair")
CR = importlib.import_module("artifacts.experiments.20261111_p1f5.credit")
TJ = CR.TJ
SMOKE = os.environ.get("GAZE_SMOKE") == "1"                            # CPU code-path check only (PLAN.md, "Smoke")
if SMOKE or os.environ.get("GAZE_CPU") == "1":                          # the summary needs no GPU
    for mod in (LB, CR, CR.NP.IF):
        mod.DEV = torch.device("cpu")
DEV = LB.DEV
H, W, B = 8, (16 if SMOKE else LB.W), (2 if SMOKE else LB.B)
UPDATES, EVALS = (3, (3,)) if SMOKE else (LB.UPDATES, LB.EVALS)
READ = EVALS if SMOKE else (1500, 3000)                                 # the verdict's reading points
BANKS = {"energy": (66, 2), "hunger": (64, 0), "recovery": (63, 4)}    # exact-history bank -> (digit cell, event column)
AP = lambda y, s: (lambda a: float("nan") if a is None else a)(GT.average_precision(y, s))   # None: no positives


class Gaze(FR.Forecaster):
    """fair.py's forecaster of the chooser's distribution from the chooser's own inputs (its head's trunk, a score per
    cell, a window's logit the sum of its nine cells' plus a bias), with one more input added to every cell, as
    oracle.py's exact phase: the predicted belief before the frame."""

    def __init__(self, width):
        super().__init__()
        self.belief = nn.Linear(width, 64)

    def forward(self, xx, b):
        ph, ctx, act = xx
        t = self.trunk
        x = t.local(ph.transpose(1, 2).flatten(2)) + (t.context(ctx) + t.action(act) + self.belief(b))[:, None] + t.position
        return self.score(t.blocks(x))[..., 0][:, C.WIN].sum(-1) + self.bias


def load_chooser():
    state = torch.load(EXP / "20261113_p1f6/run/chooser_last.pt", map_location="cpu", weights_only=False)
    pol = dict(state["config"])
    assert pol["history"] == H
    head = C.Head(H)
    head.load_state_dict(state["head"])
    for p in head.parameters():
        p.requires_grad_(False)
    return head.to(DEV).eval(), pol


@torch.no_grad()
def teacher(model, chooser, pol, full, every=None):
    """The chooser's inputs and its window distribution after the first H frames of `full` (gather arrays), computed as
    in training and in the gaze EDA (fp32), flattened over (row, frame); with `every` = W, only frames 1.. of each run
    of W (frame 0 of a window has no predicted belief; an episode's frame 0 has no previous action and is not scored)."""
    p, v = C.coarse(model.perception, torch.as_tensor(full["frames"], device=DEV))
    xx, y = C.inputs(p, v, full, H, H)
    if every is not None:
        keep = torch.arange(len(y), device=DEV) % every >= 1
        xx, y = tuple(z[keep] for z in xx), y[keep]
    q = []
    for a in range(0, len(y), 1024):
        x = tuple(z[a:a + 1024] for z in xx)
        taken, noop = chooser(x), chooser((x[0], x[1], torch.zeros_like(x[2])))
        c = (taken - noop).square().sum(-1)[:, C.WIN].sum(-1)
        s = (y[a:a + 1024] - taken).square().sum(-1)[:, C.WIN].sum(-1)
        q.append(C.policy(c, s, pol))
    return xx, torch.cat(q)


def windows(rng, Ltr, eps, long_eps):
    """learnable.main's draws, in its order (episodes, starts, a uniform prior fixation per frame), plus the H frames
    before each window (no draw; the chooser's history only)."""
    es = rng.choice(long_eps, B)
    starts = [int(rng.integers(0, Ltr["ep_T"][e] - W + 1)) for e in es]
    recs = [[(int(e), int(e), s + t, *D.fixation_prior(rng, ())) for t in range(W)] for e, s in zip(es, starts)]
    pre = [[None] * max(0, H - s) + [(int(e), int(e), t, 1, 1) for t in range(max(0, s - H), s)] for e, s in zip(es, starts)]
    full = D.gather(eps, [a + b for a, b in zip(pre, recs)])
    return full, TR.tensors({k: v[:, H:] for k, v in full.items()}, DEV)


def cohorts(L):
    """Per state t (transition t-1 -> t), masks [N, 5] over learnable.EVENTS, named by what they select (Codex 11-15:
    they remove particular cues, not every correlate of elapsed time). steady8: eligible and the last 8 frames (t-8 ..
    t-1) in the eligible sleep state with the event's digit unchanged. after_reset: steady8 and, for food / drink, an
    intake rise of that digit or a sleep earlier in the episode; for the awake energy drop, a sleep earlier in the
    episode; for the asleep energy rise, WAKE_UP already earned (Codex's repeat-sleep cohort); recovery: steady8."""
    _, ok = LB.events(L)
    body, sleep, wake = L["s_body"].astype(int), L["s_sleep"].astype(bool), L["s_ach"][:, 15].astype(bool)
    off = np.r_[0, np.cumsum(L["ep_T"] + 1)]
    n = len(body)
    t = np.arange(n) - np.repeat(off[:-1], L["ep_T"] + 1)
    digit = (1, 2, 3, 3, 0)
    steady = ok & (t >= H)[:, None]
    for j, d in enumerate(digit):
        for k in range(2, H + 1):
            i = np.flatnonzero(steady[:, j])
            same = (body[i - k, d] == body[i - 1, d]) & (sleep[i - k] == sleep[i - 1])
            steady[i[~same], j] = False
    rise = np.zeros_like(body, bool)
    rise[1:] = body[1:] > body[:-1]
    rise[off[:-1]] = False
    before = lambda x: np.concatenate([np.maximum.accumulate(x[a:b]) for a, b in zip(off[:-1], off[1:])])
    slept, ate, drank = (np.r_[False, before(z)[:-1]] for z in (sleep, rise[:, 1], rise[:, 2]))   # at a state < t
    first = np.zeros(n, bool); first[off[:-1]] = True
    slept[first], ate[first], drank[first] = False, False, False
    after = steady.copy()
    after[:, 0] &= ate | slept
    after[:, 1] &= drank | slept
    after[:, 2] &= slept
    after[:, 3] &= np.r_[False, wake[:-1]] & ~first
    return {"steady8": steady, "after_reset": after}


def probe(X, y, ok, fold, offset=None):
    """learnable.probe (one linear logistic probe per event on the standardised belief, fit on folds 0-3, L2 1e-3,
    LBFGS), optionally with a fixed per-row logit offset (the time-and-light reader); per event the held-out logits
    and the probe (weights, bias, standardisation) for reuse on the banks."""
    mu, sd = X[fold < 4].mean(0), X[fold < 4].std(0).clamp(min=1e-6)
    Z = (X - mu) / sd
    out = {}
    for j, n in enumerate(LB.EVENTS):
        tr, te = (fold < 4) & ok[:, j], (fold == 4) & ok[:, j]
        Xt, yt = Z[tr].to(DEV), torch.as_tensor(y[tr, j], dtype=torch.float32, device=DEV)
        ot = torch.zeros(len(yt), device=DEV) if offset is None else torch.as_tensor(offset[tr, j], dtype=torch.float32, device=DEV)
        w = torch.zeros(X.shape[1] + 1, device=DEV, requires_grad=True)
        opt = torch.optim.LBFGS([w], max_iter=200, line_search_fn="strong_wolfe")

        def closure():
            opt.zero_grad()
            l = F.binary_cross_entropy_with_logits(ot + Xt @ w[:-1] + w[-1], yt) + 1e-3 * w[:-1].square().sum()
            l.backward()
            return l
        opt.step(closure)
        oe = 0 if offset is None else torch.as_tensor(offset[te, j], dtype=torch.float32)
        w = w.detach().cpu()
        out[n] = {"logit": (oe + Z[te] @ w[:-1] + w[-1]).numpy(), "w": w, "mu": mu, "sd": sd}
    return out


def time_light(L, y, ok, fold):
    """Codex's adversarial reader (NOTEBOOK 11-15): LightGBM on the episode step and the previous frame's lighting only
    (31 leaves, learning rate 0.06, min leaf 50, 200 rounds, seed 0). Per event, logits over all states: fold-4 rows
    from the fit on folds 0-3, folds 0-3 rows cross-fitted by fold (so a probe fitted on top never sees a reader
    trained on its own rows)."""
    import lightgbm as lgb
    off = np.r_[0, np.cumsum(L["ep_T"] + 1)]
    X = np.stack([np.arange(len(fold)) - np.repeat(off[:-1], L["ep_T"] + 1), np.roll(L["s_light"], 1)], -1)
    par = dict(objective="binary", learning_rate=.06, num_leaves=31, min_data_in_leaf=50, num_threads=2, verbosity=-1, seed=0)
    z = np.zeros(y.shape, np.float32)
    for j in range(y.shape[1]):
        for held in range(5):
            fit = ok[:, j] & (fold < 4) & (fold != held)
            pred = ok[:, j] & (fold == held)
            if not pred.any() or len(np.unique(y[fit, j])) < 2:             # only in the smoke's 10 episodes
                continue
            m = lgb.train(par, lgb.Dataset(X[fit], y[fit, j].astype(float)), num_boost_round=200)
            z[pred, j] = m.predict(X[pred], raw_score=True)
    return z


@torch.no_grad()
def bank_read(model, cfg, chooser, pol, belief, head=None):
    """F6's exact-history banks (the last 8 frames, actions and outcomes identical across the two branches, the same
    episode step; the older history makes one branch due): per pair, due branch first, the predicted belief before the
    target frame, and with a head the forecast probability that the chosen crop covers the bank's digit, with the
    chooser's own (which reads the target frame)."""
    res, beliefs = {}, {}
    for name, (cell, _) in BANKS.items():
        cover = (C.WIN == cell).any(-1).to(DEV)
        rows, bs = [], []
        for row in TJ.pair_rows(name)[:2 if SMOKE else None]:
            i, e, prefix, frames, rewards, terms, q, truth = row
            assert q >= H and np.array_equal(frames[0, q - H + 1:q + 1], frames[1, q - H + 1:q + 1])
            changed = truth[:, 1] != truth[:, 0]
            assert changed.sum() == 1, truth
            due = int(np.flatnonzero(changed)[0])
            order = [due, 1 - due]
            p, rw, td, a = CR.belief_inputs(model, cfg, row)
            with TR.autocast(DEV, cfg):
                y, _ = belief(belief.tokens(belief.percept_embed(p), rw, td, a), belief.initial(2, DEV))
            b = y[:, -1].float()
            bs.append(b[order].cpu())
            r = {"e": int(e), "due_branch": due, "direction": int(np.sign(truth[due, 1] - truth[due, 0]))}
            if head is not None:
                clip = {"frames": frames[:, q - H + 1:q + 2], "episode": np.ones((2, H + 1), np.int64),
                        "valid": np.ones((2, H + 1), bool), "prev_action": np.zeros((2, H + 1), np.int64),
                        "reward": rewards[:, q - H:q + 1].astype(np.float32), "terminated": terms[:, q - H:q + 1].astype(bool)}
                xx, qt = teacher(model, chooser, pol, clip)
                assert all(torch.equal(z[0], z[1]) for z in xx)
                f = head(xx, b).softmax(-1)
                r.update(forecast=[float(f[k, cover].sum()) for k in order], teacher=[float(qt[k, cover].sum()) for k in order])
            rows.append(r)
        res[name], beliefs[name] = rows, torch.stack(bs).numpy()
    return res, beliefs


def setup():
    torch.manual_seed(0)
    ck, model, cfg, _ = LB.DZ.load(10)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    if SMOKE:
        for m in model.belief.mixers:
            m.backend = "reference"
    chooser, pol = load_chooser()
    Ldev = dict(np.load(S.LABELS / "s0_dev_labels.npz"))
    if SMOKE:                                                            # the first 10 DEV episodes
        n = int(np.r_[0, np.cumsum(Ldev["ep_T"] + 1)][10])
        Ldev = {k: (v[:10] if k.startswith("ep_") else v[:n] if k.startswith("s_") else v[:n - 10]) for k, v in Ldev.items()}
    ydev, okdev = LB.events(Ldev)
    fold = np.repeat(np.arange(len(Ldev["ep_T"])) % 5, Ldev["ep_T"] + 1)
    return ck, model, cfg, chooser, pol, Ldev, ydev, okdev, fold, cohorts(Ldev)


def references(env):
    """A0: F6's checkpoint-10 belief, unchanged. Its predicted beliefs on all DEV and on the banks are saved for the
    summary; its natural probe APs must reproduce reference.json (the instrument's known answer)."""
    ck, model, cfg, chooser, pol, Ldev, ydev, okdev, fold, masks = env
    X = LB.encode_dev(model, model.belief, cfg, Ldev)
    s = probe(X, ydev, okdev, fold)
    known = json.loads((EXP / "20261114_learnability/reference.json").read_text())["probe"]
    res = {n: [AP(ydev[(fold == 4) & okdev[:, j], j], s[n]["logit"]), known[n]["ap"]] for j, n in enumerate(LB.EVENTS)}
    print("A0 known answer", json.dumps(res), flush=True)
    if not SMOKE:
        assert all(abs(a - b) < 2e-3 for a, b in res.values()), res
    np.save(RUN / "X_A0.npy", X.half().numpy())
    _, beliefs = bank_read(model, cfg, chooser, pol, model.belief)
    np.savez(RUN / "banks_A0.npz", **beliefs)
    (OUT / "references.json").write_text(json.dumps({"checkpoint_sha256": TR.file_sha256(ck), "known_answer": res}, indent=1) + "\n")


def trained(env):
    """F6's belief trained by the gaze forecast alone. Same draws, schedule and optimiser as learnable.main (Adam, head
    1e-3, belief 1e-4, 100 warm-up updates, clip 1, the head initialised after torch.manual_seed(2))."""
    ck, model, cfg, chooser, pol, Ldev, ydev, okdev, fold, masks = env
    Ltr = dict(np.load(S.LABELS / "s0_train_labels.npz"))
    eps = S.episodes(Ltr)
    long_eps = np.flatnonzero(Ltr["ep_T"] >= W)
    belief = copy.deepcopy(model.belief)
    belief.train()
    for p in belief.parameters():
        p.requires_grad_(True)
    torch.manual_seed(2)
    head = Gaze(cfg["belief"]["width"]).to(DEV)
    groups = [{"params": list(head.parameters()), "lr": 1e-3}, {"params": list(belief.parameters()), "lr": 1e-4}]
    opt = torch.optim.Adam(groups)
    base_lr = [g["lr"] for g in groups]
    rng = np.random.default_rng(7)                                     # learnable.main's draws
    res = {"arm": "trained", "checkpoint_sha256": TR.file_sha256(ck), "window": W, "batch": B, "evals": {}, "log": []}
    t0, losses = time.time(), []
    for u in range(1, UPDATES + 1):
        full, chunk = windows(rng, Ltr, eps, long_eps)
        with torch.no_grad(), TR.autocast(DEV, cfg):
            P = model.perception(chunk["frames"], chunk["fixation"])
        P = M.Percept(*(z.float() for z in P.parts()), P.fix)
        xx, q = teacher(model, chooser, pol, full, every=W)            # window frames 1..W-1
        with TR.autocast(DEV, cfg):
            b = LB.run_belief(belief, P, chunk).float()               # the predicted belief before each of those frames
        loss = -(q * head(xx, b.reshape(-1, b.shape[-1])).log_softmax(-1)).sum(-1).mean()
        f = min(1.0, u / 100)
        for g, lr in zip(opt.param_groups, base_lr):
            g["lr"] = lr * f
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_([p for g in groups for p in g["params"]], 1.0)
        opt.step()
        losses.append(float(loss))
        if u == 1 and DEV.type == "cuda":
            res["peak_gib_first_update"] = torch.cuda.max_memory_allocated() / 2 ** 30
            print("peak GiB after update 1", res["peak_gib_first_update"], flush=True)
        if u % 250 == 0 or SMOKE:
            res["log"].append({"update": u, "loss": float(np.mean(losses[-250:])), "sec_per_update": (time.time() - t0) / u})
            print("trained", res["log"][-1], flush=True)
        if u in EVALS:
            head.eval(); belief.eval()
            X = LB.encode_dev(model, belief, cfg, Ldev)
            np.save(RUN / f"X_trained_{u}.npy", X.half().numpy())
            s = probe(X, ydev, okdev, fold)                            # monitoring only; the summary refits on CPU
            rows, beliefs = bank_read(model, cfg, chooser, pol, belief, head)
            np.savez(RUN / f"banks_trained_{u}.npz", **beliefs)
            res["evals"][u] = {"natural_ap": {n: AP(ydev[(fold == 4) & okdev[:, j], j], s[n]["logit"]) for j, n in enumerate(LB.EVENTS)},
                               "banks": rows}
            torch.save({"belief": belief.state_dict(), "head": head.state_dict()}, RUN / f"trained_{u}.pt")
            print("trained", u, json.dumps(res["evals"][u]["natural_ap"]), flush=True)
            head.train(); belief.train()
    (OUT / "trained.json").write_text(json.dumps(res, indent=1) + "\n")


def summary(env):
    """The fixed readings (PLAN.md), on CPU from the saved beliefs: episode-bootstrap intervals over held-out DEV
    episodes, pair-bootstrap intervals over bank pairs (10,000 draws, seed 215)."""
    _, _, _, _, _, Ldev, ydev, okdev, fold, masks = env
    tr = json.loads((OUT / "trained.json").read_text())
    a1 = {int(l.split()[1]): json.loads(l.split(" ", 2)[2]) for l in (EXP / "20261114_learnability/events.log").read_text().splitlines()
          if l.startswith("events ") and l.split()[1].isdigit()}
    ref = json.loads((EXP / "20261114_learnability/reference.json").read_text())["probe"]
    ep = np.repeat(np.arange(len(Ldev["ep_T"])), Ldev["ep_T"] + 1)
    rng = np.random.default_rng(215)
    load = lambda tag: torch.as_tensor(np.load(RUN / f"X_{tag}.npy")).float()
    tags = ["A0"] + [f"trained_{u}" for u in EVALS]
    sets = {"natural": okdev, **masks}
    tl = {m: time_light(Ldev, ydev, ok, fold) for m, ok in sets.items()}
    out = {"time_light_ap": {m: {n: AP(ydev[(fold == 4) & ok[:, j], j], tl[m][(fold == 4) & ok[:, j], j])
                                 for j, n in enumerate(LB.EVENTS)} for m, ok in sets.items()},
           "screen_ap": {}, "beyond_time_light": {}, "banks": {}, "head_banks": {}, "verdict": {}}
    nll = lambda z, yy: F.binary_cross_entropy_with_logits(torch.as_tensor(z), torch.as_tensor(yy, dtype=torch.float32), reduction="none").numpy()
    per_row, natural = {}, {}
    for tag in tags:
        X = load(tag)
        natural[tag] = probe(X, ydev, okdev, fold)
        for m, ok in sets.items():
            s = probe(X, ydev, ok, fold, offset=tl[m])
            for j, n in enumerate(LB.EVENTS):
                te = (fold == 4) & ok[:, j]
                per_row[tag, m, n] = nll(s[n]["logit"], ydev[te, j])
        print("probes", tag, flush=True)
    for j, n in enumerate(LB.EVENTS):
        te = (fold == 4) & okdev[:, j]
        out["screen_ap"][n] = {"A0": AP(ydev[te, j], natural["A0"][n]["logit"]), "A1": {u: a1[u][n]["ap"] for u in sorted(a1)},
                               "threshold": ref[n]["ap"] + 0.25 * (a1[1500][n]["ap"] - ref[n]["ap"]),
                               **{u: AP(ydev[te, j], natural[f"trained_{u}"][n]["logit"]) for u in EVALS}}

    def episode_ci(d, e):
        ids, inv = np.unique(e, return_inverse=True)
        if not len(ids):
            return [float("nan")] * 3
        boot = rng.integers(len(ids), size=(10000, len(ids)))
        ns = np.bincount(inv, minlength=len(ids))[boot].sum(1)
        bs = np.bincount(inv, weights=d, minlength=len(ids))[boot].sum(1)
        return [float(d.mean()), *np.quantile(bs[ns > 0] / ns[ns > 0], [.025, .975]).tolist()]
    for m, ok in sets.items():
        out["beyond_time_light"][m] = {}
        for j, n in enumerate(LB.EVENTS):
            te = (fold == 4) & ok[:, j]
            base = nll(tl[m][te, j], ydev[te, j])
            row = {"rows": int(te.sum()), "events": int(ydev[te, j].sum()),
                   "A0_gain_over_time_light": float((base - per_row["A0", m, n]).mean())}
            for u in EVALS:
                row[u] = {"gain_over_time_light": float((base - per_row[f"trained_{u}", m, n]).mean()),
                          "trained_minus_A0": episode_ci(per_row["A0", m, n] - per_row[f"trained_{u}", m, n], ep[te])}
            out["beyond_time_light"][m][n] = row

    def separation(tag, bank, j):
        z = torch.as_tensor(np.load(RUN / f"banks_{tag}.npz")[bank]).float()       # [pairs, 2 (due, delayed), width]
        p = natural[tag][LB.EVENTS[j]]
        s = ((z - p["mu"]) / p["sd"]) @ p["w"][:-1]
        return (s[:, 0] - s[:, 1]).numpy()                                         # probe logit, due minus delayed
    for bank, (_, j) in BANKS.items():
        a0 = separation("A0", bank, j)
        out["banks"][bank] = {"pairs": len(a0), "A0_separation": float(a0.mean())}
        for u in EVALS:
            d = separation(f"trained_{u}", bank, j) - a0
            bs = d[rng.integers(len(d), size=(10000, len(d)))].mean(1)
            out["banks"][bank][u] = {"separation": float((d + a0).mean()),
                                     "trained_minus_A0": [float(d.mean()), *np.quantile(bs, [.025, .975]).tolist()]}
            rows = tr["evals"][str(u)]["banks"][bank]
            f, t = np.array([r["forecast"] for r in rows]), np.array([r["teacher"] for r in rows])
            bern = lambda a, b: a * np.log(a / b) + (1 - a) * np.log((1 - a) / (1 - b))
            out["head_banks"].setdefault(bank, {})[u] = {"forecast_due_delayed": f.mean(0).tolist(), "teacher_due_delayed": t.mean(0).tolist(),
                                                        "coverage_kl_to_teacher_due_delayed": bern(t, f).mean(0).tolist()}

    for u in READ:                                                       # the bars (PLAN.md)
        out["verdict"][u] = {}
        for n in LB.EVENTS[:4]:
            screen = out["screen_ap"][n][u] >= out["screen_ap"][n]["threshold"]
            beyond = out["beyond_time_light"]["natural"][n][u]["trained_minus_A0"][1] > 0
            bank = {"energy_down_awake": "energy", "food_down": "hunger"}.get(n)
            phase = None if bank is None else out["banks"][bank][u]["trained_minus_A0"][1] > 0
            label = ("phase taught" if screen and beyond and phase else
                     "event readability beyond time and light" if screen and beyond else "not taught")
            out["verdict"][u][n] = {"screen": bool(screen), "beyond_time_light": bool(beyond), "exact_pairs": phase, "reading": label}
    last = out["verdict"][READ[-1]]
    out["verdict"]["overall"] = ("drop the target" if not any(last[n]["beyond_time_light"] for n in LB.EVENTS[:4]) else
                                 "for the user: " + "; ".join(f"{n} {last[n]['reading']}" for n in LB.EVENTS[:4]))
    (OUT / "result.json").write_text(json.dumps(out, indent=1, default=float) + "\n")
    print(json.dumps(out["verdict"], indent=1), flush=True)


if __name__ == "__main__":
    assert os.environ.get("MAMBA_DETERMINISTIC") == "1"
    RUN.mkdir(exist_ok=True)
    steps = {"gpu": ("references", "trained"), "all": ("references", "trained", "summary")}.get(sys.argv[1], (sys.argv[1],))
    env = setup()
    for step in steps:
        target = OUT / ("result.json" if step == "summary" else f"{step}.json")
        if target.exists() and not SMOKE:
            print("skip", step, flush=True)
            continue
        print("step", step, time.strftime("%H:%M:%S"), flush=True)
        {"references": references, "trained": trained, "summary": summary}[step](env)
