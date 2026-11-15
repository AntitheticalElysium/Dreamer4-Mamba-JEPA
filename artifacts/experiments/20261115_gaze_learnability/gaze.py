"""Can the gaze forecast teach Mamba the pixel clocks? (PLAN.md) The 11-14 learnability instrument
(`../20261114_learnability/learnable.py`: F6 checkpoint 10, perception frozen, 128-step windows from the zero state,
the same window and fixation draws as its arms) with the gaze forecast as the only loss: the predicted belief before
frame t, beside the chooser's own 8-frame inputs, forecasts F6's frozen final chooser's 49-window distribution at t.
Simulator labels are used only to read the result (probes, cohorts, exact-history banks), never in a loss.

    MAMBA_DETERMINISTIC=1 python -m artifacts.experiments.20261115_gaze_learnability.gaze all
    (steps: references, none, frozen, trained, summary; each is skipped when its output exists)
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
if SMOKE:
    for mod in (LB, CR, CR.NP.IF):
        mod.DEV = torch.device("cpu")
DEV = LB.DEV
H, W, B = 8, (16 if SMOKE else LB.W), (2 if SMOKE else LB.B)
UPDATES, EVALS = (3, (3,)) if SMOKE else (LB.UPDATES, LB.EVALS)
BANKS = {"energy": 66, "hunger": 64, "recovery": 63}                     # F6's exact-history banks -> the digit's cell
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


def batch_terms(model, cfg, chooser, pol, belief, head, full, chunk, grad):
    """(predicted beliefs [B * (W - 1), width] of window frames 1..W-1, the chooser's inputs and distribution there,
    the percepts) for one draw; the belief runs with autograd only when `grad`."""
    with torch.no_grad(), TR.autocast(DEV, cfg):
        P = model.perception(chunk["frames"], chunk["fixation"])
    P = M.Percept(*(z.float() for z in P.parts()), P.fix)
    xx, q = teacher(model, chooser, pol, full, every=W)
    if belief is None:
        b = torch.zeros(len(q), cfg["belief"]["width"], device=DEV)
    else:
        with torch.set_grad_enabled(grad), TR.autocast(DEV, cfg):
            b = LB.run_belief(belief, P, chunk).float()
        b = b.reshape(-1, b.shape[-1])
    return b, xx, q, P


def soft_ce(head, xx, b, q):
    return -(q * head(xx, b).log_softmax(-1)).sum(-1).mean()


def cohorts(L):
    """Per state t (transition t-1 -> t), masks [N, 5] over learnable.EVENTS. strict: eligible and the last 8 frames
    (t-8 .. t-1) in the eligible sleep state with the event's digit unchanged (any gain needs older history).
    shortcut_free: strict and, for food / drink, an intake rise of that digit or a sleep earlier in the episode; for the
    awake energy drop, a sleep earlier in the episode (steps since the episode began no longer give the phase); for the
    asleep energy rise, WAKE_UP already earned (Codex's repeat-sleep cohort, where the reward forecast stops asking);
    recovery: strict only."""
    _, ok = LB.events(L)
    body, sleep, wake = L["s_body"].astype(int), L["s_sleep"].astype(bool), L["s_ach"][:, 15].astype(bool)
    off = np.r_[0, np.cumsum(L["ep_T"] + 1)]
    n = len(body)
    t = np.arange(n) - np.repeat(off[:-1], L["ep_T"] + 1)
    digit = (1, 2, 3, 3, 0)
    strict = ok & (t >= H)[:, None]
    for j, d in enumerate(digit):
        for k in range(2, H + 1):
            i = np.flatnonzero(strict[:, j])
            same = (body[i - k, d] == body[i - 1, d]) & (sleep[i - k] == sleep[i - 1])
            strict[i[~same], j] = False
    rise = np.zeros_like(body, bool)
    rise[1:] = body[1:] > body[:-1]
    rise[off[:-1]] = False
    before = lambda x: np.concatenate([np.maximum.accumulate(x[a:b]) for a, b in zip(off[:-1], off[1:])])
    slept, ate, drank = (np.r_[False, before(z)[:-1]] for z in (sleep, rise[:, 1], rise[:, 2]))   # at a state < t
    first = np.zeros(n, bool); first[off[:-1]] = True
    slept[first], ate[first], drank[first] = False, False, False
    free = strict.copy()
    free[:, 0] &= ate | slept
    free[:, 1] &= drank | slept
    free[:, 2] &= slept
    free[:, 3] &= np.r_[False, wake[:-1]] & ~first
    return {"strict": strict, "shortcut_free": free}


def probe_scores(X, y, ok, fold):
    """learnable.probe (one linear logistic probe per event, fit on folds 0-3), returning the held-out scores."""
    X = (X - X[fold < 4].mean(0)) / X[fold < 4].std(0).clamp(min=1e-6)
    out = {}
    for j, n in enumerate(LB.EVENTS):
        tr, te = (fold < 4) & ok[:, j], (fold == 4) & ok[:, j]
        Xt, yt = X[tr].to(DEV), torch.as_tensor(y[tr, j], dtype=torch.float32, device=DEV)
        w = torch.zeros(X.shape[1] + 1, device=DEV, requires_grad=True)
        opt = torch.optim.LBFGS([w], max_iter=200, line_search_fn="strong_wolfe")

        def closure():
            opt.zero_grad()
            l = F.binary_cross_entropy_with_logits(Xt @ w[:-1] + w[-1], yt) + 1e-3 * w[:-1].square().sum()
            l.backward()
            return l
        opt.step(closure)
        out[n] = (X[te].to(DEV) @ w[:-1] + w[-1]).detach().cpu().numpy()
    return out


def read_probes(X, L, y, ok_all, fold, masks):
    """Scores and APs per mask (natural eligibility and the two cohorts)."""
    res, scores = {}, {}
    for name, ok in [("natural", ok_all)] + list(masks.items()):
        s = probe_scores(X, y, ok, fold)
        scores[name] = s
        res[name] = {n: {"ap": AP(y[(fold == 4) & ok[:, j], j], s[n]),
                         "rows": int(((fold == 4) & ok[:, j]).sum()), "events": int(y[(fold == 4) & ok[:, j], j].sum())}
                     for j, n in enumerate(LB.EVENTS)}
    return res, scores


@torch.no_grad()
def dev_fold4(model, chooser, pol, L):
    """The held-out DEV fifth: the chooser's coarse tokens per frame, and at every frame 1..T its inputs (history as an
    index into the tokens, as fair.py), its distribution and the frame's label row."""
    off = np.r_[0, np.cumsum(L["ep_T"] + 1)]
    toks, ctx, act, q, glob, state = [], [], [], [], [], []
    base = 0
    for e, ep in enumerate(S.episodes(L)):
        if e % 5 != 4:
            continue
        T = int(L["ep_T"][e])
        full = D.gather([ep], [[None] * H + [(0, 1, t, 1, 1) for t in range(T + 1)]])
        xx, qq = teacher(model, chooser, pol, full, every=T + 1)
        p, _ = C.coarse(model.perception, torch.as_tensor(full["frames"][0, H:], device=DEV))
        toks.append(p.cpu()); ctx.append(xx[1].cpu()); act.append(xx[2].cpu()); q.append(qq.cpu())
        glob.append(base + np.arange(1, T + 1)); state.append(off[e] + np.arange(1, T + 1))
        base += T + 1
    return {"toks": torch.cat(toks), "ctx": torch.cat(ctx), "act": torch.cat(act), "q": torch.cat(q),
            "glob": torch.as_tensor(np.concatenate(glob)), "state": np.concatenate(state)}


@torch.no_grad()
def dev_kl(head, X, dv):
    """KL(chooser || forecast) at every held-out frame; X = predicted beliefs per DEV state, or None (input zero)."""
    out = []
    for a in range(0, len(dv["q"]), 2048):
        r = slice(a, a + 2048)
        ctx = dv["ctx"][r].float()
        hist = dv["glob"][r, None] - torch.arange(H, 0, -1)[None]
        ph = dv["toks"][hist.clamp(min=0)].float() * ctx.reshape(-1, H, 29)[..., 28][..., None, None]
        b = torch.zeros(len(ctx), head.belief.in_features) if X is None else X[dv["state"][r]]
        lq = head((ph.to(DEV), ctx.to(DEV), dv["act"][r].to(DEV)), b.float().to(DEV)).log_softmax(-1)
        p = dv["q"][r].to(DEV)
        out.append((p * (p.log() - lq)).sum(-1).cpu())
    return torch.cat(out).numpy()


@torch.no_grad()
def bank_read(model, cfg, chooser, pol, belief, head):
    """F6's exact-history banks (the last 8 frames, actions and outcomes identical across the two branches; the older
    history makes one branch due): per pair, the forecast probability that the chosen crop covers the bank's digit,
    per branch, from this arm's belief and head, and the chooser's own (which reads the target frame)."""
    res = {}
    for name, cell in BANKS.items():
        cover = (C.WIN == cell).any(-1).to(DEV)
        rows = []
        for row in TJ.pair_rows(name)[:2 if SMOKE else None]:
            i, e, prefix, frames, rewards, terms, q, truth = row
            assert q >= H and np.array_equal(frames[0, q - H + 1:q + 1], frames[1, q - H + 1:q + 1])
            changed = truth[:, 1] != truth[:, 0]
            assert changed.sum() == 1, truth
            due = int(np.flatnonzero(changed)[0])
            clip = {"frames": frames[:, q - H + 1:q + 2], "episode": np.ones((2, H + 1), np.int64),
                    "valid": np.ones((2, H + 1), bool), "prev_action": np.zeros((2, H + 1), np.int64),
                    "reward": rewards[:, q - H:q + 1].astype(np.float32), "terminated": terms[:, q - H:q + 1].astype(bool)}
            xx, qt = teacher(model, chooser, pol, clip)
            assert all(torch.equal(z[0], z[1]) for z in xx)
            if belief is None:
                b = torch.zeros(2, cfg["belief"]["width"], device=DEV)
            else:
                p, rw, td, a = CR.belief_inputs(model, cfg, row)
                with TR.autocast(DEV, cfg):
                    y, _ = belief(belief.tokens(belief.percept_embed(p), rw, td, a), belief.initial(2, DEV))
                b = y[:, -1].float()
            f = head(xx, b).softmax(-1)
            order = [due, 1 - due]
            rows.append({"e": int(e), "due_branch": due, "direction": int(np.sign(truth[due, 1] - truth[due, 0])),
                         "forecast": [float(f[k, cover].sum()) for k in order], "teacher": [float(qt[k, cover].sum()) for k in order]})
        res[name] = rows
    return res


def dose(model, cfg, chooser, pol, head, Ltr, eps, long_eps, batches=16):
    """On F6's own belief: gradients into the belief's parameters from the gaze forecast (the frozen arm's fitted head),
    F6's reward forecast (distance 1, as trained in F6) and F6's token forecast, on the same windows (own draws)."""
    belief = copy.deepcopy(model.belief).train()
    params = list(belief.parameters())
    for p in params:
        p.requires_grad_(True)
    flat = lambda gs: torch.cat([(g if g is not None else torch.zeros_like(p)).flatten() for g, p in zip(gs, params)])
    cos = lambda a, b: float(a @ b / (a.norm() * b.norm()).clamp(min=1e-30))
    rng, per = np.random.default_rng(8), []
    for _ in range(2 if SMOKE else batches):
        full, chunk = windows(rng, Ltr, eps, long_eps)
        b, xx, q, P = batch_terms(model, cfg, chooser, pol, belief, head, full, chunk, grad=True)
        with TR.autocast(DEV, cfg):
            rf = -model.outcome.reward_logp(model.outcome.features(b, None), chunk["reward"][:, 1:].reshape(-1)).mean()
        with torch.autocast(DEV.type, enabled=False):
            tf = (P.p[:, :-1].float() + model.token_forecast(b.reshape(B, W - 1, -1))).sub(P.p[:, 1:].float()).square().mean()
        terms = {"gaze": soft_ce(head, xx, b, q), "reward_forecast": rf.float(), "token_forecast": tf}
        g = {k: flat(torch.autograd.grad(v, params, retain_graph=True, allow_unused=True)) for k, v in terms.items()}
        per.append({"loss": {k: float(v) for k, v in terms.items()}, "norm": {k: float(v.norm()) for k, v in g.items()},
                    "cos": {"gaze_reward": cos(g["gaze"], g["reward_forecast"]), "gaze_token": cos(g["gaze"], g["token_forecast"]),
                            "token_reward": cos(g["token_forecast"], g["reward_forecast"])}})
    med = lambda key: {k: float(np.median([r[key][k] for r in per])) for k in per[0][key]}
    return {"batches": len(per), "loss_median": med("loss"), "norm_median": med("norm"), "cos_median": med("cos"),
            "norm_ratio_median": {k: float(np.median([r["norm"]["gaze"] / r["norm"][k] for r in per]))
                                  for k in ("reward_forecast", "token_forecast")}, "per_batch": per}


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
    """A0 (F6's checkpoint-10 belief, unchanged) and A5 (the 11-14 G1-only belief after 2,000 updates) read by the same
    probes; their natural APs must reproduce reference.json and g1.json (the instrument's known answers)."""
    ck, model, cfg, chooser, pol, Ldev, ydev, okdev, fold, masks = env
    out = {"checkpoint_sha256": TR.file_sha256(ck)}
    a5 = M.M0(cfg).to(DEV)
    a5.load_state_dict(torch.load(EXP / "20261114_learnability/g1_run/run/ckpt_02.pt", map_location="cpu", weights_only=False)["model"])
    a5.eval()
    if SMOKE:
        for m in a5.belief.mixers:
            m.backend = "reference"
    known = {"A0": json.loads((EXP / "20261114_learnability/reference.json").read_text())["probe"],
             "A5": json.loads((EXP / "20261114_learnability/g1.json").read_text())["evals"]["2000"]}
    for arm, belief in (("A0", model.belief), ("A5", a5.belief)):
        X = LB.encode_dev(model, belief, cfg, Ldev)
        out[arm], scores = read_probes(X, Ldev, ydev, okdev, fold, masks)
        out[arm]["known_answer"] = {n: [out[arm]["natural"][n]["ap"], known[arm][n]["ap"]] for n in LB.EVENTS}
        if not SMOKE:
            assert all(abs(a - b) < 2e-3 for a, b in out[arm]["known_answer"].values()), out[arm]["known_answer"]
        np.savez(RUN / f"scores_{arm}.npz", **{f"{m}__{n}": s for m, d in scores.items() for n, s in d.items()})
        print(arm, json.dumps(out[arm]["known_answer"]), flush=True)
    (OUT / "references.json").write_text(json.dumps(out, indent=1) + "\n")


def arm(env, name):
    """One arm: none (the belief input held at zero: the 8-frame baseline), frozen (F6's belief, unchanged), trained
    (F6's belief trained by the gaze forecast alone). Same head initialisation, draws, schedule and optimiser as
    learnable.main (Adam, head 1e-3, belief 1e-4, 100 warm-up updates, clip 1)."""
    ck, model, cfg, chooser, pol, Ldev, ydev, okdev, fold, masks = env
    Ltr = dict(np.load(S.LABELS / "s0_train_labels.npz"))
    eps = S.episodes(Ltr)
    long_eps = np.flatnonzero(Ltr["ep_T"] >= W)
    belief = None if name == "none" else copy.deepcopy(model.belief)
    if belief is not None:
        belief.train()
        for p in belief.parameters():
            p.requires_grad_(name == "trained")
    torch.manual_seed(2)
    head = Gaze(cfg["belief"]["width"]).to(DEV)
    groups = [{"params": list(head.parameters()), "lr": 1e-3}]
    if name == "trained":
        groups.append({"params": list(belief.parameters()), "lr": 1e-4})
    opt = torch.optim.Adam(groups)
    base_lr = [g["lr"] for g in groups]
    dv = dev_fold4(model, chooser, pol, Ldev)
    np.save(RUN / "dev_state.npy", dv["state"])
    X0 = None if name == "none" else LB.encode_dev(model, model.belief, cfg, Ldev)
    rng = np.random.default_rng(7)                                     # learnable.main's draws
    res = {"arm": name, "checkpoint_sha256": TR.file_sha256(ck), "window": W, "batch": B, "evals": {}, "log": []}
    t0, losses = time.time(), []
    for u in range(1, UPDATES + 1):
        full, chunk = windows(rng, Ltr, eps, long_eps)
        b, xx, q, _ = batch_terms(model, cfg, chooser, pol, belief, head, full, chunk, grad=name == "trained")
        loss = soft_ce(head, xx, b, q)
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
        if u % 250 == 0 or SMOKE:
            res["log"].append({"update": u, "loss": float(np.mean(losses[-250:])), "sec_per_update": (time.time() - t0) / u})
            print(name, res["log"][-1], flush=True)
        if u in EVALS:
            head.eval()
            if belief is not None:
                belief.eval()
            X = X0 if name != "trained" else LB.encode_dev(model, belief, cfg, Ldev)
            np.save(RUN / f"kl_{name}_{u}.npy", dev_kl(head, X, dv))
            e = {"dev_kl_mean": float(np.load(RUN / f"kl_{name}_{u}.npy").mean())}
            if name != "none":
                e["banks"] = bank_read(model, cfg, chooser, pol, belief, head)
            if name == "trained":
                e["probes"], scores = read_probes(X, Ldev, ydev, okdev, fold, masks)
                np.savez(RUN / f"scores_trained_{u}.npz", **{f"{m}__{n}": s for m, d in scores.items() for n, s in d.items()})
                torch.save({"belief": belief.state_dict(), "head": head.state_dict()}, RUN / f"trained_{u}.pt")
            res["evals"][u] = e
            print(name, u, json.dumps({k: v for k, v in e.items() if k != "banks"}), flush=True)
            head.train()
            if belief is not None:
                belief.train()
    if name == "frozen":
        head.eval()
        res["dose"] = dose(model, cfg, chooser, pol, head, Ltr, eps, long_eps)
        print("dose", json.dumps({k: v for k, v in res["dose"].items() if k != "per_batch"}), flush=True)
    torch.save(head.state_dict(), RUN / f"{name}_head.pt")
    (OUT / f"{name}.json").write_text(json.dumps(res, indent=1) + "\n")


def summary(env):
    """The fixed readings (PLAN.md), with episode-bootstrap intervals (1,000 draws for APs, 10,000 for mean
    differences, seed 215) and pair-bootstrap intervals for the banks."""
    _, _, _, _, _, Ldev, ydev, okdev, fold, masks = env
    ref = json.loads((OUT / "references.json").read_text())
    tr = json.loads((OUT / "trained.json").read_text())
    fz = json.loads((OUT / "frozen.json").read_text())
    a1 = {int(l.split()[1]): json.loads(l.split(" ", 2)[2]) for l in (EXP / "20261114_learnability/events.log").read_text().splitlines()
          if l.startswith("events ") and l.split()[1].isdigit()}
    ep = np.repeat(np.arange(len(Ldev["ep_T"])), Ldev["ep_T"] + 1)
    rng = np.random.default_rng(215)
    out = {"natural_ap": {}, "cohort_ap": {}, "banks": {}, "belief_adds_gaze_kl": {}, "dose": fz.get("dose", {}), "verdict": {}}
    pixel = LB.EVENTS[:4]
    for n in LB.EVENTS:
        out["natural_ap"][n] = {"A0": ref["A0"]["natural"][n]["ap"], "A5": ref["A5"]["natural"][n]["ap"],
                                "A1": {u: a1[u][n]["ap"] for u in sorted(a1)},
                                "trained": {u: tr["evals"][str(u)]["probes"]["natural"][n]["ap"] for u in EVALS}}

    def ap_diff(mask_name, j, sa, sb):
        okm = (fold == 4) & (okdev if mask_name == "natural" else masks[mask_name])[:, j]
        y, e = ydev[okm, j], ep[okm]
        ids = np.unique(e)
        if not len(ids):                                                 # an empty cohort (only in the smoke's 2 episodes)
            return [float("nan")] * 3
        rows = {i: np.flatnonzero(e == i) for i in ids}
        d = []
        for _ in range(1000):
            idx = np.concatenate([rows[i] for i in rng.choice(ids, len(ids))])
            d.append(AP(y[idx], sa[idx]) - AP(y[idx], sb[idx]))
        return [AP(y, sa) - AP(y, sb), *np.nanquantile(d, [.025, .975]).tolist()]

    refs = {a: np.load(RUN / f"scores_{a}.npz") for a in ("A0", "A5")}
    for mask_name in ("natural", "strict", "shortcut_free"):
        out["cohort_ap"][mask_name] = {}
        for j, n in enumerate(LB.EVENTS):
            key = f"{mask_name}__{n}"
            row = {"A0": ref["A0"][mask_name][n], "A5": ref["A5"][mask_name][n]}
            for u in EVALS:
                st = np.load(RUN / f"scores_trained_{u}.npz")[key]
                row[str(u)] = {"ap": tr["evals"][str(u)]["probes"][mask_name][n]["ap"],
                               "minus_A0": ap_diff(mask_name, j, st, refs["A0"][key]),
                               "minus_A5": ap_diff(mask_name, j, st, refs["A5"][key])}
            out["cohort_ap"][mask_name][n] = row

    def pair_stat(rows):
        d = np.array([r["forecast"][0] - r["forecast"][1] for r in rows])
        bs = d[rng.integers(len(d), size=(10000, len(d)))].mean(1)
        return {"pairs": len(d), "directions": sorted({r["direction"] for r in rows}),
                "forecast_due": float(np.mean([r["forecast"][0] for r in rows])),
                "forecast_delayed": float(np.mean([r["forecast"][1] for r in rows])),
                "teacher_due": float(np.mean([r["teacher"][0] for r in rows])),
                "teacher_delayed": float(np.mean([r["teacher"][1] for r in rows])),
                "due_minus_delayed": [float(d.mean()), *np.quantile(bs, [.025, .975]).tolist()]}
    for name in BANKS:
        out["banks"][name] = {arm: {u: pair_stat(res["evals"][str(u)]["banks"][name]) for u in EVALS}
                              for arm, res in (("frozen", fz), ("trained", tr))}

    st = np.load(RUN / "dev_state.npy")                                  # the held-out frames' label rows, in KL order
    e4 = ep[st]
    ids, inv = np.unique(e4, return_inverse=True)
    boot = rng.integers(len(ids), size=(10000, len(ids)))
    sel = {"all": np.ones(len(st), bool), "repeat_sleep": masks["shortcut_free"][st, 3],
           "food_strict": masks["strict"][st, 0], "drink_strict": masks["strict"][st, 1],
           "energy_awake_strict_before_sleep": masks["strict"][st, 2] & ~masks["shortcut_free"][st, 2],
           "energy_awake_strict_after_sleep": masks["shortcut_free"][st, 2]}
    for u in EVALS:
        none = np.load(RUN / f"kl_none_{u}.npy")
        out["belief_adds_gaze_kl"][u] = {}
        for arm in ("frozen", "trained"):
            diff = none - np.load(RUN / f"kl_{arm}_{u}.npy")
            row = {}
            for c, m in sel.items():
                if not m.any():                                          # an empty cohort (only in the smoke)
                    row[c] = {"rows": 0, "gain": [float("nan")] * 3}
                    continue
                ns = np.bincount(inv[m], minlength=len(ids))[boot].sum(1)
                bs = np.bincount(inv[m], weights=diff[m], minlength=len(ids))[boot].sum(1)
                row[c] = {"rows": int(m.sum()), "gain": [float(diff[m].mean()), *np.quantile(bs[ns > 0] / ns[ns > 0], [.025, .975]).tolist()]}
            out["belief_adds_gaze_kl"][u][arm] = row

    u = EVALS[0] if SMOKE else 1500                                      # the bars (PLAN.md), at A1's reading point
    for n in pixel:
        a0, a1u = out["natural_ap"][n]["A0"], out["natural_ap"][n]["A1"][1500]   # A1 read at 750 and 1,500 only
        nat = out["natural_ap"][n]["trained"][u]
        free = out["cohort_ap"]["shortcut_free"][n][str(u)]
        bank = {"energy_down_awake": "energy", "food_down": "hunger"}.get(n)
        pair_ok = bank is None or out["banks"][bank]["trained"][u]["due_minus_delayed"][1] > 0
        taught = nat >= a0 + 0.25 * (a1u - a0) and free["minus_A0"][1] > 0 and free["minus_A5"][1] > 0 and pair_ok
        out["verdict"][n] = {"natural_ap": nat, "taught_threshold": a0 + 0.25 * (a1u - a0), "drop_threshold": a0 + 0.02,
                             "shortcut_free_lower_vs_A0": free["minus_A0"][1], "shortcut_free_lower_vs_A5": free["minus_A5"][1],
                             "pairs_lower": None if bank is None else out["banks"][bank]["trained"][u]["due_minus_delayed"][1],
                             "taught": bool(taught)}
    drop = all(out["natural_ap"][n]["trained"][u] < out["natural_ap"][n]["A0"] + 0.02 for n in pixel)
    out["verdict"]["overall"] = ("taught: " + ", ".join(n for n in pixel if out["verdict"][n]["taught"])
                                 if any(out["verdict"][n]["taught"] for n in pixel) else "drop" if drop else "between (for the user)")
    (OUT / "result.json").write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out["verdict"], indent=1), flush=True)


if __name__ == "__main__":
    assert os.environ.get("MAMBA_DETERMINISTIC") == "1"
    RUN.mkdir(exist_ok=True)
    steps = ("references", "none", "frozen", "trained", "summary") if sys.argv[1] == "all" else (sys.argv[1],)
    env = setup()
    for step in steps:
        target = OUT / ("result.json" if step == "summary" else f"{step}.json")
        if target.exists() and not SMOKE:
            print("skip", step, flush=True)
            continue
        print("step", step, time.strftime("%H:%M:%S"), flush=True)
        if step == "references":
            references(env)
        elif step == "summary":
            summary(env)
        else:
            arm(env, step)
