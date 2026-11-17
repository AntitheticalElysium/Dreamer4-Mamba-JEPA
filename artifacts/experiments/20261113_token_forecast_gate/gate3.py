"""Round 3 of the Option A gate (PLAN.md round 3): plain-MSE token forecast, absolute (round 2's head: belief only)
against a fixed skip (observed current tokens + the same head on the belief). Frozen F5 checkpoints, fresh heads from
one initialisation, same draws and budget; no world training.

    MAMBA_DETERMINISTIC=1 python -m artifacts.experiments.20261113_token_forecast_gate.gate3 [checkpoints...]
"""
import copy
import importlib
import json
import os
import sys

import numpy as np
import torch

from saccade import train as TR
from saccade.eval import s1 as S

G = importlib.import_module("artifacts.experiments.20261113_token_forecast_gate.gate")
G2 = importlib.import_module("artifacts.experiments.20261113_token_forecast_gate.gate2")
CR, TJ, NP, DEV, CV, F5 = G.CR, G.TJ, G.NP, G.DEV, G.CV, G.F5
OUT = G.HERE / os.environ.get("GATE_OUT", "readings/round3")
STEPS, BATCH, FOILS, SETS = G.STEPS, G.BATCH, G.FOILS, G2.SETS
ARMS = ("absolute", "skip")
mse = lambda a, b, slot=None: ((a - b) ** 2 if slot is None else (a - b)[..., slot, :] ** 2).mean()


def checkpoint(k):
    path = OUT / f"ckpt_{k:02d}.json"
    if path.exists():
        return
    torch.manual_seed(0)
    ck = (F5 / f"run/ckpt_{k:02d}.pt").resolve()
    model, cfg = NP.IF.load(ck)
    model.eval()
    L_ = dict(np.load(CV.LABELS / "s0_dev_labels.npz"))
    enc = S.encode(model, L_, cfg)
    tgt, prev, bps, ep_of = [], [], [], []
    for e, r in enumerate(enc):
        tgt.append(r["p"][1:].half()); prev.append(r["p"][:-1].half()); bps.append(r["bp"][1:].float())
        ep_of.append(torch.full((len(r["p"]) - 1,), e))
    tgt, prev, bp, ep_of = map(torch.cat, (tgt, prev, bps, ep_of))
    del enc
    held = ep_of % 5 == 4
    tr_ix, te_ix = torch.nonzero(~held)[:, 0], torch.nonzero(held)[:, 0]
    res = {"checkpoint": k, "sha256": TR.file_sha256(ck)}
    init = G2.make_head("plain", bp.shape[-1])                              # round 2's plain head: same position in the RNG stream
    heads = {}
    own = getattr(model, "token_forecast", None)                            # a world trained with the change-form forecast:
    arms = ARMS + (("native",) if own is not None else ())                  # its own head as a third arm, not refitted
    for arm in arms:
        net = own if arm == "native" else copy.deepcopy(init)
        base = (lambda ix: prev[ix].to(DEV).float()) if arm != "absolute" else (lambda ix: 0.)
        opt = torch.optim.Adam(net.parameters(), 1e-3)
        g = torch.Generator().manual_seed(1)
        for s in range(0 if arm == "native" else STEPS):
            ix = tr_ix[torch.randint(len(tr_ix), (BATCH,), generator=g)]
            loss = mse(base(ix) + net(bp[ix].to(DEV)), tgt[ix].to(DEV).float())
            opt.zero_grad(); loss.backward(); opt.step()
            if s % 5000 == 0:
                print(k, arm, s, float(loss), flush=True)
        net.eval()
        heads[arm] = net
        with torch.no_grad():
            lh = ls = lc = 0.
            shuf = te_ix[torch.randperm(len(te_ix), generator=torch.Generator().manual_seed(5))]
            for a in range(0, len(te_ix), 8192):
                ix, jx = te_ix[a:a + 8192], shuf[a:a + 8192]
                x = tgt[ix].to(DEV).float()
                lh += float(mse(base(ix) + net(bp[ix].to(DEV)), x)) * len(ix)
                ls += float(mse(base(ix) + net(bp[jx].to(DEV)), x)) * len(ix)
                lc += float(mse(prev[ix].to(DEV).float(), x)) * len(ix)
            res[f"heldout/{arm}"] = {"head": lh / len(te_ix), "shuffled_belief": ls / len(te_ix), "copy_previous": lc / len(te_ix)}
        print(k, arm, json.dumps(res[f"heldout/{arm}"]), flush=True)
    if F5 == G.F5:                                                          # the round-2 reproduction is F5's only
        r2 = json.loads((G.HERE / f"readings/round2/ckpt_{k:02d}.json").read_text())["heldout/plain"]["head"]
        res["round2_plain_reproduction"] = {"round2": r2, "now": res["heldout/absolute"]["head"]}
    del tgt, prev, bp
    for prm in model.parameters():
        prm.requires_grad_(True)
    params = [x for x in model.belief.parameters() if x.requires_grad]
    grad = lambda f, keep=True: G2.flat(torch.autograd.grad(f, params, retain_graph=keep, allow_unused=True), params)
    rows = {name: TJ.pair_rows(name) for name in SETS}

    def forward(arm, row):
        """(forecast [2, 81, 16], observed current tokens [2, 81, 16]) at the due step."""
        p, rw, td, act = CR.belief_inputs(model, cfg, row)
        with TR.autocast(DEV, cfg):
            y, _ = model.belief(model.belief.tokens(model.belief.percept_embed(p), rw, td, act), model.belief.initial(2, DEV))
        cur = p.p[:, -1].float().detach()
        return (cur if arm != "absolute" else 0.) + heads[arm](y[:, -1].float()), cur

    fold_sum = {}
    for name, slot in SETS.items():
        per = {arm: [] for arm in arms}
        for n_, row in enumerate(rows[name]):
            x, xb = G2.due_targets(model, row)
            for arm in arms:
                out, cur = forward(arm, row)
                loss = mse(out[0], x[0]) + mse(out[1], x[1])
                sep = mse(out[1], x[0]) + mse(out[0], x[1]) - loss
                dig = mse(out[0], x[0], slot) + mse(out[1], x[1], slot)
                gl, gd, gs = grad(loss), grad(dig), grad(sep, False)
                fold_sum.setdefault((name, arm), [torch.zeros_like(gl), torch.zeros_like(gl)])[n_ % 2] += gl
                with torch.no_grad():
                    dx = (x[0] - x[1])[slot]
                    frac = lambda u: float((u * dx).sum() / dx.square().sum().clamp(min=1e-12))
                    r = [(out[m] - x[m])[slot] for m in (0, 1)]
                    cons = sum(mse(out[m], x[0]) - mse(out[m], x[1]) for m in (0, 1)) / 2
                    foil = {str(a): float(sum(mse(out[m], xb[a]) - mse(out[m], x[1]) for m in (0, 1)) / 2) for a in FOILS}
                    per[arm].append({
                        "cosine": G2.cosine(gs, gl), "cosine_digit_cell": G2.cosine(gs, gd), "grad_sep_norm": float(gs.norm()),
                        "kappa": frac((out[0] - out[1])[slot]),
                        "change_due": frac((out[0] - cur[0])[slot]), "change_not_due": frac((out[1] - cur[1])[slot]),
                        "true_change_not_due": frac((x[1] - cur[1])[slot]), "true_change_due": frac((x[0] - cur[0])[slot]),
                        "current_tokens_equal": float((cur[0] - cur[1]).abs().max()),
                        "digit_common_error": float(((r[0] + r[1]) / 2).square().mean()),
                        "digit_branch_error": float(((r[0] - r[1]) / 2).square().mean()),
                        "all_common_error": float((((out[0] - x[0]) + (out[1] - x[1])) / 2).square().mean()),
                        "consequence": float(cons), "foil": foil})
                del out
        for arm, P in per.items():
            med = lambda f: float(np.median([e[f] for e in P]))
            summ = {"pairs": len(P), "cosine_mean_ci": G.boot_mean([e["cosine"] for e in P]),
                    "cosine_digit_cell_mean_ci": G.boot_mean([e["cosine_digit_cell"] for e in P]),
                    "kappa_mean_ci": G.boot_mean([e["kappa"] for e in P]),
                    "change_due_mean_ci": G.boot_mean([e["change_due"] for e in P]),
                    "change_not_due_mean_ci": G.boot_mean([e["change_not_due"] for e in P]),
                    "true_change_due_median": med("true_change_due"), "true_change_not_due_median": med("true_change_not_due"),
                    "current_tokens_max_difference": float(max(e["current_tokens_equal"] for e in P)),
                    "digit_common_error_median": med("digit_common_error"), "digit_branch_error_median": med("digit_branch_error"),
                    "all_common_error_median": med("all_common_error"), "grad_sep_norm_median": med("grad_sep_norm"),
                    "foil_beats_consequence": {a: int(sum(e["foil"][a] >= e["consequence"] for e in P)) for a in map(str, FOILS)}}
            res[f"{name}/{arm}"] = summ | {"per_pair": P}
            print(k, name, arm, json.dumps(summ), flush=True)
        d = np.subtract([e["cosine_digit_cell"] for e in per["skip"]], [e["cosine_digit_cell"] for e in per["absolute"]])
        res[f"{name}/skip_minus_absolute_digit_cosine"] = G.boot_mean(d)
    unit = {key: [-s_ / s_.norm().clamp(min=1e-30) for s_ in v] for key, v in fold_sum.items()}
    del fold_sum
    for name, slot in SETS.items():                                      # matched held-out update, plain-gradient geometry
        eff = {arm: [] for arm in arms}
        for n_, row in enumerate(rows[name]):
            x, _ = G2.due_targets(model, row)
            dx = (x[0] - x[1])[slot]
            for arm in arms:
                out, _ = forward(arm, row)
                kap = ((out[0] - out[1])[slot] * dx).sum() / dx.square().sum().clamp(min=1e-12)
                eff[arm].append(float(grad(kap, False) @ unit[(name, arm)][1 - n_ % 2]))
        res[f"matched/{name}"] = {arm: {"mean_ci": G.boot_mean(v), "per_pair": v} for arm, v in eff.items()}
        print(k, "matched", name, json.dumps({a: res[f"matched/{name}"][a]["mean_ci"] for a in arms}), flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(res, indent=1) + "\n")
    del model
    torch.cuda.empty_cache()


if __name__ == "__main__":
    assert os.environ.get("MAMBA_DETERMINISTIC") == "1"
    OUT.mkdir(parents=True, exist_ok=True)
    for k in (map(int, sys.argv[1:]) if len(sys.argv) > 1 else (1, 2, 10)):
        checkpoint(k)
