"""PLAN v2 test E: is a zombie attack at t predictable from what the denoiser is conditioned on? One-sided. GPU.

CLAIM: the denoiser's conditioning (8 context frames' percepts with fixations, their predicted beliefs = the memory
tokens' source, the current predicted belief, a_{t-1}, the target fixation) contains what predicts the attack, including
the hidden cooldown (test A: the field ignores it).
COMPETING: it does not, so no generator could place the risk from this conditioning.
CONTROL: the same probe protocol (S1 fit: MLP 256-256, AdamW, early stopping) on (b) the current predicted belief +
action + fixation only, (c) the last context frame + action (no history), and (d) the attack rule's causal inputs
(game_logic.update_mobs after craftax_step's do_action / move_player: zombie at L1 1 from the player's POST-move
position, cooldown <= 0, mask, sleeping; no realised random draw), which must be near-deterministic.
Event: a zombie's cooldown resets to 5 during t-1 -> t (an attack). Population: zombie within L1 2 at t-1.
Reading: a strong probe on (a) shows the information is accessible to this reader; a weak one shows nothing.
Revised 10-29 after Codex's review: the cooldown readout is isolated among rows with a zombie adjacent after the move
(ready vs not ready; the first version compared ready-and-adjacent with everything else), and a drink probe asks
whether the drink consequence is accessible from the conditioning.
Context frames' reward / termination columns are zeroed (the denoiser does not see them directly).
Frozen P1 world seed 1, checkpoint 10; dev split, one third of physical episodes held out.
Usage (project venv, GPU): python -m artifacts.experiments.20261029_generation_diagnosis.information_probe OUT_JSON
"""
import importlib, json, sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from saccade import model as M, train as TR
from saccade.eval import s1 as S, gates as G

FC = importlib.import_module("artifacts.experiments.20261029_generation_diagnosis.field_calibration")
ROOT = Path(__file__).resolve().parents[3]
DEV = FC.DEV
HISTORY = 8
FRAME = 1296 + 576 + 49 + 8 + 1 + 1 + 17 + 256 + 1                      # frame_features + bp + validity (s1)


def population(L, enc):
    s_off, a_off = np.r_[0, np.cumsum(L["ep_T"] + 1)], np.r_[0, np.cumsum(L["ep_T"])]
    pos, zp, zm, cd = L["s_pos"].astype(int), L["s_zombies_p"].astype(int), L["s_zombies_m"], L["s_zombies_cd"]
    rows = []
    for e in range(len(enc)):
        for t in range(HISTORY, len(enc[e]["p"])):
            g = s_off[e] + t - 1
            if not (zm[g] & (np.abs(zp[g] - pos[g]).sum(-1) <= 2)).any():
                continue
            attack = bool((zm[g] & (cd[g] <= 0) & (cd[g + 1] == 5)).any())
            post = np.abs(zp[g] - pos[g + 1]).sum(-1) == 1                 # player's post-move position
            rule = np.r_[post & zm[g], (cd[g] <= 0) & zm[g], zm[g], L["s_sleep"][g]].astype(np.float32)
            rows.append((e, t, attack, rule, bool((post & zm[g] & (cd[g] <= 0)).any()), bool((post & zm[g]).any())))
    return rows


def drink_population(L, enc):
    """Added 10-29 (Codex review point 3): is the drink consequence accessible from the conditioning? Rows: awake,
    a_{t-1} = DO, drink <= 8, facing anything; label: drink + 1 at t (do_action: facing water)."""
    s_off, a_off = np.r_[0, np.cumsum(L["ep_T"] + 1)], np.r_[0, np.cumsum(L["ep_T"])]
    rows = []
    for e in range(len(enc)):
        for t in range(HISTORY, len(enc[e]["p"])):
            g = s_off[e] + t - 1
            if L["s_sleep"][g] or int(L["t_action"][a_off[e] + t - 1]) != 5 or L["s_body"][g, 2] > 8:
                continue
            rows.append((e, t, bool(L["s_body"][g + 1, 2] == L["s_body"][g, 2] + 1)))
    return rows


def auc(score, positive):
    """Mann-Whitney AUC of score for positive vs negative rows (ties averaged)."""
    r = np.argsort(np.argsort(score, kind="stable"), kind="stable").astype(float)
    for v in np.unique(score):
        m = score == v
        r[m] = r[m].mean()
    n1, n0 = positive.sum(), (~positive).sum()
    return float((r[positive].sum() - n1 * (n1 - 1) / 2) / (n1 * n0)) if n1 and n0 else None


def features(enc, rows):
    X = S.conditioning_features(enc, [r[0] for r in rows], [r[1] for r in rows], HISTORY)
    for h in range(HISTORY):                                             # zero reward / termination of context frames
        X[:, h * FRAME + 1929:h * FRAME + 1931] = 0
    return X


def probe(X, y, ep, rng):
    te = G.episode_split(ep, ep[:0], rng)
    tr_eps = np.unique(ep[~te])
    va = ~te & np.isin(ep, rng.choice(tr_eps, max(1, len(tr_eps) // 10), replace=False))
    parts = [np.flatnonzero(~te & ~va), np.flatnonzero(va), np.flatnonzero(te)]
    Xt = torch.as_tensor(X[parts[0]]).float()
    mean, std = Xt.mean(0).to(DEV), Xt.std(0).clamp(min=1e-3).to(DEV)
    yt = torch.as_tensor(y, device=DEV).long()
    mx = lambda idx, val: (torch.as_tensor(X[parts[int(val)][idx.cpu().numpy()]], device=DEV).float() - mean) / std
    loss = lambda out, idx, val: F.cross_entropy(out["y"], yt[parts[int(val)]][idx])
    model, log = S.fit(mx, loss, len(parts[0]), len(parts[1]), {"y": 2}, 0, DEV)
    with torch.no_grad():
        p = torch.cat([model((torch.as_tensor(X[parts[2][i.numpy()]], device=DEV).float() - mean) / std)["y"]
                       .softmax(-1)[:, 1].cpu() for i in torch.arange(len(parts[2])).split(4096)]).numpy()
    yte = y[parts[2]]
    eps_ = 1e-6
    bins = np.clip((p * 10).astype(int), 0, 9)
    ece = sum(abs(p[bins == b].mean() - yte[bins == b].mean()) * (bins == b).mean() for b in range(10) if (bins == b).any())
    return {"n_train": len(parts[0]), "n_test": len(parts[2]), "test_rate": float(yte.mean()),
            "test_mean_p": float(p.mean()), "ap": G.average_precision(yte, p),
            "log_loss": float(-(yte * np.log(p + eps_) + (1 - yte) * np.log(1 - p + eps_)).mean()),
            "base_log_loss": float(-(yte * np.log(yte.mean()) + (1 - yte) * np.log(1 - yte.mean())).mean()),
            "ece_10": float(ece), "epochs": len(log)}, parts[2], p


def main():
    out = Path(sys.argv[1]); out.parent.mkdir(parents=True, exist_ok=True)
    ck = torch.load(FC.CKPT, weights_only=False, map_location="cpu")
    cfg = ck["cfg"]
    model = M.M0(cfg).to(DEV); model.load_state_dict(ck["model"]); model.eval()
    L = dict(np.load(ROOT / "artifacts/eda/s0/s0_dev_labels.npz"))
    enc = S.encode(model, L, cfg)
    rows = population(L, enc)
    y = np.array([r[2] for r in rows]); ep = np.array([r[0] for r in rows]); ready = np.array([r[4] for r in rows])
    adj = np.array([r[5] for r in rows])
    X = features(enc, rows)
    n_last = X.shape[1] - (256 + 17 + 49)
    inputs = {"a_full_conditioning": X, "b_belief_action_fixation": X[:, n_last:],
              "c_last_frame_action": np.concatenate([X[:, (HISTORY - 1) * FRAME:HISTORY * FRAME], X[:, n_last + 256:]], 1),
              "d_rule_causal_inputs": np.stack([r[3] for r in rows])}
    res = {"checkpoint_sha256": TR.file_sha256(FC.CKPT), "rows": len(rows), "attack_rate": float(y.mean()),
           "rule_ready_and_adjacent_rate": float(ready.mean()),
           "attack_given_ready_adjacent": float(y[ready].mean()), "attack_given_not": float(y[~ready].mean()),
           "adjacent_after_move_rows": int(adj.sum()), "attack_given_adjacent_ready": float(y[adj & ready].mean()),
           "attack_given_adjacent_not_ready": float(y[adj & ~ready].mean())}
    for name, Xi in inputs.items():
        r, te_rows, p = probe(Xi, y, ep, np.random.default_rng(2029))
        r["mean_p_ready_adjacent"] = float(p[ready[te_rows]].mean()); r["mean_p_not"] = float(p[~ready[te_rows]].mean())
        a_te, r_te = adj[te_rows], ready[te_rows]                        # cooldown isolated: adjacent after the move
        r["adjacent_after_move"] = {"rows": int(a_te.sum()), "mean_p_ready": float(p[a_te & r_te].mean()),
                                    "mean_p_not_ready": float(p[a_te & ~r_te].mean()),
                                    "auc_ready_vs_not": auc(p[a_te], r_te[a_te])}
        res[name] = r
        print(name, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()}, flush=True)
        out.write_text(json.dumps(res, indent=1))
    drows = drink_population(L, enc)
    yd, ed = np.array([r[2] for r in drows]), np.array([r[0] for r in drows])
    Xd = features(enc, [(r[0], r[1]) for r in drows])
    res["drink"] = {"rows": len(drows), "rate": float(yd.mean())}
    for name, Xi in {"a_full_conditioning": Xd, "b_belief_action_fixation": Xd[:, n_last:],
                     "c_last_frame_action": np.concatenate([Xd[:, (HISTORY - 1) * FRAME:HISTORY * FRAME], Xd[:, n_last + 256:]], 1)}.items():
        res["drink"][name] = probe(Xi, yd, ed, np.random.default_rng(2029))[0]
        print("drink", name, res["drink"][name], flush=True)
        out.write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
