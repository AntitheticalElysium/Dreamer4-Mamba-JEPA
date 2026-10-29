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
            rows.append((e, t, attack, rule, bool((post & zm[g] & (cd[g] <= 0)).any())))
    return rows


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
    X = features(enc, rows)
    n_last = X.shape[1] - (256 + 17 + 49)
    inputs = {"a_full_conditioning": X, "b_belief_action_fixation": X[:, n_last:],
              "c_last_frame_action": np.concatenate([X[:, (HISTORY - 1) * FRAME:HISTORY * FRAME], X[:, n_last + 256:]], 1),
              "d_rule_causal_inputs": np.stack([r[3] for r in rows])}
    res = {"checkpoint_sha256": TR.file_sha256(FC.CKPT), "rows": len(rows), "attack_rate": float(y.mean()),
           "rule_ready_and_adjacent_rate": float(ready.mean()),
           "attack_given_ready_adjacent": float(y[ready].mean()), "attack_given_not": float(y[~ready].mean())}
    for name, Xi in inputs.items():
        r, te_rows, p = probe(Xi, y, ep, np.random.default_rng(2029))
        r["mean_p_ready_adjacent"] = float(p[ready[te_rows]].mean()); r["mean_p_not"] = float(p[~ready[te_rows]].mean())
        res[name] = r
        print(name, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()}, flush=True)
        out.write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
