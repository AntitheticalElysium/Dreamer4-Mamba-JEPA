"""F6.1 dissection, check 4 (PLAN.md): along the world's own generated sleep rollouts, does the predicted belief expect
the energy rise at the due step that the denoiser draws late? The quiet bank's and the repeat-sleep bank's own
generation (`../20261107_danger_adversarial/sleep.py`, unchanged), with every per-step belief output recorded; a linear
probe for the asleep energy rise (learnable.probe on the world's own DEV beliefs from check 1, fold 4 held out) read on
those beliefs. Known answer: the quiet-bank generation equals F6.1's queue run bitwise.

    MAMBA_DETERMINISTIC=1 python -m artifacts.experiments.20261115_f61_dissection.rollout_belief run    (GPU)
    python -m artifacts.experiments.20261115_f61_dissection.rollout_belief read                        (CPU)
"""
import importlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

G = importlib.import_module("artifacts.experiments.20261115_gaze_learnability.gaze")
SL = importlib.import_module("artifacts.experiments.20261107_danger_adversarial.sleep")
HZ = importlib.import_module("artifacts.experiments.20261115_f61_dissection.hazard")
HERE = Path(__file__).resolve().parent
CKPTS = {"F6.1": "artifacts/experiments/20261114_p1f61/run/ckpt_10.pt", "F6": "artifacts/experiments/20261113_p1f6/run/ckpt_10.pt"}
BANKS = {"quiet_bank": (SL.HERE / "bank.pt", HERE / "run/rollout_belief"), "repeat_sleep_bank": (HERE / "run/sleep_paid/bank.pt", HERE / "run/sleep_paid")}


def run():
    log, original = [], SL.CV.IF.load

    def load(path):                                                    # record every per-step belief output
        model, cfg = original(path)
        forward = model.belief.forward

        def recorded(*a, **k):
            y, c = forward(*a, **k)
            if y.shape[1] == 2 and y.shape[0] > 1:                     # one generated step for a batch of draws
                log.append(y[:, -1].detach().float().cpu())
            return y, c
        model.belief.forward = recorded
        return model, cfg
    SL.CV.IF.load = load
    for bank, (src, out) in BANKS.items():
        for name, ck in CKPTS.items():
            d = out / name
            d.mkdir(parents=True, exist_ok=True)
            if (d / "beliefs.pt").exists():
                continue
            (d / "bank.pt").write_bytes(src.read_bytes())
            log.clear()
            SL.HERE, SL.CKPT = d, ck
            SL.run()
            n = len(torch.load(d / "bank.pt", map_location="cpu", weights_only=False)["rows"])
            assert len(log) == -(-n // 8) * SL.H, (len(log), n)           # only the generation steps were recorded
            torch.save(torch.stack(log), d / "beliefs.pt")              # [batches x steps, rows, width], batch-major
            print(bank, name, len(log), flush=True)


def read():
    L = dict(np.load(G.S.LABELS / "s0_dev_labels.npz"))
    y, ok = G.LB.events(L)
    fold = np.repeat(np.arange(len(L["ep_T"])) % 5, L["ep_T"] + 1)
    res = {"known_answer_quiet_bank_equals_queue": bool(np.array_equal(
        torch.load(HERE / "run/rollout_belief/F6.1/run.pt", map_location="cpu", weights_only=False)["generated"],
        torch.load(G.EXP / "20261114_p1f61/readings/followups/sleep/run.pt", map_location="cpu", weights_only=False)["generated"]))}
    for name in CKPTS:
        X = torch.as_tensor(np.load(HERE / f"run/X_{name}.npy")).float()
        p = G.probe(X, y, ok, fold)["energy_up_asleep"]
        te = (fold == 4) & ok[:, 3]
        dev_prob = torch.sigmoid(torch.as_tensor(p["logit"])).numpy()
        res[name] = {"dev_fold4": {"mean_p_at_rises": float(dev_prob[y[te, 3]].mean()), "mean_p_elsewhere": float(dev_prob[~y[te, 3]].mean())}}
        for bank, (_, out) in BANKS.items():
            d = out / name
            z = torch.load(d / "run.pt", map_location="cpu", weights_only=False)
            a, b, h, k = z["generated"], z["truth"], z["H"], z["K"]
            n = len(z["rows"])
            bel = torch.load(d / "beliefs.pt")                          # per batch of 8 scenes: h steps x (scenes x draws)
            prob = np.zeros((n, k, h + 1))                              # P(rise at generated step j | belief before j), j >= 1
            for i, start in enumerate(range(0, n, 8)):
                m = min(8, n - start)
                for s in range(h):
                    zz = (bel[i * h + s][:m * k] - p["mu"]) / p["sd"]
                    prob[start:start + m, :, s + 1] = torch.sigmoid(zz @ p["w"][:-1] + p["w"][-1]).reshape(m, k).numpy()
            prob = prob[:, :, :h]
            dd = HZ.first(b[..., 0] >= 9, h).min(1)                       # the phase's own rise step (hazard.py)
            live = a[..., 3].astype(bool) & ~a[..., 2].astype(bool)
            at8 = np.ones(a.shape[:3], bool)
            at8[:, :, 1:] = np.cumprod(live[:, :, :-1] & (a[:, :, :-1, 0] < 9) & (a[:, :, :-1, 1] > .5), 2).astype(bool)
            at8[:, :, 0] = False                                        # step 0's belief is the recorded prefix's, not logged
            rise = live & (a[..., 0] >= 9)
            rel = np.arange(h)[None, None] - dd[:, None, None]
            row = {}
            for label, m in (("before_due", rel < 0), ("at_due", rel == 0), ("1-3_after", (rel >= 1) & (rel <= 3)), ("4+_after", rel >= 4)):
                sel = at8 & m
                row[label] = {"at_risk": int(sel.sum()), "belief_p_rise": float(prob[sel].mean()) if sel.any() else None,
                              "drawn_rise": float(rise[sel].mean()) if sel.any() else None}
            res[name][bank] = row
            print(name, bank, json.dumps(row), flush=True)
    (HERE / "rollout_belief.json").write_text(json.dumps(res, indent=1) + "\n")


if __name__ == "__main__":
    {"run": run, "read": read}[sys.argv[1]]()
