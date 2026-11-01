"""CPU checks of the corrections to Codex's review of b90e61dc (findings 1 and 3), on the corrected functions.
Usage: python -m artifacts.experiments.20261030_m05_guided_fovea.checks.verify_review1
"""
import importlib, json
from pathlib import Path

import numpy as np
import torch

E = importlib.import_module("artifacts.experiments.20261030_m05_guided_fovea.evaluate")
OUT = Path(__file__).resolve().parent / "verify_review1.json"
res = {}

# 1. Codex's counterexample: 100 eligible samples, 10 verified successes, 90 unknown; the event recorded on 10 rows
#    chosen independently of readability.
n = 100
ok = np.r_[np.ones(10, bool), np.zeros(90, bool)]
oc = {"use": np.ones(n, bool), "eligible": np.ones(n, bool), "readable": ok, "unknown": ~ok, "start": np.zeros(n, int),
      "end": np.ones(n, int), "recorded": np.r_[np.zeros(45, bool), np.ones(10, bool), np.zeros(45, bool)],
      "success": ok.copy(), "copy": np.zeros(n, bool), "other": np.zeros(n, bool), "false_change": np.zeros(n, bool),
      "real_read": np.ones(n, bool), "real_unreadable": np.zeros(n, bool), "real_correct": np.ones(n, bool),
      "coherence_ok": np.zeros(n, bool), "coherent": np.zeros(n, bool)}
s = E.summarise(oc, np.arange(n), "drink")["appearance"]
res["counterexample"] = {k: (None if s[k] is None else round(s[k]["est"], 4))
                         for k in ("success", "unknown", "success_upper_bound", "success_among_readable", "recorded")}
assert abs(s["success"]["est"] - 0.10) < 1e-9 and abs(s["unknown"]["est"] - 0.90) < 1e-9
assert abs(s["success_upper_bound"]["est"] - 1.0) < 1e-9 and abs(s["recorded"]["est"] - 0.10) < 1e-9
# paired: B has more unknowns on the same population; the verified difference stays over the population
ob = {**oc, "readable": np.r_[np.ones(5, bool), np.zeros(95, bool)]}
ob.update(unknown=~ob["readable"], success=ob["readable"].copy())
d = E.paired_prediction(oc, ob, np.arange(n), "drink")["appearance"]
res["paired_counterexample"] = {"success": round(d["success"]["est"], 4), "unknown": round(d["unknown"]["est"], 4),
                                "among_readable": round(d["success_among_readable"]["est"], 4)}
assert abs(d["success"]["est"] + 0.05) < 1e-9 and abs(d["unknown"]["est"] - 0.05) < 1e-9

# 3. The real-token reference is the observed value: Codex's rows (first 1,000 of each damage class).
L = dict(np.load(E.EV.ROOT / "artifacts/eda/s0/s0_dev_labels.npz"))
dummy = [{"p": range(int(t) + 1)} for t in L["ep_T"]]
ctx = E.FC.contexts(L, dummy, np.random.default_rng(2029))
s_off = np.r_[0, np.cumsum(L["ep_T"] + 1)]
res["health_reference"] = {}
for kind in ("adjacent", "risky"):
    rows = ctx[kind][:1000]
    obs = E.observed(kind, rows, np.arange(len(rows)), L)
    actual = L["s_body"][[s_off[r[0]] + r[1] for r in rows], 0].clip(min=0)
    start, end = np.array([r[3] for r in rows]), np.array([r[4] for r in rows])
    old = np.where(actual == end, end, start)
    res["health_reference"][kind] = {"rows": len(rows), "observed_equals_labels": bool((obs == actual).all()),
                                     "rows_the_old_reference_misstated": int((old != actual).sum())}
    assert (obs == actual).all()

# the corrected glimpse reader interface (eligibility first, reading second), CPU stand-in network
class AlwaysOne(torch.nn.Module):
    def forward(self, x):
        return torch.stack([torch.zeros(len(x), device=x.device), torch.ones(len(x), device=x.device)], -1)


old = E.IN.DEV
E.IN.DEV = torch.device("cpu")
rd = {"g": {("drink", 65): {"net": AlwaysOne(), "classes": np.array([0, 1]), "qualified": True,
                            "support": {(0, 1): (torch.ones(64), 1.0)}}}}
elig = E.eligible_glimpse(rd, "drink", np.array([65, 65, 66]), np.array([0, 0, 0]), np.array([1, 1, 1]), np.ones(3, bool))
tok = torch.ones(3, 9, 64); tok[1] += 5.0                                 # row 1 outside the support
val, ok_ = E.read_glimpse(rd, "drink", tok, np.array([4, 4, 4]), np.array([65, 65, 66]), elig)
E.IN.DEV = old
res["glimpse_interface"] = {"eligible": elig.tolist(), "read": val.tolist(), "readable": ok_.tolist()}
assert elig.tolist() == [True, True, False] and val.tolist()[:2] == [1, 1] and ok_.tolist() == [True, False, False]
OUT.write_text(json.dumps(res, indent=1))
print(json.dumps(res, indent=1))
