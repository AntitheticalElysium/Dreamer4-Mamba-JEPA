"""Memorization control: does the winning rung survive destroyed action-outcome correspondence?

`pooled*_pca192` reaches a perfect 36/36. A perfect score on 36 roots deserves the control the
plan requires: permute death labels WITHIN each root, which preserves every marginal -- the
number of fatal forks per root, the per-action base rates -- and destroys only which action was
fatal. A head that still scores well is fitting the panel, not reading the representation.
"""
import json, sys
from pathlib import Path
import numpy as np, torch
ROOT = Path("/home/antithetical/EPITA/PERSO/DynamicHorizons-Mamba-JEPA"); HERE = Path(__file__).parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(HERE))
from readout import Fit, evaluate, fit_head, standardize
from phase2 import apply_pca, fit_pca, WIDTH

SIDECAR = ROOT/"artifacts/lewm_gates_20260906/m03_bootstrap/evaluation_v2/sidecar/sidecar.probe_only.pt"
side = torch.load(SIDECAR, map_location="cpu", weights_only=False)["splits"]
y = {s: side[s]["outcomes"].float() for s in ("train","dev")}
taps = torch.load(HERE/"cache/taps.mamba_raw.pt", map_location="cpu", weights_only=False)
spec, out = Fit(steps=2000), []
for tap in ("pooled4","pooled2"):
    raw = {s: taps[s]["successor"][tap].float().flatten(2) for s in ("train","dev")}
    mean, basis, _ = fit_pca(raw["train"], WIDTH)
    rung = {s: apply_pca(raw[s], mean, basis) for s in ("train","dev")}
    xtr, xdv = standardize(rung["train"], rung["dev"])
    for permute in (False, True):
        got = []
        for seed in (0,1,2):
            m,_,p = fit_head(xtr, y["train"], family="mlp128", objective="rank", seed=seed,
                             spec=spec, device="cuda", permute_labels=permute)
            got.append({"train": evaluate(m,xtr,y["train"],"cuda"), "dev": evaluate(m,xdv,y["dev"],"cuda")})
        row = {"rung": f"{tap}_pca{WIDTH}", "permuted_labels": permute, "parameters": p,
               "train_safe": [g["train"]["safe_choice"] for g in got],
               "train_opportunity": got[0]["train"]["opportunity_roots"],
               "dev_safe": [g["dev"]["safe_choice"] for g in got],
               "dev_opportunity": got[0]["dev"]["opportunity_roots"]}
        out.append(row); print(json.dumps(row), flush=True)
(HERE/"evidence/phase2").mkdir(parents=True, exist_ok=True)
(HERE/"evidence/phase2/memorization.json").write_text(json.dumps({"schema":"d4mj_memorization_v1","rows":out}, indent=2)+"\n")
print(json.dumps({"status":"memorization_complete"}), flush=True)
