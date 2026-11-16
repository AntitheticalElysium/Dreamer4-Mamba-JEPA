"""Temporal memory pass, test 3 (PLAN.md): test 2 with a nonlinear per-cell read of the record (MLP, last layer
zero-initialised) in place of the linear map; 4,000 updates, then the natural catalogue and check 17's native read.

    MAMBA_DETERMINISTIC=1 python -m artifacts.experiments.20261116_temporal_memory.record_mlp_ft
"""
import importlib
import json
import shutil

import torch
import torch.nn as nn

RFT = importlib.import_module("artifacts.experiments.20261116_temporal_memory.record_ft")
TF, UC = RFT.TF, RFT.UC
RUN = RFT.HERE / "run/record_mlp"


def install(den):
    """record_ft.install (the same additive path and bootstrap rows), its map replaced by the per-cell MLP."""
    RFT.install(den)
    dev = next(den.parameters()).device
    den.trace_proj = nn.Sequential(nn.Linear(RFT.D, den.W), nn.GELU(), nn.Linear(den.W, den.W)).to(dev)
    nn.init.zeros_(den.trace_proj[2].weight); nn.init.zeros_(den.trace_proj[2].bias)


if __name__ == "__main__":
    RUN.mkdir(parents=True, exist_ok=True)
    TF.traces, TF.install, TF.OUT, TF.HERE = RFT.record, install, RUN, RUN
    TF.UPDATES, TF.READS = 4000, (2000, 4000)
    TF.main()                                                           # RUN/traces_ft.json, RUN/denoiser_XXXX.pt
    shutil.copy(RUN / "traces_ft.json", RFT.HERE / "record_mlp_ft.json")
    cat = RUN / "cat4000"
    cat.mkdir(exist_ok=True)
    (cat / "denoiser_2000.pt").unlink(missing_ok=True)
    (cat / "denoiser_2000.pt").symlink_to(RUN / "denoiser_4000.pt")     # use_check.load reads TF.OUT / denoiser_2000.pt
    TF.OUT, UC.HERE = cat, cat
    UC.catalogue()
    shutil.copy(cat / "catalogue.json", RFT.HERE / "record_mlp_catalogue_4000.json")
    SP = importlib.import_module("artifacts.experiments.20261115_f61_dissection.schedule_pairs")
    model, cfg = SP.DS.NP.IF.load(TF.CKPT.resolve())
    for p in model.parameters():
        p.requires_grad_(False)
    install(model.denoiser)
    model.denoiser.trace_proj.requires_grad_(False)                     # use_check.load: no autocast weight cache
    model.eval()
    model.denoiser.load_state_dict(torch.load(RUN / "denoiser_4000.pt", map_location=SP.DS.DEV, weights_only=False)["denoiser"])
    readers = SP.TJ.readers(model, cfg, slots=(64, 66))
    res = {"record_mlp_4000": {}}
    for name, rows, slot, sign in SP.banks():
        res["record_mlp_4000"][name] = SP.read_bank(model, cfg, rows, slot, sign, readers[slot][0])
        print(4000, name, json.dumps(res["record_mlp_4000"][name]), flush=True)
    (RFT.HERE / "record_mlp_schedule.json").write_text(json.dumps(res, indent=1) + "\n")
