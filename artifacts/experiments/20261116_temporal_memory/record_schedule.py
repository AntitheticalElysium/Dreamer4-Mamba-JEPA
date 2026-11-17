"""Temporal memory pass, test 2 added reading (PLAN.md; Codex 11-16): check 17's native read (drawn rate and first-step
expectation, both energy / hunger banks: original and on-schedule) on the S4D-Lin record fine-tune at 4,000 and 6,000.

    MAMBA_DETERMINISTIC=1 python -m artifacts.experiments.20261116_temporal_memory.record_schedule
"""
import importlib
import json
import sys

import torch

SP = importlib.import_module("artifacts.experiments.20261115_f61_dissection.schedule_pairs")
RFT = importlib.import_module("artifacts.experiments.20261116_temporal_memory.record_ft")
TF, DS, TJ = SP.TF, SP.DS, SP.TJ

if __name__ == "__main__":
    TF.traces, TF.install = RFT.record, RFT.install
    model, cfg = DS.NP.IF.load(TF.CKPT.resolve())
    for p in model.parameters():
        p.requires_grad_(False)
    TF.install(model.denoiser)
    model.denoiser.trace_proj.requires_grad_(False)                     # use_check.load: no autocast weight cache
    model.eval()
    readers = TJ.readers(model, cfg, slots=(64, 66))
    res = {}
    for u in [int(a) for a in sys.argv[1:]] or (4000, 6000):
        model.denoiser.load_state_dict(torch.load(RFT.RUN / f"denoiser_{u}.pt", map_location=DS.DEV, weights_only=False)["denoiser"])
        res[f"record_{u}"] = {}
        for name, rows, slot, sign in SP.banks():
            res[f"record_{u}"][name] = SP.read_bank(model, cfg, rows, slot, sign, readers[slot][0])
            print(u, name, json.dumps(res[f"record_{u}"][name]), flush=True)
    (RFT.HERE / "record_schedule.json").write_text(json.dumps(res, indent=1) + "\n")
