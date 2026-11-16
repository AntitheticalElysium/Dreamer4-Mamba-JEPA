"""Temporal memory pass, test 2 (PLAN.md): check 18's fine-tune with a fixed S4D-Lin record of each cell's raw token in
place of the exact recency, then check 16b's natural catalogue at 6,000.

    MAMBA_DETERMINISTIC=1 python -m artifacts.experiments.20261116_temporal_memory.record_ft
"""
import importlib
import json
import shutil
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

HERE = Path(__file__).resolve().parent
TF = importlib.import_module("artifacts.experiments.20261115_f61_dissection.traces_ft")
UC = importlib.import_module("artifacts.experiments.20261115_f61_dissection.use_check")
RUN = HERE / "run/record"
N, CH, MODES = 81, 16, 32
DT = 1.0 / 32
A = torch.complex(torch.full((MODES,), -0.5, dtype=torch.float64), np.pi * torch.arange(MODES, dtype=torch.float64))
ABAR = (1 + DT * A / 2) / (1 - DT * A / 2)                              # the S4D listing's bilinear discretisation, B = 1
BBAR = DT / (1 - DT * A / 2)
D = 2 * CH * MODES                                                      # real and imaginary parts: 1,024 features


def record(p, first, e0, p0=None):
    """traces_ft.traces' contract: p [B, L, 81, 16] coarse tokens, first [B, L], e0 the state after the frame before
    frame 0 as features [B, 81, D] (an all-zero e0 of another width is the zero state). Returns the record through the
    previous frame for each frame [B, L, 81, D] and the state after each frame [B, L, 81, D]."""
    if e0.shape[-1] != D:
        assert not e0.any()
        e0 = e0.new_zeros(*e0.shape[:-1], D)
    dev = p.device
    a, b = ABAR.to(dev).to(torch.complex64), BBAR.to(dev).to(torch.complex64)
    s = torch.complex(e0[..., :CH * MODES], e0[..., CH * MODES:]).unflatten(-1, (CH, MODES))
    feat = lambda z: torch.cat([z.real.flatten(-2), z.imag.flatten(-2)], -1)
    inp, st = [], []
    for i in range(p.shape[1]):
        s = torch.where(first[:, i, None, None, None], torch.zeros_like(s), s)
        inp.append(feat(s))
        s = a * s + b * p[:, i].float()[..., None]
        st.append(feat(s))
    return torch.stack(inp, 1), torch.stack(st, 1)


def install(den):
    """traces_ft.install with a zero-initialised linear map of the record's features (no log)."""
    den.trace_proj = nn.Linear(D, den.W).to(next(den.parameters()).device)
    nn.init.zeros_(den.trace_proj.weight); nn.init.zeros_(den.trace_proj.bias)
    embed = den.embed

    def with_record(x, memory, fix):
        tok, ang = embed(x, memory, fix)
        tr = den.trace_input if den.trace_input.shape[0] == tok.shape[0] else den.trace_sub
        if tr.shape[-1] != D:                                           # the zero input of an untrained read
            tr = tr.new_zeros(*tr.shape[:-1], D)
        add = den.trace_proj(tr.float()).to(tok.dtype)
        return torch.cat([tok[..., :TF.N, :] + add, tok[..., TF.N:, :]], -2), ang
    den.embed = with_record
    loss = TF.M.shortcut_loss

    def shortcut_loss(d, x0, prev_action, memory, fix, episode, valid, sc, bootstrap, start, levels=None, noise=None):
        d.trace_sub = d.trace_input[bootstrap]
        return loss(d, x0, prev_action, memory, fix, episode, valid, sc, bootstrap, start, levels, noise)
    TF.M.shortcut_loss = shortcut_loss


if __name__ == "__main__":
    RUN.mkdir(parents=True, exist_ok=True)
    TF.traces, TF.install, TF.OUT, TF.HERE = record, install, RUN, RUN  # traces_ft.main reads these
    TF.UPDATES, TF.READS = 6000, (2000, 4000, 6000)
    TF.main()                                                           # RUN/traces_ft.json, RUN/denoiser_XXXX.pt
    shutil.copy(RUN / "traces_ft.json", HERE / "record_ft.json")
    cat = RUN / "cat6000"
    cat.mkdir(exist_ok=True)
    (cat / "denoiser_2000.pt").unlink(missing_ok=True)
    (cat / "denoiser_2000.pt").symlink_to(RUN / "denoiser_6000.pt")     # use_check.load reads TF.OUT / denoiser_2000.pt
    TF.OUT, UC.HERE = cat, cat
    UC.catalogue()
    shutil.copy(cat / "catalogue.json", HERE / "record_catalogue.json")
