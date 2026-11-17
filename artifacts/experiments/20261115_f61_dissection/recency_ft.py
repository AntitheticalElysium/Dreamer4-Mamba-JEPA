"""F6.1 dissection, check 16 (PLAN.md): input form or incentive? check 12 rerun with each periphery cell's exact recency
(frames since its own token last changed by more than 1, check 11's threshold) as a zero-initialised embedding added to
that cell's denoiser token, in place of the soft traces. Everything else is traces_ft's (same fine-tune, reads, seeds),
then use_check.expect on the result.

    MAMBA_DETERMINISTIC=1 python -m artifacts.experiments.20261115_f61_dissection.recency_ft
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
RUN = HERE / "run/recency"
CAP = 63


def recency(p, first, e0, p0=None):
    """traces_ft.traces' contract with the state u = max(0, CAP - frames since the cell's last change > 1) in both
    channels: CAP right after a change, 0 when none in the last CAP frames or since the episode start."""
    x = torch.zeros(p.shape[:3], device=p.device)
    x[:, 1:] = (p[:, 1:] - p[:, :-1]).float().square().sum(-1)
    if p0 is not None:
        x[:, 0] = (p[:, 0] - p0).float().square().sum(-1)
    x = x * (~first)[..., None]
    e, inp, st = e0, [], []
    for i in range(p.shape[1]):
        e = torch.where(first[:, i, None, None], torch.zeros_like(e), e)
        inp.append(e)
        e = torch.where((x[:, i] > 1)[..., None], torch.full_like(e, CAP), (e - 1).clamp(min=0))
        st.append(e)
    return torch.stack(inp, 1), torch.stack(st, 1)


def install(den):
    """traces_ft.install with an embedding of u (zero-initialised) in place of the linear map of log(1 + trace)."""
    den.trace_proj = nn.Embedding(CAP + 1, den.W).to(next(den.parameters()).device)
    nn.init.zeros_(den.trace_proj.weight)
    embed = den.embed

    def with_recency(x, memory, fix):
        tok, ang = embed(x, memory, fix)
        tr = den.trace_input if den.trace_input.shape[0] == tok.shape[0] else den.trace_sub
        add = den.trace_proj(tr[..., 0].round().long().clamp(0, CAP)).to(tok.dtype)
        return torch.cat([tok[..., :TF.N, :] + add, tok[..., TF.N:, :]], -2), ang
    den.embed = with_recency
    loss = TF.M.shortcut_loss

    def shortcut_loss(d, x0, prev_action, memory, fix, episode, valid, sc, bootstrap, start, levels=None, noise=None):
        d.trace_sub = d.trace_input[bootstrap]
        return loss(d, x0, prev_action, memory, fix, episode, valid, sc, bootstrap, start, levels, noise)
    TF.M.shortcut_loss = shortcut_loss


@torch.no_grad()
def pair_recency():
    """u at the digit's cell for the due frame, per bank and branch (the input the read gives)."""
    model, cfg = TF.DS.NP.IF.load(TF.CKPT.resolve())
    out = {}
    for name, rows, slot, sign in TF.pairs():
        u = []
        for prefix, frames, rewards, terms, q, acts, truth in rows:
            ft = torch.as_tensor(np.concatenate([np.repeat(prefix[0][None], 2, 0), frames[:, :q + 1]], 1), device=TF.DS.DEV)
            pc = torch.stack([TF.C.coarse(model.perception, ft[b])[0] for b in (0, 1)])
            first = torch.zeros(pc.shape[:2], dtype=torch.bool, device=pc.device); first[:, 0] = True
            _, st = recency(pc, first, torch.zeros(2, TF.N, 2, device=pc.device))
            u.append(st[:, -1, slot, 0].tolist())
        u = np.array(u)
        out[name] = {"due_mean_frames_since": float(CAP - u[:, 0].mean()), "delayed_mean_frames_since": float(CAP - u[:, 1].mean()),
                     "differ": float((u[:, 0] != u[:, 1]).mean()), "per_pair_u": u.tolist()}
        print(name, json.dumps({k: v for k, v in out[name].items() if k != "per_pair_u"}), flush=True)
    del model
    torch.cuda.empty_cache()
    return out


if __name__ == "__main__":
    RUN.mkdir(parents=True, exist_ok=True)
    TF.traces, TF.install, TF.OUT, TF.HERE, UC.HERE = recency, install, RUN, RUN, RUN   # traces_ft.main / use_check.expect read these
    (RUN / "pair_recency.json").write_text(json.dumps(pair_recency(), indent=1) + "\n")
    TF.main()                                                           # reads at 0, 1,000, 2,000 -> RUN/traces_ft.json
    UC.expect()                                                         # first-step expectation -> RUN/expect.json
    for a, b in (("traces_ft.json", "recency_ft.json"), ("expect.json", "recency_expect.json"), ("pair_recency.json", "recency_pairs.json")):
        shutil.copy(RUN / a, HERE / b)
