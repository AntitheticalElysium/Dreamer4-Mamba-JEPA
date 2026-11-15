"""Temporal memory pass, measurement 3 (PLAN.md): where does a learned Mamba memory lose readable time?

Seven memories trained from scratch in lockstep on the same packed training-split episodes (F6.1's frozen coarse
tokens and the actions), under one label-free objective: every cell's next-frame change (squared token change > 1,
measurement 1's event). Read at the clock events of DEV fold 4 (measurement 1's rows) from each model's own head.

    MAMBA_DETERMINISTIC=1 python -m artifacts.experiments.20261116_temporal_memory.architecture tokens
    MAMBA_DETERMINISTIC=1 python -m artifacts.experiments.20261116_temporal_memory.architecture train SEED
"""
import importlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

HERE = Path(__file__).resolve().parent
RUN = HERE / "run"
ROOT = HERE.parents[2]
UC = importlib.import_module("artifacts.experiments.20261115_f61_dissection.use_check")
G, TF = UC.G, UC.TF
from saccade import data as D                                           # noqa: E402
from saccade import model as M                                          # noqa: E402

CFG = yaml.safe_load(open(ROOT / "saccade/m0.yaml"))
DEV = torch.device("cuda")
N, TOK, ACTIONS, DIGITS = 81, 16, 17, [63, 64, 65, 66]
BUDGET, UPDATES, READS, WARMUP = 4096, 3000, (1000, 2000, 3000), 250   # frames per update; the belief's 1,024,000
LR, WD, CLIP = 1e-4, 0.01, 1.0                                          # warm-up frames / 4,096; m0.yaml optimizer
EVENTS = {"food_down": (0, 64), "drink_down": (1, 65), "energy_down_awake": (2, 66), "energy_up_asleep": (3, 66)}


# ---- data: F6.1's frozen coarse tokens for the training split and DEV fold 4 ----
@torch.no_grad()
def tokens():
    model, _ = G.CR.NP.IF.load(TF.CKPT.resolve())
    model.to(DEV)
    sp = D.load_split(CFG)
    L = dict(np.load(G.S.LABELS / "s0_dev_labels.npz"))
    dev = G.S.episodes(L)
    for name, eps in (("train", sp["train"]), ("fold4", [dev[e] for e in range(len(dev)) if e % 5 == 4])):
        off = np.r_[0, np.cumsum([e.T + 1 for e in eps])]
        p = np.lib.format.open_memmap(RUN / f"arch_{name}_p.npy", "w+", np.float16, (int(off[-1]), N, TOK))
        a = np.zeros(off[-1], np.int64)
        for i, e in enumerate(eps):
            f = G.D.RP.episode_fields(e.shard, e.slot)
            fr = f["observations"]
            p[off[i]:off[i + 1]] = torch.cat([G.C.coarse(model.perception, z.to(DEV))[0] for z in fr.split(512)]).half().cpu().numpy()
            a[off[i]:off[i + 1] - 1] = f["actions_taken"].numpy()
        p.flush()
        np.save(RUN / f"arch_{name}_meta.npy", {"off": off, "action": a})
        print(name, len(eps), "episodes", off[-1], "frames", flush=True)


def load(name):
    meta = np.load(RUN / f"arch_{name}_meta.npy", allow_pickle=True).item()
    return np.load(RUN / f"arch_{name}_p.npy", mmap_mode="r"), meta["off"], meta["action"]


def pack(p, off, action, eps):
    """Whole episodes concatenated (cut at BUDGET): tokens [L, N, 16], actions [L], segment index [L] (1, 2, ...),
    next-change target [L, N] and its mask (the last frame of each episode has none)."""
    segs, total = [], 0
    for e in eps:
        a, b = off[e], off[e + 1]
        segs.append((a, min(b, a + BUDGET - total)))
        total += segs[-1][1] - a
        if total >= BUDGET:
            break
    x = torch.as_tensor(np.concatenate([p[a:b] for a, b in segs]).astype(np.float32))
    act = torch.as_tensor(np.concatenate([action[a:b] for a, b in segs]))
    seg = torch.as_tensor(np.concatenate([np.full(b - a, i + 1) for i, (a, b) in enumerate(segs)]))
    nxt = torch.cat([x[1:], x[-1:]])
    tgt = ((nxt - x).square().sum(-1) > 1.0).float()
    ok = torch.cat([seg[1:] == seg[:-1], torch.tensor([False])])
    ok &= torch.as_tensor(np.concatenate([np.arange(a, b) < off[np.searchsorted(off, a, "right")] - 1 for a, b in segs]))
    return x, act, seg, tgt, ok


# ---- memories ----
def mamba(width, headdim, kind):
    if kind == "m2":
        return M.FunctionalMamba2({**CFG["belief"], "width": width, "headdim": headdim, "backend": "triton"})
    from mamba_ssm.modules.mamba3 import Mamba3                         # its defaults (SISO, rope_fraction 0.5)
    return Mamba3(d_model=width, d_state=CFG["belief"]["d_state"], expand=CFG["belief"]["expand"], headdim=headdim,
                  is_mimo=False, chunk_size=64)


class Stack(nn.Module):
    """The belief's residual stack (6 x RMSNorm + mixer, final RMSNorm) with Mamba-2 or Mamba-3 mixers."""

    def __init__(self, width, headdim, kind):
        super().__init__()
        self.kind = kind
        self.norms = nn.ModuleList([nn.RMSNorm(width, eps=1e-5) for _ in range(CFG["belief"]["layers"])])
        self.mixers = nn.ModuleList([mamba(width, headdim, kind) for _ in range(CFG["belief"]["layers"])])
        self.final = nn.RMSNorm(width, eps=1e-5)

    def forward(self, x, seg):                                          # x [R, L, W], seg [L] (shared by the rows)
        R, L, W = x.shape
        if self.kind == "m3":                                           # rows x episodes as one varlen sequence
            starts = torch.nonzero(torch.diff(seg, prepend=seg[:1] - 1)).flatten()
            cu = torch.cat([(starts[None] + L * torch.arange(R, device=x.device)[:, None]).flatten(),
                            torch.tensor([R * L], device=x.device)]).to(torch.int32)
        for norm, mixer in zip(self.norms, self.mixers):
            if self.kind == "m2":
                y, _ = mixer(norm(x), mixer.initial(R, x.device), seg[None].expand(R, -1))
            else:
                y = mixer(norm(x).reshape(1, R * L, W), cu_seqlens=cu).reshape(R, L, W)
            x = x + y
        return self.final(x)


class Head(nn.Module):
    """Shared readout form: a cell's memory feature, its current token and its identity -> logit of a change."""

    def __init__(self):
        super().__init__()
        self.cell = nn.Embedding(N, 16)
        self.mlp = nn.Sequential(nn.Linear(32 + TOK + 16, 64), nn.GELU(), nn.Linear(64, 1))

    def forward(self, f, x, cells):                                     # [L, n, 32], [L, n, 16], n cell indices
        return self.mlp(torch.cat([f, x, self.cell.weight[cells].expand(x.shape[0], -1, -1)], -1))[..., 0]


class Global(nn.Module):
    """The belief's form: the whole percept flattened through one linear map into a 256-wide Mamba stack."""

    def __init__(self, kind, readout):
        super().__init__()
        self.readout = readout
        self.inp, self.act = nn.Linear(N * TOK, 256), nn.Embedding(ACTIONS, 256)
        self.stack = Stack(256, CFG["belief"]["headdim"], kind)
        if readout == "direct":
            self.read = nn.Linear(256, N * 32)
        else:                                                           # K = 4 memory tokens read by each cell
            self.mem, self.query, self.read = nn.Linear(256, 4 * 256), nn.Linear(TOK + 16, 256), nn.Linear(256, 32)
            self.qcell = nn.Embedding(N, 16)
        self.head = Head()

    def forward(self, x, act, seg, cells):
        b = self.stack((self.inp(x.flatten(-2)) + self.act(act))[None], seg)[0]          # [L, 256]
        if self.readout == "direct":
            f = self.read(b).unflatten(-1, (N, 32))[:, cells]
        else:
            mt = self.mem(b).unflatten(-1, (4, 256))                                     # [L, 4, 256]
            q = self.query(torch.cat([x[:, cells], self.qcell.weight[cells].expand(x.shape[0], -1, -1)], -1))
            w = torch.softmax(q @ mt.transpose(1, 2) / 16.0, -1)                         # [L, n, 4]
            f = self.read(w @ mt)
        return self.head(f, x[:, cells], cells)


class PerCell(nn.Module):
    """One Mamba stack shared by every cell, each cell its own sequence and state (Po et al.'s block-wise scan at
    block size 1); input = the cell's token + its identity + the action."""

    def __init__(self, kind):
        super().__init__()
        self.inp, self.cell, self.act = nn.Linear(TOK, 64), nn.Embedding(N, 64), nn.Embedding(ACTIONS, 64)
        self.stack, self.read, self.head = Stack(64, 16, kind), nn.Linear(64, 32), Head()

    def forward(self, x, act, seg, cells):                              # cells are independent: any subset is exact
        u = self.inp(x[:, cells]) + self.cell.weight[cells] + self.act(act)[:, None]    # [L, n, 64]
        h = self.stack(u.transpose(0, 1).contiguous(), seg).transpose(0, 1)
        return self.head(self.read(h), x[:, cells], cells)


ARMS = {"G2": (lambda: Global("m2", "direct"), "all"), "G2K": (lambda: Global("m2", "memory"), "all"),
        "P2": (lambda: PerCell("m2"), "all"), "G3": (lambda: Global("m3", "direct"), "all"),
        "P3": (lambda: PerCell("m3"), "all"), "G2_digits": (lambda: Global("m2", "direct"), "digits"),
        "P2_digits": (lambda: PerCell("m2"), "digits")}
ALL = list(range(N))


def groups(model, target):
    """Cell groups per pass: per-cell memories in thirds (memory; the gradient summed over groups is the full one),
    the digit-only objective on its four cells; global memories in one pass."""
    if target == "digits":
        return [DIGITS]
    return [ALL[i:i + 27] for i in range(0, N, 27)] if isinstance(model, PerCell) else [ALL]


def loss_of(logit, tgt, ok, cells):
    """Sum of log-losses over these cells' valid frames, divided by (valid frames x the objective's cell count)."""
    l = F.binary_cross_entropy_with_logits(logit.float(), tgt[:, cells], reduction="none")
    return (l * ok[:, None]).sum()


@torch.no_grad()
def evaluate(model, fold4):
    """DEV fold 4, every episode from its zero state: P(change into t) at the digit cells from frame t - 1, AP at
    measurement 1's rows; generic next-change log-loss over all cells and over the digit cells."""
    p, off, action = fold4
    model.eval()
    probs, nll, nll_d, cnt = [], 0.0, 0.0, 0
    e = 0
    while e < len(off) - 1:
        k = e
        while k < len(off) - 1 and off[k + 1] - off[e] <= BUDGET:
            k += 1
        k = max(k, e + 1)
        x, act, seg, tgt, ok = (z.to(DEV) for z in pack(p, off, action, range(e, k)))
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logit = torch.cat([model(x, act, seg, c).float() for c in groups(model, "all")], 1)
        l = F.binary_cross_entropy_with_logits(logit, tgt, reduction="none")
        nll += float((l * ok[:, None]).sum()); nll_d += float((l[:, DIGITS] * ok[:, None]).sum()); cnt += int(ok.sum())
        probs.append(torch.sigmoid(logit[:, DIGITS]).cpu().numpy())
        e = k
    model.train()
    probs = np.concatenate(probs)
    L = dict(np.load(G.S.LABELS / "s0_dev_labels.npz"))
    y, _ = G.LB.events(L)
    after = G.cohorts(L)["after_reset"]
    loff = np.r_[0, np.cumsum(L["ep_T"] + 1)]
    fold = [e for e in range(len(L["ep_T"])) if e % 5 == 4]
    out = {"nll_all_cells": nll / (cnt * N), "nll_digit_cells": nll_d / (cnt * 4)}
    for n, (col, cell) in EVENTS.items():
        s, t_ = [], []
        for i, e in enumerate(fold):
            ts = np.array([t for t in range(9, L["ep_T"][e] + 1) if after[loff[e] + t, col]], int)
            if len(ts):
                s.append(probs[off[i] + ts - 1, DIGITS.index(cell)]); t_.append(y[loff[e] + ts, col])
        out[n] = G.AP(np.concatenate(t_), np.concatenate(s))
    return out


def train(seed):
    torch.manual_seed(seed)
    tr, fold4 = load("train"), load("fold4")
    models, opts = {}, {}
    for i, (name, (make, _)) in enumerate(ARMS.items()):
        torch.manual_seed(seed * 100 + i)
        models[name] = make().to(DEV)
        decay = [q for q in models[name].parameters() if q.ndim >= 2 and q.requires_grad]
        rest = [q for q in models[name].parameters() if q.ndim < 2 and q.requires_grad]
        opts[name] = torch.optim.AdamW([{"params": decay, "weight_decay": WD}, {"params": rest, "weight_decay": 0.0}],
                                       lr=LR, betas=(0.9, 0.999), eps=1e-8)
    rng = np.random.default_rng(seed)
    out = {"seed": seed, "params": {n: sum(q.numel() for q in m.parameters()) for n, m in models.items()},
           "log": [], "reads": {}}
    t0 = time.time()
    for u in range(1, UPDATES + 1):
        lr = LR * (min(1.0, u / WARMUP) * (0.1 + 0.9 * 0.5 * (1 + np.cos(np.pi * u / UPDATES))))
        x, act, seg, tgt, ok = (z.to(DEV) for z in pack(*tr, rng.permutation(len(tr[1]) - 1)))
        losses = {}
        for name, (_, target) in ARMS.items():
            for g in opts[name].param_groups:
                g["lr"] = lr
            opts[name].zero_grad(set_to_none=True)
            denom, total = ok.sum() * (len(DIGITS) if target == "digits" else N), 0.0
            for c in groups(models[name], target):                      # gradients accumulate over cell groups
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    loss = loss_of(models[name](x, act, seg, c), tgt, ok, c) / denom
                loss.backward()
                total += float(loss)
            torch.nn.utils.clip_grad_norm_(models[name].parameters(), CLIP)
            opts[name].step()
            losses[name] = total
        if u % 100 == 0:
            out["log"].append({"update": u, "sec": time.time() - t0, **losses})
            print(json.dumps(out["log"][-1]), flush=True)
        if u in READS:
            out["reads"][u] = {name: evaluate(m, fold4) for name, m in models.items()}
            print(u, json.dumps(out["reads"][u]), flush=True)
            (HERE / f"architecture_seed{seed}.json").write_text(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    RUN.mkdir(parents=True, exist_ok=True)
    if sys.argv[1] == "tokens":
        tokens()
    else:
        train(int(sys.argv[2]))
