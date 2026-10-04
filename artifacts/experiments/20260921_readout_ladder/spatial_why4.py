"""Post hoc, fourth step: is the damage forecastable from what the world sees, and does its predictor
compute it without writing it into the generated state?

spatial_why3: the per-tile worlds copy the health tile on 100% of damaging transitions (TRAIN and
DEV), and the continuation head's training success was a position shortcut. Why the copy? Three
explanations, each pointing at a different repair:

  not_forecastable  damage is not predictable from the context + action the world receives; L1
                    regression then returns the conditional median, "no change"   -> uncertainty model
  not_written       it is predictable, and the predictor's hidden state holds it, but the generated
                    state does not carry it                                        -> output/loss
  not_computed      it is predictable from the frames, but the predictor's hidden state does not
                    hold it                                                        -> what the world learns

One-step damage (health loss of 2 or more, deaths included) at a transition t -> t+1, with t's 4-frame
context. Probes fitted on the pool's TRAIN windows (main windows, transitions 3->4 and 4->5, plus
terminal windows' last transition), judged on DEV transitions (spatial_why2's DEV construction:
damage, terminal and ordinary), balanced BCE, three probe seeds, AUC of P(damage):
  frame_tokens  the real frame t's 81 layer-normed tokens + the action     (what the frame shows)
  hidden_T      the T world's predictor output tokens at t (history, before projection) + nothing
  hidden_TH     the same for TH
  generated_TH  TH's generated t+1 tokens with frame t's (the pair the health head reads)
Head: frozen_ladder's tokens_attn readout (learned positions, attention pooling) -> 512 -> 1, with a
learned action embedding added before pooling for frame_tokens.

Readings (committed before the run; post hoc), on DEV AUC:
  frame_tokens < 0.75                                     -> not_forecastable
  frame_tokens >= 0.85 and hidden_TH >= 0.85
      and generated_TH < 0.65                             -> not_written
  frame_tokens >= 0.85 and hidden_TH < 0.70               -> not_computed
  otherwise                                               -> mixed
"""

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

ROOT = Path("/home/antithetical/EPITA/PERSO/DynamicHorizons-Mamba-JEPA")
HERE = Path(__file__).parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

from d4mj.data import _sha256

from spatial import D, N, POOL, TOKENS, WIDTH, WORLDS, World, bridge  # noqa: E402
from spatial_why2 import dev_population  # noqa: E402

SEED, STEPS = 20260929, 3000


def train_transitions(gen, cap=16_000):
    """5-frame windows (4 context + successor) from the pool's TRAIN windows, damage oversampled
    by keeping every damaging transition and a seeded subsample of the rest."""
    pool = torch.load(POOL / "pool.pt", weights_only=False, mmap=True)
    rows, starts = [], []
    for t in (3, 4):
        main = ~pool["terminal"]
        rows.append(torch.where(main)[0]); starts.append(torch.full((int(main.sum()),), t - 3))
    term = torch.where(pool["terminal"])[0]
    rows.append(term); starts.append(torch.ones(len(term), dtype=torch.long))
    rows, starts = torch.cat(rows), torch.cat(starts)
    dh = pool["dh"][rows, starts + 3]
    damage = dh <= -2
    keep = torch.cat([torch.where(damage)[0],
                      torch.where(~damage)[0][torch.randperm(int((~damage).sum()), generator=gen)[:cap]]])
    rows, starts = rows[keep], starts[keep]
    z = torch.stack([pool["z"][r][s:s + 5] for r, s in zip(rows.tolist(), starts.tolist())])
    tokens = torch.stack([pool["tokens"][r][s:s + 5] for r, s in zip(rows.tolist(), starts.tolist())])
    actions = torch.stack([pool["actions"][r][s:s + 4] for r, s in zip(rows.tolist(), starts.tolist())])
    return {"z": z, "tokens": tokens, "actions": actions, "dh": pool["dh"][rows, starts + 3],
            "terminal": ~pool["alive"][rows, starts + 4]}


@torch.no_grad()
def world_features(world, config, population, device, batch=64):
    """Predictor output tokens at t (the history) and the generated t+1 tokens, fp16 on CPU."""
    from d4mj.train import autocast_context
    hidden, generated = [], []
    for i in range(0, len(population["tokens"]), batch):
        s = population["tokens"][i:i + batch].float().to(device)
        with autocast_context(config):
            predicted, history = world(s[:, :4], population["actions"][i:i + batch].to(device))
        hidden.append(history[:, 3].float().half().cpu())
        generated.append(predicted[:, 3].float().half().cpu())
    return torch.cat(hidden), torch.cat(generated)


class Probe(nn.Module):
    """frozen_ladder's tokens_attn readout (positions + attention pooling -> 512 -> 1), optional action."""

    def __init__(self, width, action):
        super().__init__()
        self.position = nn.Parameter(torch.zeros(TOKENS, width))
        self.action = nn.Embedding(N, width) if action else None
        self.value, self.score = nn.Linear(width, 128), nn.Linear(width, 1)
        self.net = nn.Sequential(nn.GELU(), nn.Linear(128, 512), nn.ReLU(), nn.Linear(512, 1))

    def forward(self, x, a=None):
        x = x + self.position
        if self.action is not None:
            x = x + self.action(a)[:, None]
        return self.net((self.value(x) * self.score(x).softmax(1)).sum(1))[:, 0]


def fit(x_train, a_train, y_train, x_dev, a_dev, device, action, seed):
    torch.manual_seed(seed)
    probe = Probe(x_train.shape[-1], action).to(device)
    opt = torch.optim.AdamW(probe.parameters(), lr=1e-3, weight_decay=1e-4)
    gen = torch.Generator().manual_seed(seed)
    pos = (1 - y_train).sum() / y_train.sum().clamp_min(1)
    for _ in range(STEPS):
        idx = torch.randint(len(x_train), (256,), generator=gen)
        logit = probe(x_train[idx].float().to(device), a_train[idx].to(device))
        loss = F.binary_cross_entropy_with_logits(logit, y_train[idx].to(device), pos_weight=pos.to(device))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    probe.eval()
    with torch.no_grad():
        score = torch.cat([probe(x_dev[i:i + 512].float().to(device), a_dev[i:i + 512].to(device)).cpu()
                           for i in range(0, len(x_dev), 512)])
    return score


def auc(y, score):
    """Mann-Whitney AUC with average ranks for ties."""
    order = score.argsort()
    ranks = torch.empty(len(score), dtype=torch.float64)
    ranks[order] = torch.arange(1, len(score) + 1, dtype=torch.float64)
    for value in score[order].unique_consecutive():
        tied = score == value
        ranks[tied] = ranks[tied].mean()
    pos = y.bool()
    n1, n0 = int(pos.sum()), int((~pos).sum())
    return float((ranks[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def main():
    started = time.time()
    log = lambda **kw: print(json.dumps({**kw, "seconds": round(time.time() - started, 1)}), flush=True)
    device = torch.device("cuda")
    os.chdir(ROOT)
    gen = torch.Generator().manual_seed(SEED)
    encoder, config = bridge()
    dev = dev_population(encoder, config, device, gen, log)
    del encoder
    torch.cuda.empty_cache()
    train = train_transitions(gen)
    log(stage="populations", train=len(train["dh"]), train_damage=int((train["dh"] <= -2).sum()),
        dev=len(dev["dh"]), dev_damage=int((dev["dh"] <= -2).sum()))
    y_train, y_dev = (train["dh"] <= -2).float(), (dev["dh"] <= -2).float()

    inputs = {"frame_tokens": (train["tokens"][:, 3], dev["tokens"][:, 3], True)}
    for arm in ("T", "TH"):
        stored = torch.load(WORLDS / f"{arm}.pt", map_location="cpu", weights_only=False)
        world = World(TOKENS, arm == "TH").to(device)
        world.load_state_dict(stored["world"])
        world.eval()
        h_train, g_train = world_features(world, config, train, device)
        h_dev, g_dev = world_features(world, config, dev, device)
        inputs[f"hidden_{arm}"] = (h_train, h_dev, False)
        if arm == "TH":
            inputs["generated_TH"] = (torch.cat([train["tokens"][:, 3], g_train], -1),
                                      torch.cat([dev["tokens"][:, 3], g_dev], -1), False)
        del world
        torch.cuda.empty_cache()
    a_train, a_dev = train["actions"][:, 3], dev["actions"][:, 3]
    result = {}
    for name, (xt, xd, action) in inputs.items():
        aucs = [auc(y_dev, fit(xt, a_train, y_train, xd, a_dev, device, action, s)) for s in range(3)]
        result[name] = {"dev_auc": float(np.mean(aucs)), "per_seed": aucs}
        log(input=name, dev_auc=round(result[name]["dev_auc"], 4))
    f, h, g = (result[k]["dev_auc"] for k in ("frame_tokens", "hidden_TH", "generated_TH"))
    reading = ("not_forecastable" if f < 0.75 else "not_written" if f >= 0.85 and h >= 0.85 and g < 0.65
               else "not_computed" if f >= 0.85 and h < 0.70 else "mixed")
    evidence = {"schema": "d4mj_spatial_why4_v1", "status": "POST HOC", "script_sha256": _sha256(Path(__file__)),
                "populations": {"train": len(y_train), "train_damage": int(y_train.sum()), "dev": len(y_dev),
                                "dev_damage": int(y_dev.sum())}, "reading": reading, "result": result}
    (HERE / "evidence/spatial_why4.json").write_text(json.dumps(evidence, indent=2) + "\n")
    log(status="spatial_why4_complete", reading=reading)


if __name__ == "__main__":
    main()
