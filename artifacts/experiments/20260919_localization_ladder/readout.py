"""Phase 1/2 readout instrument: is the representation readable for the within-root decision?

The gate fits one decoder on six binary outcomes with a BCE objective and then asks a question
it never trained for -- "which of these 17 actions kills me" -- by taking an argmin. Those are
different objectives, and the plan's first job is to separate decision-objective mismatch from
representation content. So every rung is scored under three supervisions on identical rows:

  bce6        the published protocol's six-target BCE -- the original-protocol ANCHOR
  bce_death   ordinary single-target death BCE
  rank        within-root pair ranking, softplus(score_safe - score_fatal), averaged over the
              pairs inside a root and then over roots, so a root with many fatal forks cannot
              dominate the gradient

and four capacity families, because a linear/MLP-128 tie settles nothing, and because equal
hidden width is not equal parameter capacity -- parameter counts are reported per fit:

  logistic    linear scorer with L2
  mlp128      the published width
  mlp512x2    two hidden layers of 512
  token       a small position-aware token readout, for spatial rungs only

Primary metric is safe-choice rate on roots that actually offer both a fatal and a safe action.
Within-root AUC, the selected-action histogram and argmin ties are reported beside it, because
a rate alone hides an action prior -- the published Raw head picks SLEEP on 24/36 roots where
SLEEP is safe on 6.

Nothing here trains an encoder or a world model.
"""

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

DEATH = 0        # index of `death` within OUTCOME_BINARY
N_ACTIONS = 17


@dataclass(frozen=True)
class Fit:
    steps: int = 2000
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    batch: int = 128
    report_every: int = 100


class TokenHead(nn.Module):
    """Position-aware readout for [B, tokens, width] rungs.

    A flattened grid through an MLP discards which token sat where only if the first layer
    cannot recover it; a learned position embedding plus attention pooling keeps position
    explicit while staying far smaller than a flattened 15552-wide first layer.
    """

    def __init__(self, tokens, width, out_dim, hidden=128):
        super().__init__()
        self.position = nn.Parameter(torch.zeros(tokens, width))
        self.value = nn.Linear(width, hidden)
        self.score = nn.Linear(width, 1)
        self.head = nn.Sequential(nn.GELU(), nn.Linear(hidden, out_dim))

    def forward(self, x):
        lead, (tokens, width) = x.shape[:-2], x.shape[-2:]
        flat = x.reshape(-1, tokens, width) + self.position
        weight = self.score(flat).softmax(1)
        pooled = (self.value(flat) * weight).sum(1)
        return self.head(pooled).reshape(*lead, -1)


def build_head(family, shape, out_dim, generator, device):
    """`shape` is the trailing feature shape: (D,) for vectors, (tokens, width) for grids."""
    if family == "token":
        if len(shape) != 2:
            raise ValueError("the token readout needs an unflattened [tokens, width] rung")
        model = TokenHead(shape[0], shape[1], out_dim)
    else:
        in_dim = int(np.prod(shape))
        if family == "logistic":
            model = nn.Linear(in_dim, out_dim)
        elif family == "mlp128":
            model = nn.Sequential(nn.Linear(in_dim, 128), nn.GELU(), nn.Linear(128, out_dim))
        elif family == "mlp512x2":
            model = nn.Sequential(nn.Linear(in_dim, 512), nn.GELU(),
                                  nn.Linear(512, 512), nn.GELU(), nn.Linear(512, out_dim))
        else:
            raise ValueError(f"unknown readout family {family!r}")
    for module in model.modules():
        if isinstance(module, nn.Linear):
            bound = (1.0 / module.in_features) ** 0.5
            with torch.no_grad():
                module.weight.uniform_(-bound, bound, generator=generator)
                module.bias.uniform_(-bound, bound, generator=generator)
    return model.to(device)


def rank_loss(scores, labels):
    """Vectorised softplus(safe - fatal): mean over a root's pairs, then over usable roots."""
    fatal = labels.bool()
    safe = ~fatal
    pair = nn.functional.softplus(scores[:, None, :] - scores[:, :, None])   # [B, fatal, safe]
    mask = (fatal[:, :, None] & safe[:, None, :]).float()
    count = mask.sum((1, 2))
    usable = count > 0
    if not bool(usable.any()):
        return scores.sum() * 0.0
    per_root = (pair * mask).sum((1, 2))[usable] / count[usable]
    return per_root.mean()


def _flat(x):
    return x.reshape(*x.shape[:2], -1) if x.dim() > 3 else x


def fit_head(train_x, train_y, *, family, objective, seed, spec, device, inner=None,
             permute_labels=False):
    """Fit one head; return it, its curve, and its parameter count.

    Batching is over ROOTS so the ranking objective always sees a complete 17-action fan and
    BCE sees exactly the same rows rather than a differently shaped sample.  `permute_labels`
    shuffles death labels *within* each root, destroying the action-outcome correspondence
    while preserving every marginal -- the memorization control the plan requires for the
    larger heads.
    """
    generator = torch.Generator(device="cpu").manual_seed(seed)
    shape = train_x.shape[2:]
    out_dim = train_y.shape[-1]
    model = build_head(family, shape, out_dim, generator, device)
    parameters = sum(p.numel() for p in model.parameters())
    optimizer = torch.optim.AdamW(model.parameters(), lr=spec.learning_rate,
                                  weight_decay=spec.weight_decay)
    y = train_y.clone()
    if permute_labels:
        for row in range(len(y)):
            y[row, :, DEATH] = y[row, torch.randperm(N_ACTIONS, generator=generator), DEATH]
    n = len(train_x)
    curve = []
    for step in range(spec.steps):
        index = torch.randint(n, (min(spec.batch, n),), generator=generator)
        xb, yb = train_x[index].to(device), y[index].to(device)
        scores = model(xb)
        if objective == "bce6":
            loss = nn.functional.binary_cross_entropy_with_logits(scores, yb)
        elif objective == "bce_death":
            loss = nn.functional.binary_cross_entropy_with_logits(scores[..., DEATH], yb[..., DEATH])
        elif objective == "rank":
            loss = rank_loss(scores[..., DEATH], yb[..., DEATH])
        else:
            raise ValueError(f"unknown objective {objective!r}")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if (step + 1) % spec.report_every == 0:
            entry = {"step": step + 1, "loss": float(loss.detach())}
            if inner is not None:
                entry["inner_safe_rate"] = evaluate(model, inner[0], inner[1], device)["safe_choice_rate"]
            curve.append(entry)
    return model, curve, parameters


@torch.inference_mode()
def scores_of(model, x, device, batch=128):
    return torch.cat([model(x[i:i + batch].to(device))[..., DEATH].cpu()
                      for i in range(0, len(x), batch)])


def summarize(scores, y):
    """Safe-choice rate, within-root AUC, argmin ties and the selected-action histogram."""
    labels = y[..., DEATH].bool()
    fatal, safe = labels, ~labels
    usable = fatal.any(1) & safe.any(1)
    rows = torch.where(usable)[0]
    if len(rows) == 0:
        return {"opportunity_roots": 0}
    chosen = scores[rows].argmin(1)
    correct = safe[rows, chosen]
    aucs = []
    for row in rows:
        f, s = scores[row][fatal[row]], scores[row][safe[row]]
        aucs.append(float((s[None, :] < f[:, None]).float().mean()
                          + 0.5 * (s[None, :] == f[:, None]).float().mean()))
    ties = int(sum(int((scores[row] == scores[row].min()).sum() > 1) for row in rows))
    return {"safe_choice": int(correct.sum()), "opportunity_roots": int(usable.sum()),
            "safe_choice_rate": float(correct.float().mean()),
            "within_root_auc": float(np.mean(aucs)),
            "argmin_ties": ties,
            "selected_actions": np.bincount(chosen.numpy(), minlength=N_ACTIONS).tolist()}


def evaluate(model, x, y, device):
    return summarize(scores_of(model, x, device), y)


def standardize(train, *others):
    """TRAIN-only affine, as the gate does; a leaked DEV mean would flatter every rung."""
    flat = train.reshape(-1, train.shape[-1])
    mean, scale = flat.mean(0), flat.std(0, unbiased=False).clamp_min(1e-6)
    return [(t - mean) / scale for t in (train, *others)]


def one_hot_actions(n):
    return torch.eye(N_ACTIONS).expand(n, N_ACTIONS, N_ACTIONS).clone()


def opportunity_counts(y, names):
    """Informative roots per target -- death, damage and reward do NOT share a count."""
    out = {}
    for index, name in enumerate(names):
        labels = y[..., index].bool()
        both = labels.any(1) & (~labels).any(1)
        out[name] = {"opportunity_roots": int(both.sum()), "roots": int(len(labels))}
    return out
