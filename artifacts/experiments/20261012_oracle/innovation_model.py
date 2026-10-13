"""Observation-only categorical innovations; no simulator inputs or pixel decoder.

Six learned factors are not assumed to be objects. Each has eight hard choices.
The prior models their joint distribution autoregressively. Only hard choices
cross the future-conditioned posterior/continuous-completion boundary.
"""
from contextlib import contextmanager

import torch
from torch import nn
import torch.nn.functional as F

SLOTS, CHOICES, WIDTH, TOKEN_WIDTH = 6, 8, 256, 192


class _HardCode(torch.autograd.Function):
    @staticmethod
    def forward(ctx, probability):
        return F.one_hot(probability.argmax(-1), CHOICES).to(probability.dtype)

    @staticmethod
    def backward(ctx, gradient):
        return gradient


class FactorEncoder(nn.Module):
    def __init__(self, input_width):
        super().__init__()
        self.project = nn.Sequential(nn.LayerNorm(input_width), nn.Linear(input_width, WIDTH), nn.GELU())
        self.queries = nn.Parameter(torch.randn(SLOTS, WIDTH)*.02)
        self.attention = nn.MultiheadAttention(WIDTH, 4, batch_first=True)
        self.norm = nn.LayerNorm(WIDTH)

    def forward(self, features):
        shape = features.shape[:-2]
        x = self.project(features.flatten(0, -3))
        query = self.queries[None].expand(len(x), -1, -1)
        out = self.norm(query + self.attention(query, x, x, need_weights=False)[0])
        return out.reshape(*shape, SLOTS, WIDTH)


class Posterior(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = FactorEncoder(WIDTH+2*TOKEN_WIDTH)
        self.logits = nn.Linear(WIDTH, CHOICES)

    def forward(self, belief, current, successor):
        return self.logits(self.encoder(torch.cat([belief, current, successor-current], -1)))


class JointPrior(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = FactorEncoder(WIDTH)
        self.embedding = nn.Embedding(CHOICES, WIDTH)
        self.cell = nn.GRUCell(2*WIDTH, WIDTH)
        self.logits = nn.Linear(WIDTH, CHOICES)

    def forward(self, belief, codes=None, *, sample=False, generator=None):
        features = self.encoder(belief)
        shape = features.shape[:-2]
        x = features.reshape(-1, SLOTS, WIDTH)
        state = x.mean(1)
        previous = torch.zeros_like(state)
        targets = codes.reshape(-1, SLOTS) if codes is not None else None
        logits, chosen = [], []
        for slot in range(SLOTS):
            state = self.cell(torch.cat([x[:,slot], previous], -1), state)
            row = self.logits(state).float()
            if targets is not None:
                index = targets[:,slot]
            elif sample:
                index = torch.multinomial(row.softmax(-1), 1, generator=generator).squeeze(-1)
            else:
                index = row.argmax(-1)
            logits.append(row)
            chosen.append(index)
            previous = self.embedding(index)
        return torch.stack(logits,1).reshape(*shape,SLOTS,CHOICES), torch.stack(chosen,1).reshape(*shape,SLOTS)


class Innovation(nn.Module):
    def __init__(self):
        super().__init__()
        self.posterior = Posterior()
        self.prior = JointPrior()
        self.values = nn.Parameter(torch.randn(SLOTS, CHOICES, WIDTH)*.02)
        self.positions = nn.Parameter(torch.randn(SLOTS,WIDTH)*.02)
        self.attention = nn.MultiheadAttention(WIDTH,4,batch_first=True)
        self.output = nn.Linear(WIDTH, WIDTH)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    @staticmethod
    def hard(logits):
        probability = logits.float().softmax(-1)
        index = probability.argmax(-1)
        # An exact hard forward avoids a floating-point continuous bypass.
        return _HardCode.apply(probability), index

    def condition(self, belief, code_weights=None):
        if code_weights is None:
            return torch.zeros_like(belief)
        shape = belief.shape[:-2]
        values = torch.einsum('...sk,skd->...sd',code_weights.float(),self.values) + self.positions
        query = belief.reshape(-1,81,WIDTH)
        values = values.reshape(-1,SLOTS,WIDTH)
        delta = self.output(self.attention(query,values,values,need_weights=False)[0])
        return delta.reshape(*shape,81,WIDTH)

    def completion_parameters(self):
        prior_ids = {id(p) for p in self.prior.parameters()}
        return [p for p in self.parameters() if id(p) not in prior_ids]


@contextmanager
def conditioned(world, delta):
    assert getattr(world,'delta',None) is None
    world.delta = delta
    try:
        yield
    finally:
        world.delta = None


def frame_rate(logits, hard_weights):
    # Deterministic posterior assignments: forward value is joint code NLL.
    # ST gradient provides a rate-distortion surrogate, not a soft-posterior ELBO.
    return -(hard_weights*logits.float().log_softmax(-1)).sum((-1,-2)).mean()


def native_belief(world, frames, actions, config):
    from d4mj.train import autocast_context
    assert getattr(world,'delta',None) is None
    with torch.no_grad(),autocast_context(config):
        tiles,_ = world.backbone_full(frames.float(),actions)
    return tiles.detach().float()
