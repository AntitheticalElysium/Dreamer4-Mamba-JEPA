"""Direct continuous stochastic completion after one causal Mamba computation.

The future is only a loss target. No inverse encoder, codebook, chance prior,
simulator labels, object masks or adaptive computation are used.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F

import loss_intervention as L

INTERIOR = tuple(r * 9 + c for r in range(1, 6) for c in range(1, 8))
PATCHES = tuple(tuple(r * 9 + c for r in range(rr, rr + 3) for c in range(cc, cc + 3))
                for rr in (1, 2, 3) for cc in (1, 3, 5))
LOCAL_NOISE, SHARED_NOISE = 4, 4
ARMS = ('point', 'energy')


def draw_noise(shape, generator):
    """CPU generator makes both independently drawn samples identical across arms."""
    b, t = shape
    local = torch.randn(2, b, t, 81, LOCAL_NOISE, generator=generator)
    shared = torch.randn(2, b, t, 1, SHARED_NOISE, generator=generator).expand(-1, -1, -1, 81, -1)
    return torch.cat((local, shared), -1)


class Completion(nn.Module):
    def __init__(self, width=256):
        super().__init__()
        self.noise = nn.Linear(LOCAL_NOISE + SHARED_NOISE, width)
        self.norm = nn.LayerNorm(width)
        self.attention = nn.MultiheadAttention(width, 4, batch_first=True, dropout=0)
        self.ff = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, width * 2), nn.GELU(),
                                nn.Linear(width * 2, width))
        self.generate = nn.Linear(width, 192)
        self.route = nn.Linear(width, 6)
        # A zero output matrix killed the previous posterior's completion signal.
        # Keep a small live path from both independent noises at initialization.
        nn.init.normal_(self.generate.weight, std=.001)
        nn.init.normal_(self.route.weight, std=.001)
        nn.init.zeros_(self.generate.bias)
        nn.init.zeros_(self.route.bias)

    def forward(self, belief, noise):
        b, t, n, d = belief.shape
        x = (belief + self.noise(noise.to(belief.dtype))).reshape(b * t, n, d)
        q = self.norm(x)
        x = x + self.attention(q, q, q, need_weights=False)[0]
        x = x + self.ff(x)
        return self.generate(x).reshape(b, t, n, 192).float(), self.route(x).reshape(b, t, n, 6).float()


class DirectWorld(L.TW.TWorld):
    def __init__(self):
        super().__init__('corrt', None, 'fmamba', 'all', 0, False, 16)
        self.completion = Completion()

    def complete(self, frames, actions, belief, action_belief, noise, *, routing_noise=None):
        delta, route = self.completion(belief, noise)
        if routing_noise is not None:
            _, route = self.completion(belief, routing_noise)
        gen = self.proj(belief).float() + delta
        grid = frames.float().view(*frames.shape[:2], 9, 9, 192)
        padded = F.pad(grid, (0, 0, 1, 1, 1, 1))
        candidates = [grid] + [padded[:, :, 1 + dr:10 + dr, 1 + dc:10 + dc]
                               for dr, dc in L.TW.NEIGHBOURS]
        candidates = torch.stack([x.reshape(frames.shape) for x in candidates] + [gen], -2)
        logits = self.choose(belief).float() + route
        moved = self.frame(action_belief).float()[:, :, None, :]
        target = torch.tensor([31, 30, 32, 22, 40], device=actions.device)[actions.clamp(max=4)]
        htarget = belief.gather(2, target[:, :, None, None].expand(-1, -1, 1, belief.shape[-1]))[:, :, 0]
        moved = moved + (((actions >= 1) & (actions <= 4)).float()[:, :, None, None]
                         * self.target_gate(htarget).float()[:, :, None, :])
        logits = logits + moved * logits.new_tensor([0, 1, 1, 1, 1, 0])
        weights = logits.softmax(-1)
        out = F.layer_norm((weights[..., None] * candidates).sum(-2), (192,))
        return out, weights, gen

    def samples(self, frames, actions, noise):
        # Do not call innovation_model.native_belief: it detaches the backbone.
        belief, action_belief = self.backbone_full(frames, actions)
        completed = [self.complete(frames, actions, belief, action_belief, n) for n in noise]
        return tuple(torch.stack([x[j] for x in completed]) for j in range(3))


def distance(x, y):
    """Fixed first-power metric: global interior + overlapping local interiors.

Strictly proper energy distance on the scored 35-token domain; no squared
distance, epsilon smoothing, event masks, realized-future weights or temperature.
"""
    err = (x.float() - y.float())
    whole = torch.linalg.vector_norm(err[..., INTERIOR, :].flatten(-2), dim=-1) / math.sqrt(35 * 192)
    patches = torch.stack([torch.linalg.vector_norm(err[..., p, :].flatten(-2), dim=-1)
                           / math.sqrt(9 * 192) for p in PATCHES], -1).mean(-1)
    return .5 * whole + .5 * patches


def losses(predictions, target):
    assert predictions.shape[0] == 2 and predictions.shape[1:] == target.shape
    attraction = .5 * distance(predictions[0], target).mean() + .5 * distance(predictions[1], target).mean()
    diversity = .5 * distance(predictions[0], predictions[1]).mean()
    return {'point': attraction, 'energy': attraction - diversity,
            'attraction': attraction, 'diversity': diversity}
