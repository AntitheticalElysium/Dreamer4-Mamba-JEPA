"""Stage 3 (2026-10-15): free stochastic content synthesis vs joint stochastic reassignment of retained content.

Both arms: the D parent's trainable backbone and deterministic reference b (PairedWorld.reference, pre-LN afterstate), the
existing noise trunk (direct_model.Completion: 8-D Gaussian noise embedding + spatial attention + FF), identical trunk
initialization, the same carrier/renderer (identity: the content of tile i is its afterstate token b_i; no learned
extractor or renderer, see NOTEBOOK Stage 2 F1a), the same energy objective. Only the stochastic head differs:
  free      zhat_j = LN(b_j + G(e_j))                 a per-tile synthesized token edit
  operator  zhat   = LN(T b)                          T an exact permutation made of disjoint adjacent swaps, sampled
                                                      jointly: each tile proposes stay / up / down / left / right by
                                                      Gumbel-max on logits P(e_j); a parallel greedy matching accepts a
                                                      proposal when it beats every other proposal touching either cell.
                                                      Straight-through Gumbel-softmax (tau 1) gradient through the
                                                      linear swap relaxation; the same hard operator in training and
                                                      evaluation. HUD rows and the player cell never move.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F

import direct_model as DM
import paired_model as PM

ARMS = ('free', 'operator')
COST_ARM = 'operator_cost'      # the identical operator; only its loss adds the transition cost (TRANSPORT_COST_PREREG.md)
DIRS = ((0, 0), (-1, 0), (1, 0), (0, -1), (0, 1))                       # stay + tworld.NEIGHBOURS order
BETA = math.log(4 * 62)                                                  # ~1 proposed move per frame at initialization
PLAYER = 31


def draw_gumbel(shape, generator):
    """Two independent draws [2,b,t,81,5]; drawn in both arms from the same generator so the Gaussian streams stay identical."""
    b, t = shape
    u = torch.rand(2, b, t, 81, 5, generator=generator).clamp(1e-10, 1 - 1e-10)
    return -torch.log(-torch.log(u))


def _moves():
    """valid[j, d]: tile j may propose direction d (target inside the 7x9 map, neither endpoint the player, j a map tile)"""
    valid = torch.zeros(81, 5, dtype=torch.bool)
    target = torch.zeros(81, 5, dtype=torch.long)
    for j in range(81):
        r, c = divmod(j, 9)
        for d, (dr, dc) in enumerate(DIRS):
            rr, cc = r + dr, c + dc
            target[j, d] = rr * 9 + cc if 0 <= rr < 9 and 0 <= cc < 9 else j
            if d == 0:
                valid[j, d] = True
            elif j < 63 and j != PLAYER and 0 <= rr < 7 and 0 <= cc < 9 and rr * 9 + cc != PLAYER:
                valid[j, d] = True
    return valid, target


class TransportCompletion(DM.Completion):
    def __init__(self, arm, width=256):
        super().__init__(width)
        assert arm in ARMS
        self.arm = arm
        del self.route
        if arm == 'operator':
            del self.generate
            self.propose = nn.Linear(width, 5)
            nn.init.normal_(self.propose.weight, std=.001)
            with torch.no_grad():
                self.propose.bias.copy_(torch.tensor([BETA, 0., 0., 0., 0.]))
        valid, target = _moves()
        self.register_buffer('valid', valid, persistent=False)
        self.register_buffer('target', target, persistent=False)

    def trunk(self, belief, noise):
        b, t, n, d = belief.shape
        x = (belief + self.noise(noise.to(belief.dtype))).reshape(b * t, n, d)
        q = self.norm(x)
        x = x + self.attention(q, q, q, need_weights=False)[0]
        return (x + self.ff(x)).reshape(b, t, n, d)

    def matching(self, score, choice):
        """parallel greedy: accept a proposal j -> k when its score is the maximum over all live proposals touching j or k"""
        flat_s, flat_c = score.reshape(-1, 81), choice.reshape(-1, 81)
        m = len(flat_s)
        src = torch.arange(81, device=score.device).expand(m, -1)
        dst = self.target[src, flat_c]
        live = flat_c > 0
        partner = src.clone()
        free = torch.ones(m, 81, dtype=torch.bool, device=score.device)
        neg = torch.finfo(flat_s.dtype).min
        for _ in range(16):
            live = live & free.gather(1, src) & free.gather(1, dst)
            if not bool(live.any()):
                break
            s = torch.where(live, flat_s, torch.full_like(flat_s, neg))
            best = torch.full((m, 81), neg, device=score.device, dtype=flat_s.dtype)
            best = best.scatter_reduce(1, src, s, 'amax').scatter_reduce(1, dst, s, 'amax')
            win = live & (s >= best.gather(1, src)) & (s >= best.gather(1, dst))
            rows, js = torch.where(win)
            ks = dst[rows, js]
            partner[rows, js], partner[rows, ks] = ks, js
            free[rows, js] = False; free[rows, ks] = False
        return partner.reshape(score.shape)

    def operate(self, reference, hidden, gumbel, tau=1.0):
        logits = self.propose(hidden).float().masked_fill(~self.valid, -torch.inf)
        perturbed = logits + gumbel
        choice = perturbed.argmax(-1)
        score = perturbed.gather(-1, choice[..., None])[..., 0] - perturbed[..., 0]
        partner = self.matching(score.detach(), choice)
        # forward: the exact permutation; the reference's gradient follows the content actually moved
        hard = reference.gather(2, partner[..., None].expand(-1, -1, -1, reference.shape[-1]))
        # straight-through for the proposals only: the linear swap relaxation, on detached content
        q = (perturbed / tau).softmax(-1)                                # [b,t,81,5]; q[...,0] stay
        content = reference.detach()
        relax = torch.zeros_like(content)
        for d in range(1, 5):
            k = self.target[:, d]                                       # [81]
            other = content[:, :, k]                                    # content of j's d-neighbour
            w = q[..., d:d + 1] * self.valid[:, d].float()[:, None]
            relax = relax + w * (other - content)                       # j receives k's content
            relax = relax.index_add(2, k, w * (content - other))        # k receives j's content
        return hard + relax - relax.detach(), {'partner': partner, 'logits': logits, 'choice': choice}

    def forward(self, reference, belief, noise, gumbel):
        hidden = self.trunk(belief, noise)
        if self.arm == 'free':
            return F.layer_norm(reference + self.generate(hidden).float(), (192,)), {}
        mixed, aux = self.operate(reference, hidden, gumbel)
        return F.layer_norm(mixed, (192,)), aux


def expected_changes(logits):
    """transition cost per frame: the expected number of tiles proposing to change, sum_j (1 - P_j(stay)) [..., 81, 5] -> [...]"""
    return (1 - logits.softmax(-1)[..., 0]).sum(-1)


class TransportWorld(PM.PairedWorld):
    def __init__(self, arm, init_seed=7):
        super().__init__()
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(init_seed)
            self.completion = TransportCompletion('free' if arm == 'free' else 'operator')

    def complete(self, frames, actions, belief, action_belief, noise, gumbel):
        reference = self.reference(frames, actions, belief, action_belief).float()
        return self.completion(reference, belief, noise, gumbel)


def initialized_worlds(parent_state, device, arms=ARMS, init_seed=7):
    worlds = {}
    for arm in arms:
        w = TransportWorld(arm, init_seed)
        result = w.load_state_dict(parent_state, strict=False)
        assert not result.unexpected_keys or all(k.startswith('completion.') for k in result.unexpected_keys), result.unexpected_keys
        assert all(k.startswith('completion.') for k in result.missing_keys), result.missing_keys
        worlds[arm] = w.to(device).train()
    first = worlds[arms[0]].state_dict()
    for arm in arms[1:]:
        other = worlds[arm].state_dict()
        shared = [k for k in first if k in other]
        assert all(torch.equal(first[k], other[k]) for k in shared), arm
    return worlds, shared
