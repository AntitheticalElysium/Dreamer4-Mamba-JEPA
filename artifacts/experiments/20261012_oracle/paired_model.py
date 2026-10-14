"""Direct, continuous source-associated event editing; observation-only training.

The shared event owns both endpoint writes. Contextual token vectors are not
treated as conserved physical mass. Soft routing does not guarantee validity.
"""
import copy

import torch
from torch import nn
from torch.nn import functional as F

import direct_model as D
import direct_train as T

ARMS = ('direct', 'paired')
draw_noise, losses, distance = D.draw_noise, D.losses, D.distance
INTERIOR, PATCHES = D.INTERIOR, D.PATCHES
LOCAL_NOISE, SHARED_NOISE = D.LOCAL_NOISE, D.SHARED_NOISE
NEIGHBOURS = tuple(D.L.TW.NEIGHBOURS)
OFFSETS = ((0, 0),) + NEIGHBOURS
DirectWorld = D.DirectWorld


def shift_grid(x, dr, dc):
    """Forward scatter: source[r,c] becomes destination[r+dr,c+dc]. No wrap."""
    assert x.shape[-2] == 81 and abs(dr) <= 1 and abs(dc) <= 1
    grid = x.reshape(*x.shape[:-2], 9, 9, x.shape[-1])
    padded = F.pad(grid, (0, 0, 1, 1, 1, 1))
    return padded[..., 1-dr:10-dr, 1-dc:10-dc, :].reshape(x.shape)


class PairedCompletion(D.Completion):
    def __init__(self, width=256, context_rank=16):
        super().__init__(width)
        del self.generate
        # Factorized recipient/context conditioning avoids a large edge tensor
        # concatenation. One decoder, with a learned endpoint role distinction.
        # Preserve the control's full possible 192-D stochastic output span.
        # Only recipient conditioning is factored for parameter economy; the
        # stochastic event is never compressed through the recipient adapter.
        self.event_read = nn.Identity()
        self.context_read = nn.Sequential(nn.Linear(width, context_rank, bias=False),
                                          nn.Linear(context_rank, width, bias=False))
        self.role = nn.Embedding(2, width)
        self.offset_read = nn.Linear(2, width, bias=False)
        self.write = nn.Linear(width, 192)
        nn.init.normal_(self.write.weight, std=.001)
        nn.init.zeros_(self.write.bias)
        mask = []
        for r in range(9):
            for c in range(9):
                mask.append([0 <= r+dr < 9 and 0 <= c+dc < 9 for dr, dc in OFFSETS])
        self.register_buffer('destination_mask', torch.tensor(mask, dtype=torch.bool), persistent=False)

    def forward(self, belief, noise):
        b, t, n, d = belief.shape
        x = (belief + self.noise(noise.to(belief.dtype))).reshape(b*t, n, d)
        q = self.norm(x)
        x = x + self.attention(q, q, q, need_weights=False)[0]
        event = (x + self.ff(x)).reshape(b, t, n, d)
        logits = self.route(event).float()
        gate = logits[..., :1].sigmoid()
        routes = logits[..., 1:].masked_fill(~self.destination_mask, -torch.inf).softmax(-1)
        return {'event': event, 'gate': gate, 'routes': routes}

    def endpoint(self, event_hidden, context_hidden, role, offset):
        displacement = event_hidden.new_tensor(offset)
        hidden = event_hidden + context_hidden + self.role.weight[role] + self.offset_read(displacement)
        return self.write(F.gelu(hidden)).float()

    def edits(self, belief, packets, *, removal_packets=None, arrival_packets=None, trace=False):
        """Packet override separates paths only for frozen inference diagnostics."""
        remove = packets if removal_packets is None else removal_packets
        arrive = packets if arrival_packets is None else arrival_packets
        destination = self.context_read(belief)
        departure_event = self.event_read(remove['event'])
        arrival_event = self.event_read(arrive['event'])
        source_edits = (-remove['gate'] * (1-remove['routes'][..., :1])
                        * self.endpoint(departure_event, destination, 0, (0, 0)))
        incoming = torch.zeros_like(source_edits)
        edge_writes = []
        for edge, (dr, dc) in enumerate(NEIGHBOURS, 1):
            recipient = shift_grid(destination, -dr, -dc)
            write = (arrive['gate'] * arrive['routes'][..., edge:edge+1]
                     * self.endpoint(arrival_event, recipient, 1, (dr, dc)))
            incoming = incoming + shift_grid(write, dr, dc)
            if trace:
                edge_writes.append(write)
        aux = {'source_edits': source_edits, 'incoming': incoming}
        if trace:
            # Writes indexed by SOURCE then outward edge; self has no write.
            aux['arrivals'] = torch.stack([torch.zeros_like(source_edits)] + edge_writes, -2)
        return source_edits + incoming, aux


class PairedWorld(D.DirectWorld):
    def __init__(self):
        super().__init__()
        self.completion = PairedCompletion()

    def event_packets(self, belief, noise):
        return self.completion(belief, noise)

    def reference(self, frames, actions, belief, action_belief):
        """Exact parent corrt reference before output LN; never assumed physical."""
        gen = self.proj(belief).float()
        grid = frames.float().reshape(*frames.shape[:2], 9, 9, 192)
        padded = F.pad(grid, (0, 0, 1, 1, 1, 1))
        candidates = [grid] + [padded[:, :, 1+dr:10+dr, 1+dc:10+dc] for dr, dc in NEIGHBOURS]
        candidates = torch.stack([x.reshape(frames.shape) for x in candidates] + [gen], -2)
        logits = self.choose(belief).float()
        moved = self.frame(action_belief).float()[:, :, None, :]
        target = torch.tensor([31, 30, 32, 22, 40], device=actions.device)[actions.clamp(max=4)]
        htarget = belief.gather(2, target[:, :, None, None].expand(-1, -1, 1, belief.shape[-1]))[:, :, 0]
        moved = moved + (((actions >= 1) & (actions <= 4)).float()[:, :, None, None]
                         * self.target_gate(htarget).float()[:, :, None, :])
        logits = logits + moved * logits.new_tensor([0, 1, 1, 1, 1, 0])
        return (logits.softmax(-1)[..., None] * candidates).sum(-2)

    def complete_from_packets(self, frames, actions, belief, action_belief, packets,
                              *, trace=False, removal_packets=None, arrival_packets=None):
        reference = self.reference(frames, actions, belief, action_belief)
        delta, aux = self.completion.edits(belief, packets, removal_packets=removal_packets,
                                           arrival_packets=arrival_packets, trace=trace)
        out = F.layer_norm(reference + delta, (192,))
        if trace:
            aux.update(reference=reference, **packets)
        else:
            aux = reference
        return out, packets['routes'], aux

    def complete(self, frames, actions, belief, action_belief, noise, *, trace=False):
        return self.complete_from_packets(frames, actions, belief, action_belief,
                                          self.event_packets(belief, noise), trace=trace)


def initialized_worlds(device):
    # Preserve the exact historical control initialization, including its trunk.
    direct = T.initialized_worlds('cpu')['energy']
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(7)
        paired = PairedWorld()
    shared = {k: v for k, v in direct.state_dict().items() if not k.startswith('completion.generate.')}
    result = paired.load_state_dict(shared, strict=False)
    expected = ('completion.event_read.', 'completion.context_read.', 'completion.role.',
                'completion.offset_read.', 'completion.write.')
    assert not result.unexpected_keys and all(k.startswith(expected) for k in result.missing_keys)
    assert all(torch.equal(paired.state_dict()[k], v) for k, v in shared.items())
    return {'direct': direct.to(device).train(), 'paired': paired.to(device).train()}


def parameter_counts(worlds):
    return {arm: {'total': sum(p.numel() for p in world.parameters()),
                  'completion': sum(p.numel() for p in world.completion.parameters())}
            for arm, world in worlds.items()}


def swapped_packet(packets, other, source):
    """Freeze every packet except one AFTER shared spatial attention."""
    result = {key: value.clone() for key, value in packets.items()}
    for key in result:
        result[key][..., source, :] = other[key][..., source, :]
    return result
