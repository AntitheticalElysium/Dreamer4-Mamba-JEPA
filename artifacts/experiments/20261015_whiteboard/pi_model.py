"""Persistent state + inferred transformations (PI): the Track B world (2026-10-15 evening). TRANSPORT_PI_PREREG.md governs.

Same frozen Raw encoder, same causal backbone and chance operator as TRANSPORT_IC v3; only the deterministic afterstate
changes. CorrT's soft mixture of copy candidates and a generated token is replaced by explicit decisions:
  scroll   one hard global choice in {stay, up, down, left, right} for the 63 map cells (the HUD never scrolls), from CorrT's
           own global action term (frame + target_gate); trained by cross-entropy toward the parameter-free scroll estimate
           of the real (t, t+1) frames (label-free)
  copy     the previous frame's token at the chosen source, EXACTLY (no blend); map cells whose source lies outside the map
           ('entering') must be generated
  edit     a hard per-cell gate (new head on the belief) chooses the generated token proj(belief) instead of the copy;
           trained toward the Bayesian posterior q = sigmoid(gain / s + logit p_gate), gain = |copy - z| - |gen - z| on the
           pre-chance target, s = the frame's mean copy residual (an edit must pay its description cost)
  chance   the IC v3 operator and label-free posterior (transport_ic_train.infer with the chance head's own prior)
Recursion uses the model's own hard decisions: scroll argmax, edit = gate > 0, chance sampled by the proposals + matching.
"""
import torch
from torch import nn
from torch.nn import functional as F

import paired_model as PM
import transport_model as TM

MAP = 63
SCROLL_OFFSETS = ((0, 0),) + tuple(PM.NEIGHBOURS)                         # candidate order of CorrT: self, 4 neighbours


def entering_mask():
    """[5,81] True where, for that scroll choice, a map cell's source lies outside the 7x9 map (must be generated)"""
    m = torch.zeros(5, 81, dtype=torch.bool)
    for s, (dr, dc) in enumerate(SCROLL_OFFSETS):
        for x in range(MAP):
            r, c = divmod(x, 9)
            m[s, x] = not (0 <= r + dr < 7 and 0 <= c + dc < 9)
    return m


class PIWorld(TM.TransportWorld):
    def __init__(self, arm='pi', init_seed=7):
        super().__init__('operator', init_seed)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(init_seed + 1000)
            self.gate = nn.Linear(256, 1)
        nn.init.normal_(self.gate.weight, std=.001)
        nn.init.constant_(self.gate.bias, -4.0)                            # prior P(edit) ~ 2%
        self.register_buffer('entering', entering_mask(), persistent=False)

    def scroll_logits(self, actions, belief, action_belief):
        """[B,T,5] over (stay, up, down, left, right): CorrT's single move logit (frame + target_gate) goes to the action's
        own direction (scroll.MOVE_SHIFT: left -> 3, right -> 4, up -> 1, down -> 2); other directions are impossible"""
        moved = self.frame(action_belief).float()[..., 0]
        target = torch.tensor([31, 30, 32, 22, 40], device=actions.device)[actions.clamp(max=4)]
        htarget = belief.gather(2, target[:, :, None, None].expand(-1, -1, 1, belief.shape[-1]))[:, :, 0]
        is_move = (actions >= 1) & (actions <= 4)
        moved = moved + is_move.float() * self.target_gate(htarget).float()[..., 0]
        direction = torch.tensor([0, 3, 4, 1, 2], device=actions.device)[actions.clamp(max=4)]           # MOVE_SHIFT
        out = torch.full((*actions.shape, 5), -20.0, device=actions.device)
        out = out.scatter(-1, direction[..., None], torch.where(is_move, moved, torch.full_like(moved, -20.0))[..., None])
        out[..., 0] = 0.0                                                   # stay is always possible (non-move actions map to 0)
        return out

    def candidates(self, frames):
        grid = frames.float().reshape(*frames.shape[:2], 9, 9, 192)
        padded = F.pad(grid, (0, 0, 1, 1, 1, 1))
        return torch.stack([grid.reshape(frames.shape)] + [padded[:, :, 1 + dr:10 + dr, 1 + dc:10 + dc].reshape(frames.shape)
                                                           for dr, dc in PM.NEIGHBOURS], -2)                  # [B,T,81,5,192]

    def afterstate_parts(self, frames, actions, belief, action_belief, scroll=None):
        """copy (exact, given a hard scroll), gen, gate logits, entering mask; scroll [B,T] long or None (own argmax)"""
        sl = self.scroll_logits(actions, belief, action_belief)
        choice = sl.argmax(-1) if scroll is None else scroll
        cand = self.candidates(frames)
        idx = torch.zeros(*choice.shape, 81, dtype=torch.long, device=frames.device)
        idx[..., :MAP] = choice[..., None]                                 # map cells follow the scroll; HUD copies itself
        copy = cand.gather(-2, idx[..., None, None].expand(-1, -1, -1, 1, 192))[..., 0, :]
        enter = self.entering[choice]                                     # [B,T,81]
        gen = self.proj(belief).float()
        glogit = self.gate(belief).float()[..., 0]
        return copy, gen, glogit, enter, sl

    def complete(self, frames, actions, belief, action_belief, noise, gumbel):
        """recursion / sampling with the model's own decisions"""
        copy, gen, glogit, enter, _ = self.afterstate_parts(frames, actions, belief, action_belief)
        edit = (glogit > 0) | enter
        after = torch.where(edit[..., None], gen, copy)
        mixed, aux = self.completion.operate(after, self.completion.trunk(belief, noise), gumbel)
        aux['edits'] = edit
        return F.layer_norm(mixed, (192,)), aux

    def reference(self, frames, actions, belief, action_belief):
        """the deterministic afterstate with the model's own decisions (no chance), for evaluation code that calls reference()"""
        copy, gen, glogit, enter, _ = self.afterstate_parts(frames, actions, belief, action_belief)
        return torch.where(((glogit > 0) | enter)[..., None], gen, copy)
