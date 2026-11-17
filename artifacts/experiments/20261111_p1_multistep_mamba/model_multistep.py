"""Multi-Step Mamba belief model: extends M0 with a learned mask_token for blind rollouts."""
import torch
from torch import nn, Tensor
from saccade import model as M


class MultiStepBelief(M.Belief):
    def __init__(self, cfg):
        super().__init__(cfg)
        W = cfg["belief"]["width"]
        self.mask_token = nn.Parameter(torch.zeros(W))
        with torch.no_grad():
            self.mask_token.normal_(std=0.02)

    def rollout_tokens(self, actions: Tensor) -> Tensor:
        """Actions [B, H] -> student tokens [B, 2*H, W] interleaving action tokens and mask_token.
        Phase: token 0 is a_t, token 1 is pt_mask, token 2 is a_{t+1}, ..., token 2H-1 is pt_mask."""
        B, H = actions.shape
        W = self.mask_token.shape[0]
        act_embeds = self.action(actions)                               # [B, H, W]
        mask_embeds = self.mask_token[None, None, :].expand(B, H, W).to(act_embeds.dtype)
        return torch.stack([act_embeds, mask_embeds], dim=2).flatten(1, 2)  # [B, 2*H, W]


class MultiStepM0(M.M0):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.belief = MultiStepBelief(cfg)
