"""Mamba-JEPA dynamics model: predicts the next (frozen V-JEPA) latent from the
current latent + action. Multi-head for cheap ensemble disagreement (u_s), and the
vendored Mamba-2 keeps Delta-capture for the due-diligence probe.

Design (settled): reconstruction-free JEPA regression in a frozen-encoder latent, so
there is NO representation collapse (fixed target). Low-Lipschitz / temporal-
straightening regularizer follows "On Training in Imagination".
"""
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

DRAMA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "third_party", "Drama")
if DRAMA not in sys.path:
    sys.path.insert(0, DRAMA)
from mamba_ssm.modules.mamba2 import Mamba2
from sub_models.functions_losses import SymLogTwoHotLoss  # reuse DRAMA's symlog-two-hot


def _mlp_head(d, out, hidden=256):
    return nn.Sequential(nn.Linear(d, hidden), nn.RMSNorm(hidden), nn.SiLU(), nn.Linear(hidden, out))


class RPFHead(nn.Module):
    """Randomized-prior-function head (Osband et al. 2018): trainable MLP + a FROZEN
    random MLP prior. In-distribution the trainable part cancels the prior (heads agree
    -> low u_s); OOD it cannot (heads diverge via diverse frozen priors -> high u_s).
    This gives persistent epistemic disagreement that naive shared-trunk heads lose."""
    def __init__(self, d, out, hidden=256, prior_scale=1.0):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, hidden), nn.GELU(), nn.Linear(hidden, out))
        self.prior = nn.Sequential(nn.Linear(d, hidden), nn.GELU(), nn.Linear(hidden, out))
        for p in self.prior.parameters():
            p.requires_grad_(False)
        self.prior_scale = prior_scale

    def forward(self, h):
        return self.net(h) + self.prior_scale * self.prior(h)


class JEPADynamics(nn.Module):
    def __init__(self, latent_dim=768, action_dim=6, d_model=512, n_layers=2,
                 n_heads=5, d_state=64, headdim=64, prior_scale=1.0, spectral=False):
        super().__init__()
        self.latent_dim = latent_dim
        self.n_heads = n_heads
        self.in_proj = nn.Linear(latent_dim, d_model)
        self.act_emb = nn.Embedding(action_dim, d_model)
        self.blocks = nn.ModuleList([
            Mamba2(d_model=d_model, d_state=d_state, headdim=headdim, ngroups=1,
                   use_mem_eff_path=False)
            for _ in range(n_layers)
        ])
        self.norms = nn.ModuleList([nn.RMSNorm(d_model) for _ in range(n_layers)])
        self.head_norm = nn.RMSNorm(d_model)
        # randomized-prior heads -> disagreement (u_s) that survives OOD (see RPFHead).
        self.heads = nn.ModuleList([RPFHead(d_model, latent_dim, prior_scale=prior_scale)
                                    for _ in range(n_heads)])
        # reward/termination predicted from the Mamba hidden h (as in DRAMA).
        self.reward_head = _mlp_head(d_model, 255)     # symlog-two-hot logits
        self.term_head = _mlp_head(d_model, 1)         # BCE logit
        self.symlog = SymLogTwoHotLoss(255, -20, 20)
        self.d_model = d_model
        self.mixers = [b for b in self.blocks]  # for Delta capture
        if spectral:  # Miyato 2018 spectral norm -> bounds the dynamics Lipschitz L_f (On-Training-in-Imagination)
            from torch.nn.utils.parametrizations import spectral_norm as _sn
            self.in_proj = _sn(self.in_proj)
            for hd in self.heads:                                  # only the trainable net (not the frozen prior)
                hd.net[0] = _sn(hd.net[0]); hd.net[2] = _sn(hd.net[2])

    def set_delta_capture(self, on=True):
        for m in self.mixers:
            m._capture_delta = on

    def read_delta(self):
        ds = [m._last_delta.float() for m in self.mixers if getattr(m, "_last_delta", None) is not None]
        if not ds:
            return None
        alld = torch.cat([d.reshape(d.shape[0], -1) for d in ds], dim=-1)  # (B, sum_heads)
        return alld

    def backbone(self, z, a):
        h = self.in_proj(z) + self.act_emb(a)
        for blk, norm in zip(self.blocks, self.norms):
            h = h + blk(norm(h))
        return self.head_norm(h)

    def forward(self, z, a):
        """z (B,T,D), a (B,T) int -> preds (K,B,T,D) next-latent per head, and hidden h (B,T,d_model)."""
        h = self.backbone(z, a)
        preds = torch.stack([head(h) for head in self.heads], dim=0)
        return preds, h

    def reward_term(self, h):
        """h (.,d_model) -> (reward_logits[.,255], term_logit[.])."""
        return self.reward_head(h), self.term_head(h).squeeze(-1)

    # ---- autoregressive path (imagination rollout + Delta capture) ----
    def init_state(self, B, max_len, device, dtype=torch.float32):
        return [list(blk.allocate_inference_cache(B, max_len, dtype=dtype)) for blk in self.blocks]

    def step(self, z_t, a_t, states):
        """One autoregressive step. z_t (B,1,D), a_t (B,) -> preds (K,B,1,D). Mutates states.
        Routes through Mamba2.step() (where Delta is captured)."""
        h = self.in_proj(z_t) + self.act_emb(a_t).unsqueeze(1)   # (B,1,d_model)
        for i, (blk, norm) in enumerate(zip(self.blocks, self.norms)):
            out, states[i][0], states[i][1] = blk.step(norm(h), states[i][0], states[i][1])
            h = h + out
        h = self.head_norm(h)
        preds = torch.stack([head(h) for head in self.heads], dim=0)  # (K,B,1,D)
        return preds, h, states


def curvature_loss(zhat):
    """Temporal-straightening: penalize direction change of latent velocity (low-Lipschitz
    velocity map). zhat (B,T,D). Returns scalar in [0, 2]."""
    v = zhat[:, 1:] - zhat[:, :-1]            # (B,T-1,D)
    if v.shape[1] < 2:
        return zhat.new_zeros(())
    v0, v1 = v[:, :-1], v[:, 1:]
    cos = F.cosine_similarity(v0, v1, dim=-1)  # (B,T-2)
    return (1.0 - cos).mean()
