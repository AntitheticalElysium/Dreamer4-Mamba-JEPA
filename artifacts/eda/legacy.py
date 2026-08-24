"""Loading archived diagnostic checkpoints after the production promotion.

797b448 promoted the one-block mixer into `d4mj.transition.World` as `direct_mixer` /
`direct_norm` with a plain Linear `readout`. Every checkpoint written before that has a
different head shape, so the ordinary strict loader raises on all of them:

  abt0, abt1   pre-promotion Direct: readout is Sequential(Linear(2d, d), SiLU,
               Linear(d, d_spatial)), action broadcast over pooled features
  abn0, abn1   the same without the output tanh
  abm0, abm1   the diagnostic mixer arm: legacy readout kept but unused, plus `mixer`
               and `mix_norm`, which production now calls direct_mixer / direct_norm

These definitions exist so the foundational A/B stays re-runnable without retraining
and without carrying dead shapes in production. Nothing here belongs in d4mj/.

`open_checkpoint` is the single entry point; diagnostics should not call
`d4mj.checkpoint.load` on an archived arm directly.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch import nn

ROOT = Path("/home/antithetical/EPITA/PERSO/DynamicHorizons-Mamba-JEPA")
sys.path.insert(0, str(ROOT))
from d4mj.transition import World

DEVICE = "cuda"


class LegacyDirectWorld(World):
    """Direct as it stood before the promotion: action broadcast over pooled features."""

    def __init__(self, config):
        super().__init__(config)
        del self.direct_mixer, self.direct_norm, self.readout
        d = config.d_model
        self.readout = nn.Sequential(nn.Linear(2 * d, d), nn.SiLU(),
                                     nn.Linear(d, config.d_spatial))

    def head(self, features, action):
        world = torch.cat([features[:, :, self.spatial], features[:, :, self.register]], dim=2)
        pooled = self.pool(world.transpose(2, 3)).transpose(2, 3)
        context = self.action_embed(action)[:, :, None].expand_as(pooled)
        return self.readout(torch.cat([pooled, context], dim=-1))

    def predict(self, features, action=None):
        return torch.tanh(self.head(features, action))


class LegacyNoTanhWorld(LegacyDirectWorld):
    """The no-tanh ablation arm."""

    def predict(self, features, action=None):
        return self.head(features, action)


class LegacyMixerWorld(LegacyDirectWorld):
    """The diagnostic mixer arm, under the attribute names it was trained with.

    `readout[0:2]` was constructed but never used, so that the arm's initialisation
    stayed bit-identical to its control; it is rebuilt here for the same reason the
    checkpoint carries it.
    """

    def __init__(self, config):
        super().__init__(config)
        d = config.d_model
        self.mixer = nn.TransformerEncoderLayer(d, config.n_heads, 4 * d, dropout=0.0,
                                                 batch_first=True, norm_first=True)
        self.mix_norm = nn.LayerNorm(d)

    def mixed(self, features, action):
        world = torch.cat([features[:, :, self.spatial], features[:, :, self.register]], dim=2)
        pooled = self.pool(world.transpose(2, 3)).transpose(2, 3)
        b, t, s, d = pooled.shape
        token = torch.cat([self.action_embed(action)[:, :, None], pooled], dim=2)
        token = self.mixer(token.reshape(b * t, s + 1, d)).view(b, t, s + 1, d)
        return self.mix_norm(token[:, :, 1:])

    def predict(self, features, action=None):
        return torch.tanh(self.readout[2](self.mixed(features, action)))


KINDS = {"direct": LegacyDirectWorld, "notanh": LegacyNoTanhWorld, "mixer": LegacyMixerWorld,
         "promoted": World}
RENAME = {"mixer.": "direct_mixer.", "mix_norm.": "direct_norm.", "readout.2.": "readout."}


def open_checkpoint(path: Path, config, kind: str, device: str = DEVICE):
    """Build the right module for an archived arm and load it, strictly.

    `promoted` additionally remaps a pre-promotion mixer checkpoint onto production's
    names, which is how the promoted architecture was verified to reproduce the
    diagnostic arm exactly.
    """
    world = KINDS[kind](config).to(device)
    state = torch.load(path, weights_only=False)["modules"]["part0"]
    if kind == "promoted":
        remapped = {}
        for key, value in state.items():
            if key.startswith("readout.0."):
                continue
            for old, new in RENAME.items():
                if key.startswith(old):
                    key = new + key[len(old):]
                    break
            remapped[key] = value
        state = remapped
    world.load_state_dict(state)
    world.eval()
    for parameter in world.parameters():
        parameter.requires_grad_(False)
    return world
