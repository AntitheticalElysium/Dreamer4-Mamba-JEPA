"""Shared model construction for the predictor and Flow attribution run."""

from __future__ import annotations

import hashlib
from pathlib import Path

import torch
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint

from d4mj.checkpoint import load
from d4mj.config import Config
from d4mj.transition import World


PREDICTORS = ("current", "deep_mlp", "token_transformer")
SOURCE_FILES = (
    Path("third_party/papers/2509.24527v1.pdf"),
    Path("third_party/papers/2506.09985v1.pdf"),
    Path("third_party/sources/facebookresearch__vjepa2/src/models/ac_predictor.py"),
    Path("third_party/sources/facebookresearch__vjepa2/app/vjepa_droid/train.py"),
)


class PredictorBlock(nn.Module):
    def __init__(self, width: int, heads: int, ratio: int = 4):
        super().__init__()
        self.norm_attention = nn.LayerNorm(width, eps=1e-6)
        self.attention = nn.MultiheadAttention(
            width, heads, dropout=0.0, bias=True, batch_first=True
        )
        self.norm_mlp = nn.LayerNorm(width, eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(width, ratio * width),
            nn.GELU(),
            nn.Linear(ratio * width, width),
        )

    def forward(self, tokens: Tensor) -> Tensor:
        normalized = self.norm_attention(tokens)
        attended = self.attention(
            normalized, normalized, normalized, need_weights=False
        )[0]
        tokens = tokens + attended
        return tokens + self.mlp(self.norm_mlp(tokens))


class TokenTransformerPredictor(nn.Module):
    def __init__(self, config: Config, depth: int = 4):
        super().__init__()
        self.config = config
        self.feature_projection = nn.Linear(config.d_model, config.d_model)
        self.action_projection = nn.Linear(config.d_model, config.d_model)
        self.blocks = nn.ModuleList(
            PredictorBlock(config.d_model, config.n_heads) for _ in range(depth)
        )
        self.norm = nn.LayerNorm(config.d_model, eps=1e-6)
        self.output = nn.Linear(config.d_model, config.d_spatial)

    def forward(self, inputs: Tensor) -> Tensor:
        features = self.feature_projection(inputs[..., : self.config.d_model])
        action = self.action_projection(
            inputs[:, :, 0, self.config.d_model :]
        ).unsqueeze(2)
        shape = features.shape
        tokens = torch.cat([action, features], dim=2).flatten(0, 1)
        for block in self.blocks:
            if self.training and torch.is_grad_enabled():
                tokens = checkpoint(block, tokens, use_reentrant=False)
            else:
                tokens = block(tokens)
        tokens = tokens.unflatten(0, shape[:2])[:, :, 1 : 1 + self.config.n_spatial]
        return self.output(self.norm(tokens))


def deep_mlp(config: Config) -> nn.Sequential:
    layers: list[nn.Module] = [
        nn.Linear(2 * config.d_model, 3 * config.d_model),
        nn.SiLU(),
    ]
    for _ in range(5):
        layers.extend(
            (nn.Linear(3 * config.d_model, 3 * config.d_model), nn.SiLU())
        )
    layers.append(nn.Linear(3 * config.d_model, config.d_spatial))
    return nn.Sequential(*layers)


def make_world(config: Config, predictor: str = "current") -> World:
    if predictor not in PREDICTORS:
        raise ValueError(f"unknown predictor: {predictor}")
    if config.transition != "direct" and predictor != "current":
        raise ValueError("Flow has no external Direct predictor")
    world = World(config)
    if predictor == "deep_mlp":
        world.readout = deep_mlp(config)
    elif predictor == "token_transformer":
        world.pool = nn.Identity()
        world.readout = TokenTransformerPredictor(config)
    return world


def load_world(path: Path, config: Config, predictor: str = "current") -> World:
    world = make_world(config, predictor).to(config.device)
    load(path, config, part0=world)
    world.eval()
    for parameter in world.parameters():
        parameter.requires_grad_(False)
    return world


def parameter_report(config: Config) -> dict[str, dict[str, int]]:
    output = {}
    for predictor in PREDICTORS:
        world = make_world(config, predictor)
        names = ("pool.", "readout.")
        predictor_count = sum(
            value.numel()
            for name, value in world.named_parameters()
            if name.startswith(names)
        )
        output[predictor] = {
            "predictor": predictor_count,
            "world": sum(value.numel() for value in world.parameters()),
        }
    return output


def shared_state_digest(world: World) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(world.state_dict().items()):
        if name.startswith(("pool.", "readout.")):
            continue
        digest.update(name.encode())
        digest.update(value.detach().cpu().numpy().tobytes())
    return digest.hexdigest()

