"""M3-HJWM: Mamba-3 Hierarchical Joint-Embedding World Model.

The package is deliberately modular:
- dense JEPA encoder + EMA target
- explicit spatial mixer
- swappable temporal backend (Mamba-3 target; safe GRU fallback)
- deterministic or hard-mode-mixture future predictor
- post-transition reward/continuation heads
- calibrated, shadow-first imagination reliability
- Dreamer-style actor/critic imagination loop

Nothing in the core assumes Crafter beyond discrete actions and RGB observations.
"""
from .config import ModelConfig, TrainConfig
from .world_model import M3HJWM, WorldModelState, WorldModelOutput
from .actor_critic import ActorCritic
