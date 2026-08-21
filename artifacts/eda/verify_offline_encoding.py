"""Preflight: does offline re-encoding of the saved history reproduce the rollout Z*?

If this fails the whole capacity comparison is invalid, because each arm would be
encoding a different effective state than the production checkpoint did.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path("/home/antithetical/EPITA/PERSO/DynamicHorizons-Mamba-JEPA")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))
from d4mj.checkpoint import load
from d4mj.config import Config
from d4mj.data import patchify
from d4mj.representation import Encoder, pack

config = Config()
encoder = Encoder(config).to(config.device)
load(ROOT / "artifacts/stage_a_terminalfix/phase1a.pt", config, part0=encoder)
encoder.eval()

rows = []
for path in sorted((HERE / "root_frames").glob("shard-*.pt")):
    rows += torch.load(path, weights_only=False)
    if len(rows) >= 100:
        break
rows = rows[:100]
print(f"checking {len(rows)} roots")

worst, deltas = 0.0, []
with torch.no_grad():
    for row in rows:
        patches = patchify(row["frames"][None], config.patch).to(config.device)
        z, _, _ = encoder(patches)
        offline = pack(z, config)[0, -1].reshape(-1).cpu()
        delta = (offline - row["z_rollout"]).abs().max().item()
        worst = max(worst, delta)
        deltas.append(delta)
deltas = np.array(deltas)
scale = torch.stack([r["z_rollout"] for r in rows]).abs().mean().item()
print(f"max |offline - rollout|  {worst:.3e}")
print(f"mean                     {deltas.mean():.3e}")
print(f"mean |Z*| for scale      {scale:.4f}")
print(f"exact matches            {(deltas == 0).sum()}/{len(deltas)}")
print(f"within 1e-5              {(deltas < 1e-5).sum()}/{len(deltas)}")
print("\nVERDICT:", "PASS -- offline path reproduces the rollout latent"
      if worst < 1e-4 else "FAIL -- do not run the capacity comparison")
