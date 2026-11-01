"""Bounded mechanical checkpoint, not production P1 (see PROTOCOL.md)."""
import json
import time
from pathlib import Path

import torch

from saccade import train as TR

OUT = Path(__file__).resolve().parent

if __name__ == "__main__":
    cfg = TR.load_cfg()
    cfg["p1"]["frames"] = 20_480
    cfg["p1"]["checkpoint_every_fraction"] = .5
    cfg["p1"]["monitor"]["windows"] = 4
    cfg["optimizer"]["warmup_frames"] = 1_024
    start = time.time()
    TR.p1(cfg, 1898, OUT / "smoke", updates=20, steps=64, n_episodes=64, log_every=5)
    (OUT / "smoke_cost.json").write_text(json.dumps({"seconds": time.time() - start,
        "peak_gib": torch.cuda.max_memory_allocated() / 2**30}, indent=2) + "\n")
