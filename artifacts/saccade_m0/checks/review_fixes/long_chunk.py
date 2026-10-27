"""One accumulated 256-step update, including diagnostics and a tiny monitor; no production P1."""
import json
import time
from pathlib import Path

import torch

from saccade import train as TR

OUT = Path(__file__).resolve().parent

if __name__ == "__main__":
    cfg = TR.load_cfg()
    cfg["p1"]["frames"] = 4_096
    cfg["p1"]["checkpoint_every_fraction"] = 1.
    cfg["p1"]["monitor"]["windows"] = 2
    cfg["optimizer"]["warmup_frames"] = 256
    start = time.time()
    TR.p1(cfg, 1899, OUT / "long_chunk", updates=1, steps=256, n_episodes=64, log_every=1)
    (OUT / "long_chunk_cost.json").write_text(json.dumps({"seconds_including_initialization": time.time() - start,
        "peak_gib": torch.cuda.max_memory_allocated() / 2**30}, indent=2) + "\n")
