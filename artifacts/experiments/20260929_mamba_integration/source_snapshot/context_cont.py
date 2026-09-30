"""E1b. Stuffed Mamba's remedy -- continue training on longer sequences -- and whether L16's gap is under-training.

Chen et al. 2025 alleviate state collapse "by leveraging a strategy of continual pre-training on extended
sequences". E1 found: an L4 world is excellent under le-wm's 3-frame window but collapses with full history; an
L16 world uses its memory (+10%) but is less accurate at the same budget. Arms, everything else as E1
(frozen Raw latents, joint optimizer settings, ~384 transitions per update, same seeds):
  L4to16   the E1 L4 world, continued 5,000 updates at L=16 (B=26); fresh AdamW, 500 warmup, cosine to 5e-6
  L4to64   the E1 L4 world, continued 5,000 updates at L=64 (B=6)
  L16x3    a fresh world trained at L=16 for 30,000 updates (3x E1's budget) -- is L16's gap under-training?
  L64b24   (E1d) a fresh world at L=64 with 24 windows per update (4x E1's transitions) -- length or diversity?
  L4b512, L16b100  (E1e) the same 4x transitions per update (1,536 / 1,500) at L=4 and L=16: matched compute
  L4to64b24 (E1f) the L4 world continued at L=64 with 24 windows per update: continuation without the diversity gap
Evaluated exactly as E1 (`context_length.evaluate`).
"""
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

import torch

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import context_length as C  # noqa: E402

ARMS = {"L4to16": ("L4", 16, 26, 5000), "L4to64": ("L4", 64, 6, 5000), "L16x3": (None, 16, 26, 30000),
        # E1d: is L64's accuracy gap the length or the 6 windows per update? Same L, 4x the windows (and compute).
        "L64b24": (None, 64, 24, 10000),
        # E1e: matched compute for L64b24 (1,512 transitions per update): is its win the length, or just 4x data?
        "L4b512": (None, 4, 512, 10000), "L16b100": (None, 16, 100, 10000),
        # E1f: the canonical pipeline's shape (joint 4-frame world, then long-context continuation, as the H2 bridge
        # does) but with E1d's diversity: continue L4 at L=64 with 24 windows per update, 5,000 updates.
        "L4to64b24": ("L4", 64, 24, 5000)}


def main():
    device = torch.device("cuda")
    started = time.time()
    log = lambda **kw: print(json.dumps({**kw, "seconds": round(time.time() - started, 1)}), flush=True)
    config, episodes = C.load(device)
    out = HERE / "context_cont.json"
    result = json.loads(out.read_text()) if out.exists() else {}
    for name, (parent, length, batch, updates) in ARMS.items():
        path = C.OUT / f"{name}.pt"
        if name in result:
            continue
        cfg = replace(config, joint=replace(config.joint, steps=updates))
        C.UPDATES = updates
        if path.exists():
            world = C.new_world(cfg, device)
            world.load_state_dict(torch.load(path, map_location="cpu", weights_only=False)["world"]); world.eval()
            history = []
        else:
            if parent is None:
                world, history = C.train(cfg, episodes, length, batch, device, log)
            else:
                init = torch.load(C.OUT / f"{parent}.pt", map_location="cpu", weights_only=False)["world"]
                original = C.new_world
                C.new_world = lambda c, d: (lambda w: (w.load_state_dict(init), w)[1])(original(c, d))
                try:
                    world, history = C.train(cfg, episodes, length, batch, device, log)
                finally:
                    C.new_world = original
            torch.save({"world": world.state_dict(), "history": history, "parent": parent, "L": length, "B": batch,
                        "updates": updates}, path)
        result[name] = {"parent": parent, "L": length, "batch": batch, "updates": updates, "history": history,
                        **C.evaluate(world, cfg, episodes, device)}
        log(stage="eval", arm=name, **{k: v for k, v in result[name].items() if k != "history"})
        out.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
