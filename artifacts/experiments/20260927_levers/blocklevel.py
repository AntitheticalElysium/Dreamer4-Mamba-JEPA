"""E8 diagnostic. Does a world trained on unreliable inputs fall back on the action prior when told its input is unreliable?

blockwin.py on the E8 `noise` world, unchanged, except that every input frame is labelled with noise level k through
the world's level embedding while the frames themselves stay CLEAN (true tokens). The world can only learn how far to
trust its input from that label, so the blocked-move scroll rate as a function of k measures how much it substitutes
the action prior ("move actions usually succeed") for the drawn target tile when told the input is noisy.
Usage: blocklevel.py <noise world.pt> 0 3 6 9 -> blockwin_<name>_level<k>.json per level
"""
import sys
from pathlib import Path

import torch

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import blockwin  # noqa: E402
import teval as T  # noqa: E402


def main():
    path, levels = sys.argv[1], [int(x) for x in sys.argv[2:]]
    load = T.load_world
    for k in levels:
        def labelled(p, device, k=k):
            world, st = load(p, device)
            inputs = world.inputs

            def with_level(s, a):
                world.level = torch.full(s.shape[:2], k, dtype=torch.long, device=s.device)
                return inputs(s, a)
            world.inputs = with_level
            return world, st | {"name": f"{st['name']}_level{k}"}
        T.load_world = labelled
        sys.argv = [sys.argv[0], path]
        blockwin.main()
    T.load_world = load


if __name__ == "__main__":
    main()
