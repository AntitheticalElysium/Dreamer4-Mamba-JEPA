"""E6d. Why State Passing does not repair the L4 world: test Buitrago Ruiz & Gu's mechanism directly.

Their account (arXiv 2507.02782, s4 and Fig. 4): past the training length the recurrent state reaches a distribution
never seen in training; after State Passing "the distribution of attainable states does not change much after the
training context". Worlds (E1e/E6c checkpoints): L4b512 (base), its State Passing and TBTT post-trainings (500 updates),
L4to64b24 and L64b24 ((b)). 256 DEV/FINAL windows of 128 frames (context_length.Windows, seed 21), stepped one pair at a
time from a zero state (the full-history recurrence). Per position t and layer: mean Frobenius norm of the SSM state
(heads x headdim x d_state), and the one-step squared error of the prediction at t. Reported: norms at t = 3 (the
L4 training edge), 8, 16, 32, 64, 127, relative to t = 3, per layer and averaged; error x copy at the same positions.
"""
import json
import sys
from pathlib import Path

import torch

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import context_length as C  # noqa: E402

WORLDS = {"L4b512": "L4b512.pt", "L4b512_sp500": "L4b512_sp500.pt", "L4b512_tbtt500": "L4b512_tbtt500.pt",
          "L4to64b24": "L4to64b24.pt", "L64b24": "L64b24.pt"}
POSITIONS = (1, 2, 3, 4, 8, 16, 32, 64, 127)


@torch.no_grad()
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    backend = "triton" if device.type == "cuda" else "reference"
    config, episodes = C.load(device)
    held = [e for e in episodes if e.split in ("dev", "final")]
    z, a = C.Windows(held, 128, seed=21).sample(256)
    z, a = z.to(device), a.to(device)
    copy = (z[:, 1:] - z[:, :-1]).float().square().mean((-1, -2))                        # [n,127]
    out = {}
    for name, file in WORLDS.items():
        world = C.new_world(config, device)
        world.load_state_dict(torch.load(C.OUT / file, map_location="cpu", weights_only=False)["world"])
        world.eval()
        memory, norms, errs = None, [], []
        for t in range(127):
            pred, _, memory = world.scan_pairs(z[:, t:t + 1], a[:, t:t + 1], memory, backend=backend)
            norms.append(torch.stack([m.ssm.float().flatten(1).norm(dim=1).mean() for m in memory]))    # [layers]
            errs.append((pred.float() - z[:, t + 1:t + 2].float()).square().mean((-1, -2))[:, 0])       # [n]
        norms = torch.stack(norms)                                                         # [127 positions, layers]
        errs = torch.stack(errs, 1)                                                        # [n,127]
        # norms[t-1] is the state after consuming pair t-1 -> used to predict frame t
        rel = {t: (norms[t - 1] / norms[2]).tolist() for t in POSITIONS}
        out[name] = {"norm_mean_over_layers": {t: float(norms[t - 1].mean()) for t in POSITIONS},
                     "norm_relative_to_t3_per_layer": rel,
                     "norm_relative_to_t3_mean": {t: float((norms[t - 1] / norms[2]).mean()) for t in POSITIONS},
                     "error_x_copy": {t: float(errs[:, t - 1].mean() / copy[:, t - 1].mean()) for t in POSITIONS}}
        print(name, json.dumps({k: {t: round(v, 3) for t, v in d.items()} for k, d in out[name].items()
                                if k != "norm_relative_to_t3_per_layer"}), flush=True)
    (HERE / "statenorm.json").write_text(json.dumps(out, indent=2) + "\n")


if __name__ == "__main__":
    main()
