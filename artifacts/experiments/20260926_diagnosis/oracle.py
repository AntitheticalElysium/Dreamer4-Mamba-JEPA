"""D14. Is a successful move's change learnable from the observation, or unobservable? Rule-based oracles.

For every successful move (onestep class `moved`, key-0 successor) in the diagnostic futures:
  persist     the root frame (no change)
  shift       the root frame with its 7x9-tile map scrolled one tile opposite to the move (7 px); the revealed
              row/column filled by repeating the nearest known one; the player tile kept from the root (player
              sprite, old facing); the tile the player left filled with the tile it stepped onto; HUD from the root.
              Uses only the root frame and the fact that the move succeeded.
  strip       the TRUE successor with only its revealed row/column replaced by the same repeat-the-edge guess:
              the cost of not knowing the newly revealed content, everything else exact.
Each frame is encoded and put in every world's state space (T in its PCA-1024). Error against the true
successor / V; captured = 1 - err / err(persist). Compared with each world's own one-step prediction.
"""
import json
import sys
from pathlib import Path

import torch

HERE = Path(__file__).parent
ROOT = Path("/home/antithetical/EPITA/PERSO/DynamicHorizons-Mamba-JEPA")
LADDER = ROOT / "artifacts/experiments/20260921_readout_ladder"
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(LADDER)); sys.path.insert(0, str(HERE))
import interface as I  # noqa: E402
from onestep import classify  # noqa: E402
from rollouts import encode, load_roots, states  # noqa: E402

DATA = ROOT / "artifacts/eda/diagnosis_rollouts_v1"
WORLDS = ("H2", "Z", "sZ", "U", "W", "T")
DIRS = {1: (0, -1), 2: (0, 1), 3: (-1, 0), 4: (1, 0)}
P = 7


def shift_frame(root, a):
    """Scroll the map area (rows 0-48) opposite to the move; see module docstring."""
    dr, dc = DIRS[a]
    f = root.clone()
    m = root[:49]
    out = torch.roll(m, shifts=(-dr * P, -dc * P), dims=(0, 1))
    # revealed strip: the side the player moved toward
    if dc == -1: out[:, :P] = out[:, P:2 * P]
    if dc == 1: out[:, -P:] = out[:, -2 * P:-P]
    if dr == -1: out[:P] = out[P:2 * P]
    if dr == 1: out[-P:] = out[-2 * P:-P]
    cr, cc = 3 * P, 4 * P
    stepped = m[cr + dr * P:cr + dr * P + P, cc + dc * P:cc + dc * P + P]            # tile moved onto (root)
    out[cr - dr * P:cr - dr * P + P, cc - dc * P:cc - dc * P + P] = stepped            # the tile left behind
    out[cr:cr + P, cc:cc + P] = m[cr:cr + P, cc:cc + P]                               # player sprite
    f[:49] = out
    return f


def strip_frame(true, a):
    dr, dc = DIRS[a]
    f = true.clone()
    m = f[:49]
    if dc == -1: m[:, :P] = m[:, P:2 * P]
    if dc == 1: m[:, -P:] = m[:, -2 * P:-P]
    if dr == -1: m[:P] = m[P:2 * P]
    if dr == 1: m[-P:] = m[-2 * P:-P]
    return f


def main():
    device = torch.device("cuda")
    from whiten import whitened_pool
    pool, _ = whitened_pool()
    encoder, _ = I.load_bridge()
    meta = torch.load(DATA / "meta.pt")
    cls, _ = classify(meta)
    roots = load_roots()
    rows = [(i, a) for i in range(len(roots)) for a in (1, 2, 3, 4) if cls[i, a] == 0]
    frames = {"persist": [], "shift": [], "strip": [], "true": []}
    for i, a in rows:
        root, true = roots[i]["context"][-1], roots[i]["onestep_frames"][0, a]
        frames["persist"].append(root); frames["true"].append(true)
        frames["shift"].append(shift_frame(root, a)); frames["strip"].append(strip_frame(true, a))
    tf = torch.load(DATA / "T_fullspace.pt")
    proj = lambda x: (x.flatten(-2).float() - tf["t_mean"]) @ tf["t_basis"]
    enc = {k: states(pool, *[x.cpu() for x in encode(encoder, torch.stack(v), device)]) for k, v in frames.items()}
    idx_r = torch.tensor([i for i, _ in rows]); idx_a = torch.tensor([a for _, a in rows])
    result = {"n_moves": len(rows)}
    for w in WORLDS:
        d = torch.load(DATA / f"{w}.pt")
        V = float(torch.cov(d["true"][:, 0].flatten(0, 1).T.double()).trace())
        st = {k: (proj(v[w]) if w == "T" else v[w]) for k, v in enc.items()}
        truth = st["true"]
        err = lambda x: float(((x - truth) ** 2).sum(-1).mean() / V)
        world = d["one"][idx_r, idx_a]
        e = {"persist": err(st["persist"]), "shift_oracle": err(st["shift"]), "strip_only": err(st["strip"]),
             "world": err(world)}
        e.update({f"captured_{k}": 1 - e[k] / e["persist"] for k in ("shift_oracle", "strip_only", "world")})
        result[w] = e
        print(w, json.dumps({k: round(v, 3) for k, v in e.items()}), flush=True)
    (HERE / "oracle.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
