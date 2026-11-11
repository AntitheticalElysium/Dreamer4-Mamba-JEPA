"""Pinned pre-correction A9b correspondence, evaluation control only."""
import numpy as np
from saccade.data import _a9b

def pair_features(frames: np.ndarray):
    """frames uint8 [n + 1, 63, 63, 3] of one episode -> (features [m, F], (pair, row, col) of each feature row) for the
    map cells of every consecutive pair whose aligned source (estimated camera shift, A9b) is in view; the player's
    cell and entering cells are excluded, as in A9b."""
    A9 = _a9b()
    c = frames[:, :49].astype(np.float32).reshape(-1, 7, 7, 9, 7, 3).transpose(0, 1, 3, 2, 4, 5)   # [n+1, 7, 9, 7, 7, 3]
    c0, c1 = c[:-1], c[1:]
    cost = []
    for dr, dc in A9.SHIFTS:
        r0, r1 = max(0, -dr), min(7, 7 - dr); q0, q1 = max(0, -dc), min(9, 9 - dc)
        cost.append(np.abs(c0[:, r0:r1, q0:q1] - c1[:, r0 + dr:r1 + dr, q0 + dc:q1 + dc]).mean((1, 2, 3, 4, 5)))
    shift = np.array(A9.SHIFTS)[np.argmin(np.stack(cost, 1), 1)]                                    # [n, 2]
    rr, qq = np.meshgrid(np.arange(7), np.arange(9), indexing="ij")
    sr, sq = rr[None] - shift[:, 0, None, None], qq[None] - shift[:, 1, None, None]                 # source of each cell
    ok = (sr >= 0) & (sr < 7) & (sq >= 0) & (sq < 9) & ~((sr == 3) & (sq == 4))
    ok[:, 3, 4] = False
    pi, ri, qi = np.nonzero(ok)
    return A9.feats(c1[pi, ri, qi], c0[pi, sr[pi, ri, qi], sq[pi, ri, qi]]), (pi, ri, qi)
