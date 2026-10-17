"""Read-only adapter: score existing D-class worlds (20260927_levers/tworld.py TWorld, corrt head, fmamba backbone) on the audited
Stage 2 gates (R64), to calibrate what a D-class learner on frozen Raw tokens achieves at a given training budget. No training.

D's inputs are Raw@10000 patch tokens with an extra affine-free per-token LN, i.e. exactly spatial_eval.PatchState('raw@10000').
world(s, a) returns LN(output): position t predicts frame t+1 from frames <= t and action a_t (tworld.rollout_losses alignment).
Scratch-D snapshots have a 6-row time table (trained on 6-frame windows); the long / control D have 16 rows. The diagnosis store
offers 4 context frames (t-3..t), within both. D is deterministic (one reference prediction), so 'samples' = the prediction.
"""
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'artifacts/experiments/20260927_levers'))
import tworld as TW                                                            # noqa: E402


class DWorld:
    def __init__(self, ckpt, device='cuda'):
        ck = torch.load(ckpt, map_location='cpu', weights_only=False)
        sd = ck['world']
        rows = sd['time'].shape[0]
        self.world = TW.TWorld('corrt', None, 'fmamba', 'all', 0, False, None if rows == TW.S.W else rows)
        self.world.load_state_dict(sd)                                         # strict: the exact 145 tensors
        self.world.to(device).eval()
        self.rows, self.device = rows, device

    @torch.no_grad()
    def predict(self, frames, actions):
        """frames [n, L, 81, 192] (Raw LN tokens), actions [n, L] (a_t taken at frame t) -> next frame [n, 81, 192]"""
        assert frames.shape[1] <= self.rows
        with torch.autocast('cuda', dtype=torch.bfloat16):
            out, _, _ = self.world(frames.to(self.device).float(), actions.to(self.device).long())
        return out[:, -1].float()
