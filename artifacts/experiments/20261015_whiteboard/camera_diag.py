"""Why do imagined cameras diverge? (2026-10-15, CPU, saved recursion samples; no model runs)

transport_recursion.py found that only 50-59% of imagined H16 trajectories keep the real camera at depth 16, for every model
(deterministic or stochastic). The camera shift is the player's movement: action LEFT/RIGHT/UP/DOWN scrolls the view unless the
target tile is blocked (solid block, water, lava, a mob), so it is deterministic given the map, the action and the mobs.
From the saved samples (8 per root for stochastic models, 1 for deterministic) and the real panel future: at each depth k, among
trajectories still camera-consistent up to k-1, the first-divergence rate and its kind:
  missed     the real view scrolled, the imagined one did not
  false      the real move was blocked (no scroll), the imagined view scrolled
  wrong      both scrolled, different directions
  noop_move  the action is not a move but one of them scrolled
Also by the frozen bank tile reading of the tile the player faces (real frame at k-1) and whether a mob is decoded there,
and the share of first divergences that occur at depth 1 (teacher-forced context: a one-step dynamics error) vs later.
Faced-cell test (is divergence caused by imagined content at the cell the player faces?): for move actions at steps where the
trajectory is still camera-consistent, compare the IMAGINED frame k-1's faced cell with the real one (bank tile class, real
mob reader): agree / imagined mob missing (real mob, imagined none) / imagined mob extra / tile differs; divergence rate in
each class.
Usage: camera_diag.py <model> ... -> artifacts/eda/whiteboard/camera_<model>.json
"""
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'artifacts/experiments/20261012_oracle'))
import audit_s5_direct as A5  # noqa: E402

OUT = ROOT / 'artifacts/eda/whiteboard'
MOVE = {1: (0, -1), 2: (0, 1), 3: (-1, 0), 4: (1, 0)}                  # Craftax action -> (dr, dc) of the player
FACE = {1: 30, 2: 32, 3: 22, 4: 40}                                      # tile the move enters (player at 31)


def main():
    import direct_coherence as C
    torch.set_num_threads(8)
    panel = torch.load(A5.BASE / 'panel.pt', map_location='cpu', weights_only=False, mmap=True)
    bank = torch.load(A5.BASE / 'snap_bank_readers.pt', map_location='cpu', weights_only=False)
    head = torch.load(A5.BASE / 'learned_innovation/eval_u3000_h01/mob_heads.pt', map_location='cpu', weights_only=False)['heads']['real']
    scroll = A5.load_scroll()
    shifts = torch.as_tensor(scroll.SHIFTS)
    real = panel['tokens'][:, 15:32].float()                               # [R,17] root + 16 real successors
    act = panel['actions'][:, 15:31]                                       # action taken at depth k-1 -> frame k
    real_shift = shifts[scroll.estimate(real[:, :-1], real[:, 1:])]        # [R,16,2]
    real_dec = C.decode(real[:, :-1], bank, head)                          # facts of the frame the action is taken in
    for name in sys.argv[1:]:
        imag = torch.load(A5.BASE / f'transport/recursion_samples_{name}.pt', weights_only=False).float()   # [R,K,16,81,192]
        R, K = imag.shape[:2]
        prev = torch.cat([real[:, None, :1].expand(-1, K, -1, -1, -1), imag[:, :, :-1]], 2)
        img_shift = shifts[scroll.estimate(prev, imag)]                    # [R,K,16,2]
        ok_before = torch.ones(R, K, dtype=torch.bool)
        counts = {k: {'at_risk': 0, 'diverge': 0, 'missed': 0, 'false': 0, 'wrong': 0, 'noop_move': 0} for k in range(16)}
        by_face, faced = {}, {}
        for k in range(16):
            rs = real_shift[:, k][:, None].expand(-1, K, -1); ims = img_shift[:, :, k]
            div = ok_before & (ims != rs).any(-1)
            a = act[:, k][:, None].expand(-1, K)
            rmove, imove = (rs != 0).any(-1), (ims != 0).any(-1)
            is_move = (a >= 1) & (a <= 4)
            kind = {'missed': div & rmove & ~imove, 'false': div & ~rmove & imove, 'wrong': div & rmove & imove,
                    'noop_move': div & ~is_move}
            c = counts[k]
            c['at_risk'] += int(ok_before.sum()); c['diverge'] += int(div.sum())
            for kk, m in kind.items():
                c[kk] += int((m & (is_move if kk != 'noop_move' else torch.ones_like(m))).sum())
            # what the player faces in the frame where the action is taken (real frame; same as imagined while consistent)
            for i in range(R):
                ai = int(act[i, k])
                if ai not in FACE:
                    continue
                f = FACE[ai]
                tile = int(real_dec['tile'][i, k, f]); mob = bool(real_dec['mob']['real'][i, k, f, :3].any())
                key = f"{'mob' if mob else 'tile' + str(tile)}|{'real_moved' if bool(rmove[i, 0]) else 'real_blocked'}"
                row = by_face.setdefault(key, {'at_risk': 0, 'diverge': 0})
                row['at_risk'] += int(ok_before[i].sum()); row['diverge'] += int(div[i].sum())
            # faced-cell agreement between the imagined frame k-1 and the real one (imagined k-1 = real root at k = 0)
            img_prev = prev[:, :, k]                                       # [R,K,81,192]
            idec = C.decode(img_prev, bank, head)
            for i in range(R):
                ai = int(act[i, k])
                if ai not in FACE:
                    continue
                f = FACE[ai]
                rt, rm = int(real_dec['tile'][i, k, f]), bool(real_dec['mob']['real'][i, k, f, :3].any())
                for kk in range(K):
                    if not bool(ok_before[i, kk]):
                        continue
                    it, im = int(idec['tile'][i, kk, f]), bool(idec['mob']['real'][i, kk, f, :3].any())
                    cls = 'agree' if (it == rt and im == rm) else ('mob_missing' if (rm and not im) else ('mob_extra' if (im and not rm) else 'tile_differs'))
                    row = faced.setdefault(cls, {'at_risk': 0, 'diverge': 0})
                    row['at_risk'] += 1; row['diverge'] += int(div[i, kk])
            ok_before = ok_before & ~div
        tot = {kk: sum(c[kk] for c in counts.values()) for kk in ('at_risk', 'diverge', 'missed', 'false', 'wrong', 'noop_move')}
        res = {'model': name, 'samples_per_root': K, 'per_depth': counts, 'total': tot,
               'divergence_rate_per_step': round(tot['diverge'] / tot['at_risk'], 4),
               'share_of_divergences_at_depth1': round(counts[0]['diverge'] / max(tot['diverge'], 1), 4),
               'depth1_rate': round(counts[0]['diverge'] / counts[0]['at_risk'], 4),
               'later_rate': round(sum(counts[k]['diverge'] for k in range(1, 16)) / max(sum(counts[k]['at_risk'] for k in range(1, 16)), 1), 4),
               'faced_cell_agreement': {k: {**v, 'rate': round(v['diverge'] / max(v['at_risk'], 1), 4)} for k, v in faced.items()},
               'by_faced_cell': {k: {**v, 'rate': round(v['diverge'] / max(v['at_risk'], 1), 4)} for k, v in sorted(by_face.items(), key=lambda x: -x[1]['at_risk'])}}
        OUT.mkdir(parents=True, exist_ok=True)
        (OUT / f'camera_{name}.json').write_text(json.dumps(res, indent=1))
        print(json.dumps({k: res[k] for k in ('model', 'total', 'divergence_rate_per_step', 'depth1_rate', 'later_rate', 'share_of_divergences_at_depth1')}), flush=True)
        print(json.dumps({'faced_cell_agreement': res['faced_cell_agreement']}), flush=True)


if __name__ == '__main__':
    main()
