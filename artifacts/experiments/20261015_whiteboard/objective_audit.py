"""Objective audit (2026-10-16; user question: are we rewarding the model for predicting what matters?). GPU inference; exact simulator
state is used ONLY to label cells for this audit.

Data: diagnosis_futures_v1, 240 seeded roots. Per root: the last context frame and, for each of the 17 actions, the 4 one-step
futures (independent RNG keys). Frozen Raw tokens (the targets D / PI regress onto). Per next frame, the exact visible state
gives the true camera shift (tile-grid match over the 5 shifts) and labels every cell:
  persist      map cell whose source cell (after the shift) has the same tile class and the same mob layer
  entering     map cell with no source (revealed by the shift)
  mob_change   source exists, tile same, mob layer differs (moved / appeared / left)
  tile_change  source exists, tile class differs (mined, placed, ...)
  player       cell 31
  hud_changed  HUD cell, a displayed value changed (health / food / drink / energy, inventory; observe.py layout; not facing, sleep,
               light)
  hud_same     HUD cell, displayed values unchanged
Readouts per category: share of cells; share of the total squared token change ||z_next - z_source||^2 (the copy residual a
regression world must explain) and of its absolute value; the aleatoric share: the squared deviation from the 4-key mean
(irreducible for any predictor of the conditional mean), as a fraction of the category's change. Also, at mob_change cells where
keys disagree: whether the frozen mob reader still reads a mob in the 4-key MEAN token (the regression optimum), against the rate
in the individual keys.
Usage: objective_audit.py -> artifacts/eda/whiteboard/objective_audit.json
"""
import json
import os
import sys
from pathlib import Path

os.environ.setdefault('TRITON_F32_DEFAULT', 'ieee')
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / '20261012_oracle'))
sys.path.insert(0, str(HERE))

import torch

import recontext_stale as RS

ROOT = HERE.parents[2]
DIAG = ROOT / 'artifacts/eda/diagnosis_futures_v1'
SH = ((0, 0), (-1, 0), (1, 0), (0, -1), (0, 1))
CATS = ('persist', 'entering', 'mob_change', 'tile_change', 'player', 'hud_changed', 'hud_same')


def layout(vis):
    """vis [...,1534] -> tiles [...,63] long, mob layer [...,63] long (bitmask of zombie / cow / skeleton / other), hud [...,22]"""
    v = vis.float()
    tiles = v[..., :1071].reshape(*v.shape[:-1], 63, 17).argmax(-1)
    mob = v[..., 1071:1512].reshape(*v.shape[:-1], 63, 7) > 0
    lay = mob[..., 0].long() + 2 * mob[..., 1].long() + 4 * mob[..., 2].long() + 8 * mob[..., 3:].any(-1).long()
    shown = torch.cat([v[..., 1512:1516], v[..., 1522:1534]], -1)                    # health food drink energy + inventory (observe.py)
    return tiles, lay, (shown * 9).round()                                            # the displayed integers


def shift_of(t0, t1):
    """t0, t1 [63] tiles -> (dr, dc) minimizing the mismatch of t1[r,c] vs t0[r+dr, c+dc]"""
    a, b = t0.view(7, 9), t1.view(7, 9)
    best, arg = None, (0, 0)
    for dr, dc in SH:
        rs = slice(max(0, -dr), 7 - max(0, dr)); cs = slice(max(0, -dc), 9 - max(0, dc))
        rt = slice(max(0, dr), 7 + min(0, dr)); ct = slice(max(0, dc), 9 + min(0, dc))
        e = float((b[rs, cs] != a[rt, ct]).float().mean())
        if best is None or e < best:
            best, arg = e, (dr, dc)
    return arg


def sources(dr, dc):
    """[63] source map cell index for each next map cell (-1 if entering)"""
    src = torch.full((63,), -1, dtype=torch.long)
    for r in range(7):
        for c in range(9):
            rr, cc = r + dr, c + dc
            if 0 <= rr < 7 and 0 <= cc < 9:
                src[r * 9 + c] = rr * 9 + cc
    return src


@torch.no_grad()
def main():
    import direct_coherence as C
    import audit_s5_direct as A5
    bank = torch.load(A5.BASE / 'snap_bank_readers.pt', map_location='cpu', weights_only=False)
    head = torch.load(A5.BASE / 'learned_innovation/eval_u3000_h01/mob_heads.pt', map_location='cpu', weights_only=False)['heads']['real']
    rows = [r for f in sorted(DIAG.glob('seed-*.pt')) for r in torch.load(f, weights_only=False)]
    pick = torch.randperm(len(rows), generator=torch.Generator().manual_seed(20261060))[:240].tolist()
    enc = RS.Encoder()
    acc = {c: {'cells': 0, 'sq': 0., 'abs': 0., 'alea': 0.} for c in CATS}
    mean_mob = {'cells': 0, 'keys_read_mob': 0., 'mean_reads_mob': 0}
    n_trans = 0
    for ri in pick:
        r = rows[ri]
        root = enc.recontext(enc.content(r['context'][-1:])).float().cpu()[0]                         # [81,192]
        t0, l0, h0 = layout(r['root_visible'][None])
        t0, l0, h0 = t0[0], l0[0], h0[0]
        fr = r['onestep_frames']                                                                        # [4,17,63,63,3]
        nxt = enc.recontext(enc.content(fr.flatten(0, 1))).float().cpu().view(4, 17, 81, 192)
        t1, l1, h1 = layout(r['onestep_visible'])                                                      # [4,17,...]
        dead = r['onestep_dead'].bool()                                                                 # [4,17]
        for a in range(17):
            ok = ~dead[:, a]
            if ok.sum() < 2:
                continue
            ks = torch.nonzero(ok)[:, 0]
            mean_tok = nxt[ks, a].mean(0)                                                               # [81,192] regression optimum
            for k in ks.tolist():
                dr, dc = shift_of(t0, t1[k, a])
                src = sources(dr, dc)
                srcfull = torch.cat([src, torch.arange(63, 81)])
                cat = []
                for c in range(81):
                    if c >= 63:
                        cat.append('hud_changed' if bool((h1[k, a] != h0).any()) else 'hud_same')
                    elif c == 31:
                        cat.append('player')
                    elif src[c] < 0:
                        cat.append('entering')
                    elif t1[k, a, c] != t0[src[c]]:
                        cat.append('tile_change')
                    elif l1[k, a, c] != l0[src[c]]:
                        cat.append('mob_change')
                    else:
                        cat.append('persist')
                z = nxt[k, a]
                base = torch.where((srcfull >= 0)[:, None], root[srcfull.clamp_min(0)], torch.zeros_like(root))
                sq = (z - base).pow(2).sum(-1); ab = (z - base).abs().sum(-1)
                al = (z - mean_tok).pow(2).sum(-1)
                for c in range(81):
                    A = acc[cat[c]]
                    A['cells'] += 1; A['sq'] += float(sq[c]); A['abs'] += float(ab[c]); A['alea'] += float(al[c])
                n_trans += 1
            # regression optimum at stochastic mob cells: cells whose mob layer differs across keys (after own shift)
            lays = torch.stack([l1[k, a] for k in ks.tolist()])                                         # [K,63]
            varying = (lays != lays[0]).any(0)
            if bool(varying.any()):
                rk = C.decode(nxt[ks, a], bank, head)['mob']['real'][:, :63, :2].any(-1)                # [K,63] zombie or cow
                rm = C.decode(mean_tok[None], bank, head)['mob']['real'][0, :63, :2].any(-1)
                v = varying & rk.any(0)
                mean_mob['cells'] += int(v.sum()); mean_mob['keys_read_mob'] += float(rk[:, v].float().mean(0).sum())
                mean_mob['mean_reads_mob'] += int(rm[v].sum())
    tot_sq = sum(A['sq'] for A in acc.values()); tot_abs = sum(A['abs'] for A in acc.values()); tot_cells = sum(A['cells'] for A in acc.values())
    res = {'roots': len(pick), 'transitions': n_trans, 'categories': {}}
    for c, A in acc.items():
        res['categories'][c] = {'cell_share': round(A['cells'] / tot_cells, 4), 'sq_change_share': round(A['sq'] / tot_sq, 4),
                                'abs_change_share': round(A['abs'] / tot_abs, 4),
                                'mean_sq_change_per_cell': round(A['sq'] / max(A['cells'], 1), 3),
                                'aleatoric_fraction_of_change': round(A['alea'] / max(A['sq'], 1e-9), 4)}
    res['regression_optimum_at_stochastic_mob_cells'] = {
        'cells': mean_mob['cells'], 'mob_read_rate_in_keys': round(mean_mob['keys_read_mob'] / max(mean_mob['cells'], 1), 4),
        'mob_read_in_key_mean_token': round(mean_mob['mean_reads_mob'] / max(mean_mob['cells'], 1), 4)}
    print(json.dumps(res, indent=1))
    (ROOT / 'artifacts/eda/whiteboard/objective_audit.json').write_text(json.dumps(res, indent=1))


if __name__ == '__main__':
    main()
