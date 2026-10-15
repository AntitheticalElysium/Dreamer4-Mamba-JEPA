"""Why do imagined mobs fade under recursion? (2026-10-15, CPU, saved recursion samples; no model runs)

transport_recursion.py: imagined cow census falls to ~0.5 of reality by depth 16 for transport operators and their own
deterministic afterstates. For every interior cell decoded as holding a mob (zombie / cow / skeleton, frozen real reader) in
imagined frame k-1 (k-1 = root for k = 1), on camera-consistent trajectories, the next imagined frame k is classed:
  stay      same type decoded at the same cell
  move      not at the cell, but decoded at one of its 4 neighbours (and that neighbour had none of that type before)
  vanish    neither (the mob is gone)
The same transition statistics on the real futures (panel frames 15..31) give the reference rates. Also the mean real-reader
score of persisting (stay) mobs by depth (a geometric decay would indicate copy leakage), and of vanishing mobs at k-1
(were they already faint?).
Usage: mob_fading.py <model> ... -> artifacts/eda/whiteboard/mob_fading_<model>.json
"""
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'artifacts/experiments/20261012_oracle'))
import audit_s5_direct as A5  # noqa: E402

OUT = ROOT / 'artifacts/eda/whiteboard'
INTERIOR = [r * 9 + c for r in range(1, 6) for c in range(1, 8) if r * 9 + c != 31]
NB = {x: [x - 9, x + 9, x - 1, x + 1] for x in INTERIOR}


def transitions(dec_prev, dec_next, sc_prev, ok):
    """dec [N,63,3] bool, sc [N,63,3] score, ok [N] -> counts and scores"""
    out = {t: {'stay': 0, 'move': 0, 'vanish': 0, 'score_vanish_prev': [], 'score_stay_prev': []} for t in range(3)}
    for n in torch.where(ok)[0].tolist():
        for t in range(3):
            for x in INTERIOR:
                if not bool(dec_prev[n, x, t]):
                    continue
                if bool(dec_next[n, x, t]):
                    out[t]['stay'] += 1; out[t]['score_stay_prev'].append(float(sc_prev[n, x, t]))
                elif any(bool(dec_next[n, y, t]) and not bool(dec_prev[n, y, t]) for y in NB[x]):
                    out[t]['move'] += 1
                else:
                    out[t]['vanish'] += 1; out[t]['score_vanish_prev'].append(float(sc_prev[n, x, t]))
    return out


def main():
    import direct_coherence as C
    torch.set_num_threads(8)
    panel = torch.load(A5.BASE / 'panel.pt', map_location='cpu', weights_only=False, mmap=True)
    bank = torch.load(A5.BASE / 'snap_bank_readers.pt', map_location='cpu', weights_only=False)
    head = torch.load(A5.BASE / 'learned_innovation/eval_u3000_h01/mob_heads.pt', map_location='cpu', weights_only=False)['heads']['real']
    selected = sorted(json.loads((A5.BASE / 'direct_generator/data/existing_eval_selection.json').read_text())['selected_indices'])
    scroll = A5.load_scroll(); shifts = torch.as_tensor(scroll.SHIFTS)
    tok = panel['tokens'][selected].float()                                # bright roots only
    real_shift = shifts[scroll.estimate(tok[:, 15:31], tok[:, 16:32])]
    rd = C.decode(tok[:, 15:32], bank, head)
    rdec, rsc = rd['mob']['real'][..., :3], rd['scores']['real'][..., :3]
    names = ['REAL'] + sys.argv[1:]
    res = {}
    for name in names:
        agg = {t: {'stay': 0, 'move': 0, 'vanish': 0, 'score_vanish_prev': [], 'score_stay_prev': []} for t in range(3)}
        by_depth = []
        if name == 'REAL':
            for k in range(16):
                still = (real_shift[:, k] == 0).all(-1)
                tr = transitions(rdec[:, k], rdec[:, k + 1], rsc[:, k], still)
                for t in range(3):
                    for key in agg[t]:
                        agg[t][key] += tr[t][key]
        else:
            imag = torch.load(A5.BASE / f'transport/recursion_samples_{name}.pt', weights_only=False)[selected].float()
            Rb, K = imag.shape[:2]
            seq = torch.cat([tok[:, 15][:, None, None].expand(-1, K, 1, -1, -1), imag], 2)       # [Rb,K,17,81,192]
            d = C.decode(seq, bank, head)
            idec, isc = d['mob']['real'][..., :3], d['scores']['real'][..., :3]                  # [Rb,K,17,63,3]
            est = shifts[scroll.estimate(seq[:, :, :-1], seq[:, :, 1:])]                          # [Rb,K,16,2]
            cam_ok = (est == real_shift[:, None]).all(-1).cumprod(2).bool()
            for k in range(16):
                still = cam_ok[:, :, k] & (real_shift[:, None, k] == 0).all(-1)
                tr = transitions(idec[:, :, k].flatten(0, 1), idec[:, :, k + 1].flatten(0, 1), isc[:, :, k].flatten(0, 1), still.flatten())
                row = {}
                for t in range(3):
                    for key in agg[t]:
                        agg[t][key] += tr[t][key]
                    tot = tr[t]['stay'] + tr[t]['move'] + tr[t]['vanish']
                    row[('zombie', 'cow', 'skeleton')[t]] = round(tr[t]['vanish'] / max(tot, 1), 3)
                stay_scores = [s for t in range(3) for s in tr[t]['score_stay_prev']]
                row['mean_score_of_persisting'] = round(float(torch.tensor(stay_scores).mean()), 3) if stay_scores else None
                by_depth.append(row)
        out = {}
        for t in range(3):
            tot = agg[t]['stay'] + agg[t]['move'] + agg[t]['vanish']
            out[('zombie', 'cow', 'skeleton')[t]] = {'transitions': tot, 'stay': round(agg[t]['stay'] / max(tot, 1), 4),
                                                    'move': round(agg[t]['move'] / max(tot, 1), 4), 'vanish': round(agg[t]['vanish'] / max(tot, 1), 4),
                                                    'score_prev_vanishing_mean': round(float(torch.tensor(agg[t]['score_vanish_prev']).mean()), 3) if agg[t]['score_vanish_prev'] else None,
                                                    'score_prev_staying_mean': round(float(torch.tensor(agg[t]['score_stay_prev']).mean()), 3) if agg[t]['score_stay_prev'] else None}
        out['vanish_rate_by_depth'] = by_depth
        res[name] = out
        print(json.dumps({name: {k: v for k, v in out.items() if k != 'vanish_rate_by_depth'}}), flush=True)
        if by_depth:
            print(json.dumps({name + '_by_depth_cow_vanish': [r['cow'] for r in by_depth], 'persist_score': [r['mean_score_of_persisting'] for r in by_depth]}), flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'mob_fading.json').write_text(json.dumps(res, indent=1))


if __name__ == '__main__':
    main()
