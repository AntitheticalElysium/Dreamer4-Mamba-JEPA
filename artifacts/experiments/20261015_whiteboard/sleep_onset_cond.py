"""R48 conditional readouts: given zombies next to the player, does PI imagine the damage, asleep vs awake? (2026-10-16)

Inputs: sleep_onset_sub_<world>.pt (per future: root health h0, reader health and zombie adjacency at depths 1-16) and the REAL
futures of the same DV16 roots (deeppanel key-0 frames at depths 1, 2, 4, 8, 16; frozen readers; first action sleep / noop).
Readouts per variant, on depths 1, 2, 4, 8, 16 only so that imagined and real are read identically:
  P(drop | ZA)          share of futures with a health drop >= 0.5 below the root's, among futures with a zombie adjacent
  P(drop | no ZA)       the same among futures without
  after-onset damage    health at depth 16 minus health at depth 1 (excludes real_onset's real depth-1 HUD), and the share of
                        futures whose health falls >= 0.5 below their depth-1 health later (depths 2-16)
Usage: sleep_onset_cond.py -> artifacts/eda/whiteboard/sleep_onset_cond.json
"""
import json
import os
import sys
from pathlib import Path

os.environ.setdefault('TRITON_F32_DEFAULT', 'ieee')
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / '20261012_oracle'))
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / '20260927_levers'))

import torch

import dv16 as DV
from decision_proxies import hud_health

IDX = [0, 1, 3, 7, 15]                                                                        # depths 1, 2, 4, 8, 16


def readouts(h0, health, adj):
    """h0 [N]; health [N,5]; adj [N,5] bool -> dict"""
    drop = (health <= h0[:, None] - .5).any(-1)
    za = adj.any(-1)
    later = (health[:, 1:] <= health[:, :1] - .5).any(-1)
    return {'futures': len(h0), 'ZA': round(float(za.float().mean()), 4), 'drop': round(float(drop.float().mean()), 4),
            'P_drop_given_ZA': round(float(drop[za].float().mean()), 4) if za.any() else None,
            'P_drop_given_noZA': round(float(drop[~za].float().mean()), 4) if (~za).any() else None,
            'health_d16_minus_d1': round(float((health[:, -1] - health[:, 0]).mean()), 3),
            'later_drop_after_d1': round(float(later.float().mean()), 4),
            'later_drop_after_d1_given_ZA': round(float(later[za].float().mean()), 4) if za.any() else None,
            'health_change_by_depth': [round(float(v), 3) for v in (health - h0[:, None]).mean(0)]}


@torch.no_grad()
def main():
    import direct_coherence as C
    import audit_s5_direct as A5
    roots, _ = DV.load_roots()
    full = {}
    for f in sorted(DV.DEV.glob('seed-*.pt')):
        for r in torch.load(f, weights_only=False):
            full[(int(r['seed']), int(r['step']))] = r
    enc = DV.RS.Encoder()
    bank = torch.load(A5.BASE / 'snap_bank_readers.pt', map_location='cpu', weights_only=False)
    head = torch.load(A5.BASE / 'learned_innovation/eval_u3000_h01/mob_heads.pt', map_location='cpu', weights_only=False)['heads']['real']
    real = {a: {'h0': [], 'health': [], 'adj': []} for a in (0, 6)}
    for r in roots:
        row = full[(r['seed'], r['step'])]
        h0 = float(hud_health(enc.recontext(enc.content(r['frames'][-1:])).float().cpu(), bank['heads']['hud'])[0])
        for a in (0, 6):
            tok = enc.recontext(enc.content(row['depth_frames'][a])).float().cpu()          # [5,81,192]
            real[a]['h0'].append(h0); real[a]['health'].append(hud_health(tok, bank['heads']['hud']))
            real[a]['adj'].append(C.decode(tok, bank, head)['mob']['real'][..., DV.ADJ, 0].any(-1))
    res = {'real_sleep': readouts(torch.tensor(real[6]['h0']), torch.stack(real[6]['health']).float(), torch.stack(real[6]['adj'])),
           'real_noop': readouts(torch.tensor(real[0]['h0']), torch.stack(real[0]['health']).float(), torch.stack(real[0]['adj']))}
    for w in ('pi_s1', 'pi_s2'):
        d = torch.load(DV.OUT / f'sleep_onset_sub_{w}.pt', weights_only=False)
        for v, x in d.items():
            M = x['health'].shape[1]
            res[f'{w}_{v}'] = readouts(x['h0'][:, None].expand(-1, M).flatten(), x['health'][..., IDX].flatten(0, 1),
                                       x['adjacent'][..., IDX].flatten(0, 1))
    print(json.dumps(res, indent=1))
    (DV.OUT / 'sleep_onset_cond.json').write_text(json.dumps(res, indent=1))


if __name__ == '__main__':
    main()
