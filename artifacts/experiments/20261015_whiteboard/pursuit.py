"""Zombie pursuit in imagination (2026-10-15, CPU, saved recursion samples; no model runs).

decision_proxies.py: real zombie adjacency to the player rises over 16 steps; imagined adjacency stays flat or falls. Per
depth k, for every zombie decoded in frame k-1 (real mob reader) on camera-consistent imagined trajectories, the nearest
zombie decoded in frame k within one cell of it (same cell or a 4-neighbour; else 'lost') gives its Manhattan distance to the
player (cell 31) before and after. Classes: approach (distance falls), stay (equal), retreat, lost. Same statistics on the
real futures. Camera shifts are accounted for (positions mapped to frame k coordinates by the real shift).
Also split by the zombie's real-reader score in frame k-1 (crisp >= 0.6 vs faint < 0.6), to test whether lost pursuit is
explained by token fading (R7).
Usage: pursuit.py <model> ... -> artifacts/eda/whiteboard/pursuit.json
"""
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'artifacts/experiments/20261012_oracle'))
import audit_s5_direct as A5  # noqa: E402

OUT = ROOT / 'artifacts/eda/whiteboard'


def dist(x):
    r, c = divmod(x, 9)
    return abs(r - 3) + abs(c - 4)


def tally(prev, nxt, shift, acc, score=None, acc_crisp=None, acc_faint=None):
    """prev, nxt [63] bool zombie maps; shift (dr, dc): next[r,c] = prev[r+dr, c+dc]"""
    dr, dc = shift
    for x in torch.where(prev)[0].tolist():
        target = acc if score is None else (acc_crisp if float(score[x]) >= .6 else acc_faint)
        r, c = divmod(x, 9)
        r2, c2 = r - dr, c - dc                                            # the same world cell in frame k coordinates
        cands = [(r2 + a, c2 + e) for a, e in ((0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)) if 0 <= r2 + a < 7 and 0 <= c2 + e < 9]
        hits = [y * 9 + z for y, z in cands if bool(nxt[y * 9 + z])]
        d0 = abs(r2 - 3) + abs(c2 - 4)
        if not hits:
            target['lost'] += 1
            if score is not None:
                acc['lost'] += 1
            continue
        d1 = min(dist(h) for h in hits)
        cls = 'approach' if d1 < d0 else ('stay' if d1 == d0 else 'retreat')
        target[cls] += 1
        if score is not None:
            acc[cls] += 1


def main():
    import direct_coherence as C
    panel = torch.load(A5.BASE / 'panel.pt', map_location='cpu', weights_only=False, mmap=True)
    bank = torch.load(A5.BASE / 'snap_bank_readers.pt', map_location='cpu', weights_only=False)
    head = torch.load(A5.BASE / 'learned_innovation/eval_u3000_h01/mob_heads.pt', map_location='cpu', weights_only=False)['heads']['real']
    selected = sorted(json.loads((A5.BASE / 'direct_generator/data/existing_eval_selection.json').read_text())['selected_indices'])
    scroll = A5.load_scroll(); shifts = torch.as_tensor(scroll.SHIFTS)
    real = panel['tokens'][selected, 15:32].float()
    real_shift = shifts[scroll.estimate(real[:, :-1], real[:, 1:])]       # [Rb,16,2]
    rz = C.decode(real, bank, head)['mob']['real'][..., 0]                # [Rb,17,63] zombies
    res = {}
    for name in ['REAL'] + sys.argv[1:]:
        z0 = lambda: {'approach': 0, 'stay': 0, 'retreat': 0, 'lost': 0}
        by = {'depth1': z0(), 'depth2-5': z0(), 'depth6-16': z0(), 'crisp_depth2-16': z0(), 'faint_depth2-16': z0()}
        key = lambda k: 'depth1' if k == 0 else ('depth2-5' if k <= 4 else 'depth6-16')
        if name == 'REAL':
            for i in range(len(real)):
                for k in range(16):
                    tally(rz[i, k], rz[i, k + 1], tuple(int(v) for v in real_shift[i, k]), by[key(k)])
        else:
            imag = torch.load(A5.BASE / f'transport/recursion_samples_{name}.pt', weights_only=False)[selected].float()
            K = imag.shape[1]
            for i in range(len(real)):
                seq = torch.cat([real[i, :1][None].expand(K, 1, -1, -1), imag[i]], 1)       # [K,17,81,192]
                dd = C.decode(seq, bank, head)
                iz, isc = dd['mob']['real'][..., 0], dd['scores']['real'][..., 0]
                est = shifts[scroll.estimate(seq[:, :-1], seq[:, 1:])]
                ok = (est == real_shift[i][None]).all(-1).cumprod(1).bool()
                for s in range(K):
                    for k in range(16):
                        if bool(ok[s, k]):
                            if k == 0:
                                tally(iz[s, k], iz[s, k + 1], tuple(int(v) for v in real_shift[i, k]), by[key(k)])
                            else:
                                tally(iz[s, k], iz[s, k + 1], tuple(int(v) for v in real_shift[i, k]), by[key(k)], isc[s, k],
                                      by['crisp_depth2-16'], by['faint_depth2-16'])
        out = {}
        for kk, v in by.items():
            tot = sum(v.values())
            out[kk] = {'n': tot, **{c: round(n / max(tot, 1), 3) for c, n in v.items()}}
        res[name] = out
        print(json.dumps({name: out}), flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'pursuit.json').write_text(json.dumps(res, indent=1))


if __name__ == '__main__':
    main()
