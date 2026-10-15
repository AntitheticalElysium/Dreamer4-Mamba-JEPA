"""R29 localization 2 (2026-10-15 night, CPU, saved recursion samples): where does sampled content raise seen-terrain error?

transport_recursion's 'seen' terrain error at depth k counts cells whose world position was inside the ROOT view, on
camera-consistent trajectories, with no decoded mob in either frame. Split here by the real camera track:
  stayed      the world position stayed inside the view at every depth 1..k (never left: only copies or gate-fired edits
              could change it)
  reentered   it left the view at some depth < k and is back at k (its content was regenerated on re-entry: memory)
Usage: seen_split.py <model> ... -> prints (artifacts/eda/whiteboard/seen_split_<models>.json)
"""
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'artifacts/experiments/20261012_oracle'))
import audit_s5_direct as A5  # noqa: E402


def main():
    import direct_coherence as C
    torch.set_num_threads(8)
    panel = torch.load(A5.BASE / 'panel.pt', map_location='cpu', weights_only=False, mmap=True)
    bank = torch.load(A5.BASE / 'snap_bank_readers.pt', map_location='cpu', weights_only=False)
    head = torch.load(A5.BASE / 'learned_innovation/eval_u3000_h01/mob_heads.pt', map_location='cpu', weights_only=False)['heads']['real']
    selected = sorted(json.loads((A5.BASE / 'direct_generator/data/existing_eval_selection.json').read_text())['selected_indices'])
    scroll = A5.load_scroll(); shifts = torch.as_tensor(scroll.SHIFTS)
    tok = panel['tokens'][selected].float()
    real_shift = shifts[scroll.estimate(tok[:, 15:31], tok[:, 16:32])]                   # [Rb,16,2]
    rd = C.decode(tok[:, 16:32], bank, head)
    rmob, rtile = rd['mob']['real'][..., :3], rd['tile']
    res = {}
    for name in sys.argv[1:]:
        imag = torch.load(A5.BASE / f'transport/recursion_samples_{name}.pt', weights_only=False)[selected].float()
        Rb, K = imag.shape[:2]
        d = C.decode(imag, bank, head)
        imob, itile = d['mob']['real'][..., :3], d['tile']
        prev = torch.cat([tok[:, 15][:, None, None].expand(-1, K, 1, -1, -1), imag[:, :, :-1]], 2)
        cam_ok = (shifts[scroll.estimate(prev, imag)] == real_shift[:, None]).all(-1).cumprod(2).bool()   # [Rb,K,16]
        tally = {'stayed': torch.zeros(16, 2), 'reentered': torch.zeros(16, 2)}
        for i in range(Rb):
            cum = real_shift[i].cumsum(0)
            for k in range(16):
                ok = cam_ok[i, :, k]
                if not ok.any():
                    continue
                for x in range(63):
                    if x == 31:
                        continue
                    r, c = divmod(x, 9)
                    wr, wc = r + int(cum[k, 0]), c + int(cum[k, 1])                       # world position (root frame)
                    if not (0 <= wr < 7 and 0 <= wc < 9):
                        continue                                                           # not seen at the root
                    left = any(not (0 <= wr - int(cum[j, 0]) < 7 and 0 <= wc - int(cum[j, 1]) < 9) for j in range(k))
                    nomob = ~(imob[i, ok, k, x].any(-1) | rmob[i, k, x].any())
                    bad = (itile[i, ok, k, x] != rtile[i, k, x]) & nomob
                    key = 'reentered' if left else 'stayed'
                    tally[key][k, 0] += bad.float().sum(); tally[key][k, 1] += nomob.float().sum()
        out = {}
        for key, t in tally.items():
            tot = t.sum(0)
            out[key] = {'cells': int(tot[1]), 'tile_error_all_depths': round(float(tot[0] / tot[1].clamp_min(1)), 4),
                        'tile_error_d16': round(float(t[15, 0] / t[15, 1].clamp_min(1)), 4), 'cells_d16': int(t[15, 1])}
        res[name] = out
        print(json.dumps({name: out}), flush=True)
    (ROOT / f'artifacts/eda/whiteboard/seen_split_{"_".join(sys.argv[1:])}.json').write_text(json.dumps(res, indent=1))


if __name__ == '__main__':
    main()
