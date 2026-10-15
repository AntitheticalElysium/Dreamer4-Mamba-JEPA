"""Where does the imagined census gap come from? (2026-10-15 night, CPU, saved recursion samples; no model runs)

Per bright root and mob type (zombie, cow), the decoded count at depth k (frozen real reader, all trajectories, as
transport_recursion's census) is split by whether the root view holds that type (panel layout at the root; evaluation only):
  present   the type is in the root view: the count is mostly the root's own mobs continuing (plus some arrivals)
  absent    the type is not in the root view: every decoded mob of that type is an arrival (walked in or spawned)
Census ratio per split = summed imagined count / summed real count; for 'absent', also the real arrival rate per frame.
If persistence is solved and the gap is arrivals, 'present' is near 1 and 'absent' near 0.
Usage: census_split.py <model> ... -> prints (artifacts/eda/whiteboard/census_split.json)
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
    root_has = panel['layouts'][selected, 15].bool()[..., :2].any(1)                    # [Rb,2] zombie, cow in root view
    real = C.decode(panel['tokens'][selected, 16:32].float(), bank, head)['mob']['real'][..., :2].sum(-2).float()   # [Rb,16,2]
    res = {}
    for name in sys.argv[1:]:
        imag = torch.load(A5.BASE / f'transport/recursion_samples_{name}.pt', weights_only=False)[selected].float()
        ic = C.decode(imag, bank, head)['mob']['real'][..., :2].sum(-2).float().mean(1)   # [Rb,16,2] mean over samples
        out = {}
        for t, kind in enumerate(('zombie', 'cow')):
            for split, m in (('present', root_has[:, t]), ('absent', ~root_has[:, t])):
                out[f'{kind}_{split}'] = {'roots': int(m.sum()),
                                          'census_d1_d8_d16': [round(float(ic[m, d, t].sum() / real[m, d, t].sum().clamp_min(1)), 3) for d in (0, 7, 15)],
                                          'real_per_frame_d1_d8_d16': [round(float(real[m, d, t].mean()), 3) for d in (0, 7, 15)],
                                          'imagined_per_frame_d1_d8_d16': [round(float(ic[m, d, t].mean()), 3) for d in (0, 7, 15)]}
        res[name] = out
        print(json.dumps({name: out}), flush=True)
    (ROOT / 'artifacts/eda/whiteboard/census_split.json').write_text(json.dumps(res, indent=1))


if __name__ == '__main__':
    main()
