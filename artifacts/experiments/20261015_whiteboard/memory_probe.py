"""Track B (2026-10-15 evening): is the memory failure storage or readout? (inference + ridge probes; no training)

Training-ledger windows (16 frames, seeded sample), teacher-forced. For every transition t -> t+1 with a camera scroll, every
map cell entering the view at t+1 is classed REVISIT (its world position was inside one of the window's frames 0..t; camera
track from the parameter-free token scroll estimator) or NEW. Target: the frozen bank tile reader's class on the REAL frame
t+1 at that cell (no simulator labels).
  output_acc   the model's own prediction LN(b)[t] at that cell, read by the same tile reader
  probe_acc    a ridge probe from the backbone belief at that slot, belief[t, cell] (256-d; what the model's generated
               candidate proj(belief) reads), fitted on half the windows, tested on the other half
  probe_prev   control: the same probe from the cell's own token in frame t (the content that occupied the slot before the
               scroll; no memory)
probe_acc >> output_acc => the information is stored but not read out; probe_acc ~ output_acc ~ low => not stored/addressed.
REVISIT split by scrolls between the last frame that showed the cell and t.
Usage: memory_probe.py <checkpoint relative to transport/ | D_parent> -> artifacts/eda/whiteboard/memory_probe_<name>.json
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'artifacts/experiments/20261012_oracle'))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import audit_s5_direct as A5  # noqa: E402
import rep_layers as RL  # noqa: E402

OUT = ROOT / 'artifacts/eda/whiteboard'
N_WINDOWS = 2400


@torch.no_grad()
def main():
    import direct_coherence as C
    import direct_train as BASE
    import paired_model as PM
    import transport_model as TM
    from d4mj.train import autocast_context
    name = sys.argv[1]
    config = BASE.settings()
    _, windows, actions, tokens = BASE.load_data()
    sel = torch.randperm(len(windows), generator=torch.Generator().manual_seed(20261017))[:N_WINDOWS]
    if name == 'D_parent':
        world = PM.PairedWorld().cuda().eval()
        world.load_state_dict(torch.load(A5.BASE / 'paired_event/training/paired_u0000.pt', map_location='cpu', weights_only=False)['world'])
    else:
        world = TM.TransportWorld('operator').cuda().eval()
        world.load_state_dict(torch.load(A5.BASE / 'transport' / name, map_location='cpu', weights_only=False)['world'])
    bank = torch.load(A5.BASE / 'snap_bank_readers.pt', map_location='cpu', weights_only=False)
    head = torch.load(A5.BASE / 'learned_innovation/eval_u3000_h01/mob_heads.pt', map_location='cpu', weights_only=False)['heads']['real']
    scroll = A5.load_scroll(); shifts = torch.as_tensor(scroll.SHIFTS)
    feats, prevtok, tgt, outcls, cls, nscr, widx = [], [], [], [], [], [], []
    for b0 in range(0, N_WINDOWS, 8):
        frames, ac, _ = BASE.batch(tokens, actions, windows, sel[b0:b0 + 8])
        with autocast_context(config):
            belief, ha = world.backbone_full(frames, ac)
            pred = F.layer_norm(world.reference(frames, ac, belief, ha).float(), (192,))
        fr = frames.float().cpu(); belief = belief.float().cpu(); pred = pred.cpu()
        sh = shifts[scroll.estimate(fr[:, :-1], fr[:, 1:])]                # [b,15,2]
        real_tile = C.decode(fr[:, 1:], bank, head)['tile']                 # [b,15,63] reader classes on real successors
        pred_tile = C.decode(pred[:, :15], bank, head)['tile']
        for b in range(len(fr)):
            origin = torch.zeros(16, 2, dtype=torch.long)
            origin[1:] = sh[b].cumsum(0)
            for t in range(15):
                dr, dc = (int(v) for v in sh[b, t])
                if (dr, dc) == (0, 0):
                    continue
                o, op = origin[t + 1], origin[t]
                for r in range(7):
                    for c in range(9):
                        x = r * 9 + c
                        if x == 31:
                            continue
                        w = (r + int(o[0]), c + int(o[1]))
                        if 0 <= w[0] - int(op[0]) < 7 and 0 <= w[1] - int(op[1]) < 9:
                            continue                                       # visible at t: not entering
                        last = None
                        for f in range(t + 1):
                            of = origin[f]
                            if 0 <= w[0] - int(of[0]) < 7 and 0 <= w[1] - int(of[1]) < 9:
                                last = f
                        if real_tile[b, t, x] < 0:
                            continue
                        feats.append(belief[b, t, x]); prevtok.append(fr[b, t, x]); tgt.append(int(real_tile[b, t, x]))
                        outcls.append(int(pred_tile[b, t, x])); cls.append(last is not None)
                        nscr.append(int((sh[b, last:t] != 0).any(-1).sum()) if last is not None else -1)
                        widx.append(b0 + b)
        if b0 % 400 == 0:
            print(json.dumps({'windows_done': b0 + 8, 'cells': len(tgt)}), flush=True)
    X = torch.stack(feats); P = torch.stack(prevtok); y = torch.tensor(tgt); o = torch.tensor(outcls)
    rv = torch.tensor(cls); ns = torch.tensor(nscr); wi = torch.tensor(widx)
    train = wi % 2 == 0; test = ~train
    yoh = F.one_hot(y, 17).float()
    pb = RL.ridge(X, yoh, torch.where(train)[0], torch.where(test)[0]).argmax(-1)
    pp = RL.ridge(P, yoh, torch.where(train)[0], torch.where(test)[0]).argmax(-1)
    yt, ot, rvt, nst = y[test], o[test], rv[test], ns[test]
    res = {'model': name, 'entering_cells': len(y), 'test_cells': int(test.sum())}
    for key, m in (('REVISIT', rvt), ('NEW', ~rvt), ('REVISIT_scrolls0-2', rvt & (nst <= 2)), ('REVISIT_scrolls3+', rvt & (nst >= 3))):
        if m.sum() == 0:
            continue
        res[key] = {'cells': int(m.sum()), 'output_acc': round(float((ot[m] == yt[m]).float().mean()), 4),
                    'probe_belief_acc': round(float((pb[m] == yt[m]).float().mean()), 4),
                    'probe_prev_token_acc': round(float((pp[m] == yt[m]).float().mean()), 4),
                    'majority_class_acc': round(float((yt[m] == yt[m].mode().values).float().mean()), 4)}
    OUT.mkdir(parents=True, exist_ok=True)
    tag = name.replace('/', '_').replace('.pt', '')
    (OUT / f'memory_probe_{tag}.json').write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1), flush=True)


if __name__ == '__main__':
    main()
