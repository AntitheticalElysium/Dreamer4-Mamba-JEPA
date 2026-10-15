"""R29 localization (2026-10-15 night, GPU inference, no training): is PI_INNOV's content head calibrated on REAL contexts?

Every K 4096 centroid is classed once by the frozen readers (real mob reader: zombie / cow / skeleton; else the bank tile
class). Teacher-forced on seeded training windows NOT used to fit the head (seed 20261022), at PI's training-time innovation
cells (pi_innov_train.innovation_cells), split ENTERING / EDIT (gate posterior, non-entering) / HUD:
  expected   sum over cells of the head's probability mass on centroids of class c
  actual     number of cells whose real pre-chance target z_pre reads as class c
  argmax     number of cells whose most probable centroid reads as class c (what a sharp head would draw)
Ratios expected / actual near 1 = calibrated content; then R29's overshoot comes from recursion (imagined contexts or where
sampling is applied), not from the head. Also the head's cross-entropy and entropy on these held-out windows.
Usage: innov_calib.py -> artifacts/eda/whiteboard/innov_calib.json
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
from torch.nn import functional as F

import audit_s5_direct as A5
import direct_coherence as C
import direct_model as DM
import direct_train as BASE
import pi_model as PI
import pi_innov_train as PT

OUT = HERE.parents[1] / 'eda/whiteboard'
N_WINDOWS = 320
CLASSES = ('zombie', 'cow', 'skeleton')


def classify(tokens, bank, head):
    """[N,192] LN tokens -> class id: 0-2 mob types, 3 + tile class otherwise (read at an interior map cell)"""
    n = len(tokens)
    frame = torch.zeros(n, 81, 192)
    frame[:, 20] = tokens                                                   # an interior map cell
    d = C.decode(frame, bank, head)
    mob = d['mob']['real'][:, 20, :3]
    cls = 3 + d['tile'][:, 20].clamp_min(0)
    has = mob.any(-1)
    cls[has] = mob[has].float().argmax(-1)
    return cls


@torch.no_grad()
def main():
    config = BASE.settings()
    _, windows, actions, tokens = BASE.load_data()
    sel = torch.randperm(len(windows), generator=torch.Generator().manual_seed(20261022))[:N_WINDOWS]
    bank = torch.load(A5.BASE / 'snap_bank_readers.pt', map_location='cpu', weights_only=False)
    rhead = torch.load(A5.BASE / 'learned_innovation/eval_u3000_h01/mob_heads.pt', map_location='cpu', weights_only=False)['heads']['real']
    codes = torch.load(PT.CODES, map_location='cpu', weights_only=False)['codes'].float()
    code_cls = classify(codes, bank, rhead)                                 # [4096]
    ncls = int(code_cls.max()) + 1
    onehot = F.one_hot(code_cls, ncls).float().cuda()                       # [4096, ncls]
    codes = codes.cuda()
    scroll = A5.load_scroll()
    res = {'code_classes': {('zombie', 'cow', 'skeleton')[c] if c < 3 else f'tile{c - 3}': int((code_cls == c).sum()) for c in range(ncls) if (code_cls == c).any()}}
    for s in (1, 2):
        w = PI.PIWorld().cuda().eval()
        w.load_state_dict(torch.load(PT.OUT / f'pi_seed{s}/pi_u3000.pt', map_location='cpu', weights_only=False)['world'])
        head = torch.nn.Linear(256, 4096).cuda()
        head.load_state_dict(torch.load(PT.OUT / f'pi_seed{s}/innov_u1500.pt', map_location='cpu', weights_only=False)['head'])
        g = torch.Generator().manual_seed(7)
        acc = {k: {'expected': torch.zeros(ncls), 'actual': torch.zeros(ncls), 'argmax': torch.zeros(ncls), 'cells': 0, 'ce': 0.0, 'entropy': 0.0}
               for k in ('entering', 'edit', 'hud')}
        for b0 in range(0, N_WINDOWS, 8):
            frames, ac, _ = BASE.batch(tokens, actions, windows, sel[b0:b0 + 8])
            noise = DM.draw_noise(frames.shape[:2], g).cuda()
            belief, innov, z_pre = PT.innovation_cells(w, frames, ac, noise[0], config, scroll)
            est = scroll.estimate(frames[:, :-1].float(), frames[:, 1:].float())
            enter = w.entering[est]                                         # [B,15,81]
            cell = torch.arange(81, device=frames.device).expand_as(innov)
            for k, m in (('entering', innov & enter), ('edit', innov & ~enter & (cell < PI.MAP)), ('hud', innov & (cell >= PI.MAP))):
                if not bool(m.any()):
                    continue
                lg = head(belief[m]); p = lg.softmax(-1)
                target = torch.cdist(z_pre[m], codes).argmin(-1)
                actual_cls = classify(z_pre[m].cpu(), bank, rhead)
                a = acc[k]
                a['expected'] += (p @ onehot).sum(0).cpu()
                a['actual'] += torch.bincount(actual_cls, minlength=ncls)[:ncls].float()
                a['argmax'] += torch.bincount(code_cls[lg.argmax(-1).cpu()], minlength=ncls)[:ncls].float()
                a['cells'] += int(m.sum()); a['ce'] += float(F.cross_entropy(lg, target, reduction='sum'))
                a['entropy'] += float(-(p * p.clamp_min(1e-12).log()).sum())
        out = {}
        for k, a in acc.items():
            n = max(a['cells'], 1)
            rows = {}
            for c in range(ncls):
                if a['actual'][c] + a['expected'][c] < 5:
                    continue
                name = CLASSES[c] if c < 3 else f'tile{c - 3}'
                rows[name] = {'actual': int(a['actual'][c]), 'expected': round(float(a['expected'][c]), 1), 'argmax': int(a['argmax'][c]),
                              'expected_over_actual': round(float(a['expected'][c] / a['actual'][c].clamp_min(1)), 3)}
            out[k] = {'cells': a['cells'], 'ce': round(a['ce'] / n, 3), 'entropy': round(a['entropy'] / n, 3), 'classes': rows}
        res[f'seed{s}'] = out
        print(json.dumps({f'seed{s}': out}), flush=True)
        del w; torch.cuda.empty_cache()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'innov_calib.json').write_text(json.dumps(res, indent=1))


if __name__ == '__main__':
    main()
