"""Decision-relevant imagination (2026-10-15, CPU, saved recursion samples; no model runs).

Do self-fed imagined trajectories carry the events a decision depends on? Frozen readers on imagined and real frames alike
(real mob reader; bank HUD reader, health in units 0-9). Per root (200 DEV panel roots; recorded actions), per model:
  zombie_adj   a zombie decoded in one of the 4 cells adjacent to the player (the damage hazard)
  cow_adj      a cow adjacent (food)
  health_drop  decoded health falls by >= 0.5 units from the previous frame
Readings, bright / dark:
  rate_by_depth            mean event rate per depth, imagined (mean over samples) vs real
  auc_H (H = 4, 8, 16)     root-level discrimination: the imagined probability that the event occurs at least once within H
                           steps (share of samples; 0/1 for deterministic models) as a score for the real event (one real
                           future per root). Mann-Whitney AUC.
  health_bias_by_depth     mean imagined - real decoded health
Usage: decision_proxies.py <model> ... -> artifacts/eda/whiteboard/decision_<model>.json
"""
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'artifacts/experiments/20261012_oracle'))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import audit_s5_direct as A5  # noqa: E402
import rep_layers as RL  # noqa: E402

OUT = ROOT / 'artifacts/eda/whiteboard'
ADJ = [22, 30, 32, 40]


def hud_health(tokens, head):
    x = tokens[..., 63:81, :].flatten(-2).float()
    z = ((x - head['mu']) / head['sd']).double()
    return (torch.cat([z, torch.ones(*z.shape[:-1], 1, dtype=torch.float64)], -1) @ head['w'].double())[..., 0].float()


def events(tokens, bank, real_head):
    import direct_coherence as C
    d = C.decode(tokens, bank, real_head)['mob']['real']                  # [...,63,4] zombie, cow, skeleton, arrow
    h = hud_health(tokens, bank['heads']['hud'])
    return d[..., ADJ, 0].any(-1), d[..., ADJ, 1].any(-1), h


def main():
    torch.set_num_threads(8)
    panel = torch.load(A5.BASE / 'panel.pt', map_location='cpu', weights_only=False, mmap=True)
    bank = torch.load(A5.BASE / 'snap_bank_readers.pt', map_location='cpu', weights_only=False)
    head = torch.load(A5.BASE / 'learned_innovation/eval_u3000_h01/mob_heads.pt', map_location='cpu', weights_only=False)['heads']['real']
    selected = set(json.loads((A5.BASE / 'direct_generator/data/existing_eval_selection.json').read_text())['selected_indices'])
    real = panel['tokens'][:, 15:32].float()                               # root + 16 real successors
    rz, rc, rh = events(real, bank, head)                                  # [R,17]
    real_ev = {'zombie_adj': rz[:, 1:], 'cow_adj': rc[:, 1:], 'health_drop': (rh[:, 1:] - rh[:, :-1]) <= -.5}
    R = len(real)
    parts = {'bright': torch.tensor([i in selected for i in range(R)]), 'dark': torch.tensor([i not in selected for i in range(R)])}
    for name in sys.argv[1:]:
        imag = torch.load(A5.BASE / f'transport/recursion_samples_{name}.pt', weights_only=False)   # [R,K,16,81,192]
        K = imag.shape[1]
        iz, ic, ih = [], [], []
        for i in range(R):
            z_, c_, h_ = events(imag[i].float(), bank, head)
            iz.append(z_); ic.append(c_); ih.append(h_)
        iz, ic, ih = torch.stack(iz), torch.stack(ic), torch.stack(ih)    # [R,K,16]
        prev_h = torch.cat([rh[:, None, :1].expand(-1, K, 1), ih[:, :, :-1]], 2)
        img_ev = {'zombie_adj': iz, 'cow_adj': ic, 'health_drop': (ih - prev_h) <= -.5}
        res = {'model': name, 'samples_per_root': K}
        for part, m in parts.items():
            out = {}
            for ev in img_ev:
                im, re = img_ev[ev][m].float(), real_ev[ev][m].float()
                out[ev] = {'rate_by_depth_imagined': im.mean((0, 1)).round(decimals=3).tolist(), 'rate_by_depth_real': re.mean(0).round(decimals=3).tolist()}
                for H in (4, 8, 16):
                    score = im[:, :, :H].amax(-1).mean(1)                   # imagined P(event within H)
                    label = re[:, :H].amax(-1) > 0
                    out[ev][f'auc_{H}'] = round(RL.auc(score, label), 4) if 0 < int(label.sum()) < len(label) else None
                    out[ev][f'real_rate_within_{H}'] = round(float(label.float().mean()), 3)
                    out[ev][f'imagined_rate_within_{H}'] = round(float(score.mean()), 3)
            out['health_bias_by_depth'] = (ih[m].mean(1) - rh[m][:, 1:]).mean(0).round(decimals=3).tolist()
            res[part] = out
        OUT.mkdir(parents=True, exist_ok=True)
        (OUT / f'decision_{name}.json').write_text(json.dumps(res, indent=1))
        b = res['bright']
        print(json.dumps({'model': name, **{ev: {k: b[ev][k] for k in ('auc_4', 'auc_8', 'auc_16', 'real_rate_within_16', 'imagined_rate_within_16')} for ev in img_ev},
                          'health_bias_d1_d8_d16': [b['health_bias_by_depth'][d] for d in (0, 7, 15)]}), flush=True)


if __name__ == '__main__':
    main()
