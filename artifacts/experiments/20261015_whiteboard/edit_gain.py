"""Track B (2026-10-15 evening, CPU, inference only): where does the generated token beat the exact copy? (PI's edit evidence)

PI's gate is trained toward q = sigmoid(gain / s + logit p), gain = |copy - z| - |LN(gen) - z|, s = the frame's mean copy
residual. That is EM for the edit prior: the BCE fixed point is p = E[sigmoid(gain / s + logit p)] per context, so the gate
learns the edit rate the likelihood ratio exp(gain / s) supports. If the generated token beats the stale copy broadly, the
learned prior rises and PI regenerates by default (fading returns); if only at changed cells, persistence holds.
Measured at PI's initialization (the exact D parent, paired_u0000; gen = its proj(belief)), teacher-forced, on seeded
training-ledger windows; copy = exact copy under the label-free scroll estimate (PI's training target); chance ignored (z
taken as is: moved mobs count as changed). Cell classes from the frozen bank tile reader and real mob head on REAL frames:
  unchanged   reader tile class of the copy source at t == class at t+1, no mob at either   (persisting terrain)
  changed     tile class differs, no mob                                                    (edits: do / place / drift flips)
  mob_kept    a mob at the copy source at t and at the cell at t+1                          (persisting mobs, moved or not)
  mob_other   a mob at exactly one of the two
  hud_changed / hud_unchanged   the 18 HUD tokens (copied in place), split by the label-free change test |z - copy| > 2.5
              (tworld.EVENT_TAU, E21: the valley of the bimodal HUD change-norm distribution): is a rare consequence (health,
              food, inventory change) one the generated token already beats the copy on? If not, PI's gate never edits it and
              gen, trained only at edited cells, never learns it (a hard-EM trap)
Entering map cells (always generated) and the player cell are excluded. Per class: cells, share gain > 0, median gain / s,
share gain / s > 4 (fires at the gate's initial bias), and the EM fixed-point prior p* solved per class.
With a trained PI checkpoint (second argument, relative to transport/): the trained gen and gate. Per class, additionally,
the trained gate's mean probability sigmoid(gate), the share where MAP fires (gate > 0), and the share the posterior edits
(sigmoid(gain / s + gate) > 0.5, as in training); and the gate's AUC separating changed from unchanged cells (map, HUD) and
posterior edits from the rest. Separates 'gate calibrated but below 0.5' from 'gate learned nothing'.
Usage: edit_gain.py [windows] [checkpoint] -> artifacts/eda/whiteboard/edit_gain[_<checkpoint tag>].json
"""
import dataclasses
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
import direct_train as BASE
import pi_model as PI
import rep_layers as RL
from d4mj.mamba_recurrence import FunctionalMamba2

OUT = HERE.parents[1] / 'eda/whiteboard'
N_WINDOWS = 160


def em_prior(g, iters=200):
    """p* with p = mean(sigmoid(g + logit p)): the BCE fixed point of PI's gate in one context (EM for the edit prior)"""
    l = torch.tensor(-4.0)
    for _ in range(iters):
        p = torch.sigmoid(g + l).mean().clamp(1e-6, 1 - 1e-6)
        l = torch.log(p / (1 - p))
    return float(torch.sigmoid(l))


@torch.no_grad()
def main():
    torch.manual_seed(0)
    n_windows = int(sys.argv[1]) if len(sys.argv) > 1 else N_WINDOWS
    _, windows, actions, tokens = BASE.load_data()
    sel = torch.randperm(len(windows), generator=torch.Generator().manual_seed(20261018))[:n_windows]
    w = PI.PIWorld(init_seed=7)
    ckpt = sys.argv[2] if len(sys.argv) > 2 else None
    if ckpt is None:
        res = w.load_state_dict(torch.load(A5.BASE / 'paired_event/training/paired_u0000.pt', map_location='cpu', weights_only=False)['world'], strict=False)
        assert all(k.startswith(('completion.', 'gate.')) for k in res.missing_keys), res.missing_keys
    else:
        w.load_state_dict(torch.load(A5.BASE / 'transport' / ckpt, map_location='cpu', weights_only=False)['world'])
    for m in w.modules():
        if isinstance(m, FunctionalMamba2):
            m.settings = dataclasses.replace(m.settings, backend='reference')
    w = w.eval()
    bank = torch.load(A5.BASE / 'snap_bank_readers.pt', map_location='cpu', weights_only=False)
    head = torch.load(A5.BASE / 'learned_innovation/eval_u3000_h01/mob_heads.pt', map_location='cpu', weights_only=False)['heads']['real']
    scroll = A5.load_scroll()
    rows = {k: [] for k in ('unchanged', 'changed', 'mob_kept', 'mob_other', 'hud_changed', 'hud_unchanged')}
    gates = {k: [] for k in rows}
    agree, total = 0, 0
    for b0 in range(0, n_windows, 8):
        frames, ac, _ = BASE.batch(tokens, actions, windows, sel[b0:b0 + 8], device='cpu')
        est = scroll.estimate(frames[:, :-1], frames[:, 1:])
        belief, ha = w.backbone_full(frames, ac)
        copy, gen, glogit, enter, sl = w.afterstate_parts(frames, ac, belief, ha, scroll=F.pad(est, (0, 1)))
        copy, gen, glogit, enter, sl = (x[:, :15] for x in (copy, gen, glogit, enter, sl))
        agree += int((sl.argmax(-1) == est).sum()); total += est.numel()
        z = frames[:, 1:]
        res_copy = (F.layer_norm(copy, (192,)) - z).norm(dim=-1)
        gain = res_copy - (F.layer_norm(gen, (192,)) - z).norm(dim=-1)
        g = gain / res_copy[..., :PI.MAP].mean(-1, keepdim=True).clamp_min(1e-6)            # [b,15,81]
        dz = C.decode(z, bank, head); dc = C.decode(F.layer_norm(copy, (192,)), bank, head)
        mob_z = dz['mob']['real'].any(-1); mob_c = dc['mob']['real'].any(-1)               # [b,15,63]
        tz, tc = dz['tile'], dc['tile']
        keep = ~enter[..., :PI.MAP] & (tz >= 0)                                             # excludes entering + player cell
        gm = g[..., :PI.MAP]
        for name, m in (('unchanged', keep & ~mob_z & ~mob_c & (tz == tc)), ('changed', keep & ~mob_z & ~mob_c & (tz != tc)),
                        ('mob_kept', keep & mob_z & mob_c), ('mob_other', keep & (mob_z ^ mob_c))):
            rows[name].append(gm[m]); gates[name].append(glogit[..., :PI.MAP][m])
        hud_change = (copy[:, :, PI.MAP:] - z[:, :, PI.MAP:]).norm(dim=-1) > 2.5
        rows['hud_changed'].append(g[..., PI.MAP:][hud_change]); rows['hud_unchanged'].append(g[..., PI.MAP:][~hud_change])
        gates['hud_changed'].append(glogit[..., PI.MAP:][hud_change]); gates['hud_unchanged'].append(glogit[..., PI.MAP:][~hud_change])
        print(json.dumps({'windows_done': b0 + 8}), flush=True)
    out = {'windows': n_windows, 'parent': ckpt or 'paired_u0000 (PI init)', 'scroll_argmax_agrees_with_estimate': round(agree / total, 4)}
    for name, parts in rows.items():
        x = torch.cat(parts)
        if len(x) == 0:
            out[name] = {'cells': 0}; continue
        out[name] = {'cells': len(x), 'share_gain_pos': round(float((x > 0).float().mean()), 4),
                     'median_gain_over_s': round(float(x.median()), 3), 'share_fires_at_init_bias': round(float((x > 4).float().mean()), 4),
                     'em_fixed_point_prior': round(em_prior(x), 4)}
        if ckpt is not None:
            gl = torch.cat(gates[name])
            out[name].update({'gate_prob_mean': round(float(torch.sigmoid(gl).mean()), 4), 'map_fires': round(float((gl > 0).float().mean()), 4),
                              'posterior_edits': round(float((torch.sigmoid(x + gl) > .5).float().mean()), 4)})
    if ckpt is not None:
        cat = {k: (torch.cat(rows[k]), torch.cat(gates[k])) for k in rows}
        for a, b in (('changed', 'unchanged'), ('hud_changed', 'hud_unchanged')):
            sc = torch.cat([cat[a][1], cat[b][1]]); lab = torch.cat([torch.ones(len(cat[a][1])), torch.zeros(len(cat[b][1]))])
            out[f'gate_auc_{a}_vs_{b}'] = round(float(RL.auc(sc, lab)), 4)
        allg = torch.cat([v[0] for v in cat.values()]); alll = torch.cat([v[1] for v in cat.values()])
        out['gate_auc_posterior_edit_vs_rest'] = round(float(RL.auc(alll, torch.sigmoid(allg + alll) > .5)), 4)
        out['gate_prob_max'] = round(float(torch.sigmoid(alll).max()), 4)
    OUT.mkdir(parents=True, exist_ok=True)
    tag = '' if ckpt is None else '_' + ckpt.replace('/', '_').replace('.pt', '')
    (OUT / f'edit_gain{tag}.json').write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1), flush=True)


if __name__ == '__main__':
    main()
