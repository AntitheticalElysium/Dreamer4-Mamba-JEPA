"""Audit (10-14), stage 3: where do the generated-prefix arm's extra self-fed false cows sit? Saved self-fed predictions
(generated_prefix/eval_u3000/pred_{teacher,prefix}.pt) on the 200 DEV roots, depths 9-16, frozen AllProbes cow reader. False cow
cells split by the step's camera alignment (the saved per-step `aligned` mask: imagined cumulative scroll == recorded) and by whether
the cell had a decoded cow in the arm's OWN previous generated frame (copied residue vs new). Also the routing weights (corrt's six
candidates: self, 4 neighbours, generator) at false cells vs true-positive cells.
Usage: audit_s3_alignment.py -> stdout (recorded in NOTEBOOK)"""
import json, sys
from pathlib import Path
import torch
ROOT = Path(__file__).resolve().parents[3]
for d in ('20260927_levers', '20260926_diagnosis', '20260921_readout_ladder', '20261012_oracle'):
    sys.path.insert(0, str(ROOT / 'artifacts/experiments' / d))
import teval as T, scoreboard as SB, oracle_eval as OE
OUT = OE.OUT
panel = torch.load(OUT / 'panel.pt', weights_only=False); lay = panel['layouts']; R = len(lay)
meta, tr, ts = T.split(); cache = T.build_cache('raw', torch.device('cpu'))
probes = SB.AllProbes(T, cache, meta, tr, ts); thr = float(probes.thr[OE.PROBE_CH['cow']])
truth = lay[..., OE.LAYOUT_CH['cow']] > 0; truth[:, :, 31] = False; tru = truth[:, 16:32]
for arm in ('teacher', 'prefix'):
    p = torch.load(OUT / f'generated_prefix/eval_u3000/pred_{arm}.pt', map_location='cpu', weights_only=False)
    gen, al, w = p['gen'], p['aligned'], p['gen_weights']
    sc = torch.full((R, 16, 63), -9.0)
    sc[:, :, OE.MAPC] = probes.mob(gen[:, :, OE.MAPC].flatten(0, 2).float()).view(R, 16, 62, 4)[..., OE.PROBE_CH['cow']]
    dec = sc > thr
    root_dec = torch.zeros(R, 63, dtype=torch.bool)
    root_dec[:, OE.MAPC] = probes.mob(panel['tokens'][:, 15, OE.MAPC].flatten(0, 1).float()).view(R, 62, 4)[..., OE.PROBE_CH['cow']] > thr
    prev = torch.cat([root_dec[:, None], dec[:, :-1]], 1)
    late = torch.zeros(R, 16, dtype=torch.bool); late[:, 8:] = True
    fp = dec & ~tru
    out = {}
    for nm, m in (('aligned', al & late), ('misaligned', ~al & late)):
        mm = m[..., None].expand_as(fp)
        out[nm] = {'steps': int(m.sum()), 'false_cows': int((fp & mm).sum()), 'per_step': round(float((fp & mm).sum()) / max(int(m.sum()), 1), 3),
                   'copied_from_own_previous': int((fp & mm & prev).sum()), 'true_positive': int((dec & tru & mm).sum()), 'true': int((tru & mm).sum())}
    # routing at false vs true-positive cells (late, all steps): mean weight on [self, 4 neighbours, generator]
    wl = w[:, :, :63][late]                                                     # [n,63,6]
    fpl, tpl = fp[late], (dec & tru)[late]
    out['routing_at_false_cells'] = [round(float(x), 3) for x in wl[fpl].mean(0)]
    out['routing_at_true_positive_cells'] = [round(float(x), 3) for x in wl[tpl].mean(0)]
    print(arm, json.dumps(out), flush=True)
