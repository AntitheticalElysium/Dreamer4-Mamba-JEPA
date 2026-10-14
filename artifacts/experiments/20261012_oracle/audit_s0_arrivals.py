"""Audit (10-14), stage 0: probe cow scores at arrival / source / stay cells, teacher-forced depths 1-16, for the four oracle-test arms
and the true tokens (reader: scoreboard.AllProbes; DEV panel). Usage: audit_s0_arrivals.py -> stdout (recorded in NOTEBOOK)."""
import sys, json, torch
from pathlib import Path
ROOT = Path('/home/antithetical/EPITA/PERSO/DynamicHorizons-Mamba-JEPA')
for d in ('20260927_levers', '20260926_diagnosis', '20260921_readout_ladder', '20261012_oracle'):
    sys.path.insert(0, str(ROOT / 'artifacts/experiments' / d))
import teval as T, scoreboard as SB, oracle_eval as OE
OUT = OE.OUT
panel = torch.load(OUT / 'panel.pt', weights_only=False)
lay = panel['layouts']; R = len(lay)
meta, tr, ts = T.split(); cache = T.build_cache('raw', torch.device('cpu'))
probes = SB.AllProbes(T, cache, meta, tr, ts); thr = float(probes.thr[OE.PROBE_CH['cow']])
truth = lay[..., OE.LAYOUT_CH['cow']] > 0; truth[:, :, 31] = False
tru, prv = truth[:, 16:32], truth[:, 15:31]
arrive, left, stay = tru & ~prv, prv & ~tru, tru & prv
res = {'threshold': thr}
for n in ('TRUE', 'control_l1', 'oracle_l1', 'control_deltairis', 'oracle_deltairis'):
    tf = panel['tokens'][:, 16:32].float() if n == 'TRUE' else torch.load(OUT / f'pred_{n}.pt', weights_only=False)['tf'].float()
    sc = torch.full((R, 16, 63), -9.0)
    s = probes.mob(tf[:, :, OE.MAPC].flatten(0, 2)).view(R, 16, 62, 4)[..., OE.PROBE_CH['cow']]
    sc[:, :, OE.MAPC] = s
    row = {}
    for nm, m in (('arrive', arrive), ('left', left), ('stay', stay), ('empty', ~tru & ~prv)):
        v = sc[m]
        row[nm] = {'n': int(m.sum()), 'mean': round(float(v.mean()), 3), 'q25': round(float(v.quantile(0.25)), 3), 'q75': round(float(v.quantile(0.75)), 3),
                   'above_thr': round(float((v > thr).float().mean()), 3), 'above_half_thr': round(float((v > thr / 2).float().mean()), 3)} if m.any() else None
    res[n] = row
    print(n, json.dumps(row))
