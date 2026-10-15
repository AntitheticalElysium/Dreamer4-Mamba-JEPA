"""Can a copied token be re-rendered from one global rendering summary? (2026-10-16, CPU; R40 follow-up; exact state used only to
select persisting cells and for probe labels)

R40: a stale imagined frame breaks the belief's readouts through the rendering of copied cells (within-frame inconsistency).
R15: rendering steps are not predictable in TIME. Question here: GIVEN that the frame's rendering moved, is the new token of a
persisting cell a learnable function of its old token and a frame-global summary of the move?
Data: diagnosis_futures_v1, 240 seeded roots (160 fit / 80 test) x 5 futures; pairs (t, t+m), m in {1, 4, 8}, with no camera
scroll in between and the cell's exact content unchanged throughout (persisting), at most 24 seeded cells per pair (memory: the
first run kept every cell for m = 1..8 and was OOM-killed at 15 GB); Raw tokens (what PI copies).
Global summary g = mean over the OTHER persisting map cells of (token_{t+m} - token_t) (leave-one-out; a frame-level rendering
state a world model could carry). Predictors of token_{t+m}:
  copy        token_t (PI)
  shift       token_t + g
  mlp         token_t + MLP([token_t, g]) (192*2 -> 512 -> 192), MSE, 4,000 steps, Adam 1e-3, fit on fit roots
Test: relative error |pred - token_{t+m}| / |token_t - token_{t+m}| (1 = no better than copying), split by event size, and the
frozen readers on predicted tokens (tile agreement, zombie / cow kept) for cells whose rendering changed.
Usage: relight_learn.py -> artifacts/eda/whiteboard/relight_learn.json
"""
import json
import os
import sys
from pathlib import Path

os.environ.setdefault('TRITON_F32_DEFAULT', 'ieee')
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / '20261012_oracle'))
sys.path.insert(0, str(HERE))

import numpy as np
import torch
import torch.nn.functional as F

import content_audit as CA
import rep_layers as RL


def main():
    torch.set_num_threads(16)
    import rollouts as Rr
    import audit_s5_direct as A5
    import direct_coherence as C
    files = sorted(Rr.STORE.glob('seed-*.pt'))
    roots = [r for f in files for r in torch.load(f, weights_only=False)]
    pick = torch.randperm(len(roots), generator=torch.Generator().manual_seed(20261029))[:240].tolist()
    fit_roots = set(pick[:160])
    enc = RL.encoder()
    bank = torch.load(A5.BASE / 'snap_bank_readers.pt', map_location='cpu', weights_only=False)
    head = torch.load(A5.BASE / 'learned_innovation/eval_u3000_h01/mob_heads.pt', map_location='cpu', weights_only=False)['heads']['real']
    rows = {'raw': []}                                                             # (old, new, g, fit?, m, cell)
    sub_g = torch.Generator().manual_seed(4)
    for ri in pick:
        r = roots[ri]
        for k in range(5):
            frames = torch.cat([r['context'][-1:], r['future_frames'][k]])
            vis = torch.cat([r['root_visible'][None], r['future_visible'][k]])
            tiles, mobs, _ = RL.facts(vis)
            content, raw = CA.represent(enc, frames.float() / 255.0)
            still = [RL.camera(tiles[t], tiles[t + 1]) == (0, 0) for t in range(16)]
            for t in range(16):
                for m in (1, 4, 8):
                    if t + m > 16 or not all(still[t:t + m]):
                        break
                    keep = torch.ones(63, dtype=torch.bool); keep[31] = False
                    for u in range(t, t + m + 1):
                        keep &= (tiles[u, :63] == tiles[t, :63]) & (mobs[u, :63] == mobs[t, :63]).all(-1)
                    cells = torch.where(keep)[0]
                    if len(cells) < 8:
                        continue
                    for name, z in (('raw', raw),):
                        old, new = z[t, cells].float(), z[t + m, cells].float()
                        d = new - old
                        g = (d.sum(0, keepdim=True) - d) / (len(cells) - 1)        # leave-one-out frame mean shift
                        sub = torch.randperm(len(cells), generator=sub_g)[:24]
                        rows[name].append((old[sub].half(), new[sub].half(), g[sub].half(), ri in fit_roots, torch.full((len(sub),), m), cells[sub]))
    res = {}
    for name, rs in rows.items():
        old = torch.cat([x[0] for x in rs]).float(); new = torch.cat([x[1] for x in rs]).float(); g = torch.cat([x[2] for x in rs]).float()
        fit = torch.cat([torch.full((len(x[0]),), x[3]) for x in rs]); ms = torch.cat([x[4] for x in rs]); cell = torch.cat([x[5] for x in rs])
        moved = (new - old).norm(dim=-1)
        ev = moved > 0.05 * float(moved.median() + 1e-6) + 1e-3                         # the cell's rendering changed
        torch.manual_seed(0)
        net = torch.nn.Sequential(torch.nn.Linear(384, 512), torch.nn.GELU(), torch.nn.Linear(512, 192))
        opt = torch.optim.Adam(net.parameters(), lr=1e-3)
        tr = torch.where(fit & ev)[0]
        gen = torch.Generator().manual_seed(1)
        for step in range(4000):
            b = tr[torch.randint(len(tr), (2048,), generator=gen)]
            pred = old[b] + net(torch.cat([old[b], g[b]], -1))
            loss = F.mse_loss(pred, new[b])
            opt.zero_grad(); loss.backward(); opt.step()
        te = ~fit & ev
        with torch.no_grad():
            preds = {'copy': old[te], 'shift': old[te] + g[te], 'mlp': old[te] + net(torch.cat([old[te], g[te]], -1))}
        base = (old[te] - new[te]).norm(dim=-1)
        out = {'test_changed_cells': int(te.sum()), 'fit_changed_cells': int(len(tr))}
        for nm, pr in preds.items():
            rel = (pr - new[te]).norm(dim=-1) / base.clamp_min(1e-6)
            out[f'{nm}_rel_err_median'] = round(float(rel.median()), 4)
            out[f'{nm}_rel_err_mean'] = round(float(rel.mean()), 4)
            for lo, hi, tag in ((1, 1, 'm1'), (4, 4, 'm4'), (8, 8, 'm8')):
                mk = (ms[te] >= lo) & (ms[te] <= hi)
                out[f'{nm}_rel_err_median_{tag}'] = round(float(rel[mk].median()), 4) if mk.any() else None
        if name == 'raw':                                                          # frozen readers on predicted tokens
            n = min(int(te.sum()), 200000)
            for nm, pr in list(preds.items()) + [('real', new[te])]:
                tcls, mob = [], []
                for i in range(0, n, 4096):                                         # chunked: one frame per cell is 12 GB at once
                    frame = torch.zeros(min(4096, n - i), 81, 192)
                    frame[:, 20] = pr[i:i + len(frame)]
                    dd = C.decode(frame, bank, head)
                    tcls.append(dd['tile'][:, 20]); mob.append(dd['mob']['real'][:, 20, :3])
                out[f'{nm}_tile_class'] = torch.cat(tcls); out[f'{nm}_mob'] = torch.cat(mob)
            for nm in ('copy', 'shift', 'mlp'):
                out[f'{nm}_tile_agreement'] = round(float((out[f'{nm}_tile_class'] == out['real_tile_class']).float().mean()), 4)
                rm = out['real_mob']
                out[f'{nm}_mob_agreement_on_mobs'] = round(float((out[f'{nm}_mob'] == rm)[rm.any(-1)].all(-1).float().mean()), 4) if rm.any() else None
            for nm in ('copy', 'shift', 'mlp', 'real'):
                out.pop(f'{nm}_tile_class'); out.pop(f'{nm}_mob')
        res[name] = out
        print(json.dumps({name: out}), flush=True)
    (HERE.parents[1] / 'eda/whiteboard/relight_learn.json').write_text(json.dumps(res, indent=1))


if __name__ == '__main__':
    main()
