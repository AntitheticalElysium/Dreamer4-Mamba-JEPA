"""Health failure, link by link (2026-10-08). Where does an ordinary hit stop being predicted?

Data: the E20 factual TRAIN pool (artifacts/eda/levers_e20_v1/pool: 71,918 events, 16 true frames t-14..t+1, actions t-14..t,
health labels; target = transition 14 -> 15). A fixed subset (seed 20261008): all 1,005 fresh ordinary hits (E20's definition:
a zombie in frame 14 sits beside the player's POST-move cell, none beside in frames 11-14, no damage in 11-13; 1,005 of the
1,013 fresh events are hits), 5,000 other ordinary hits (living, dh <= -2), 8,000 unchanged (dh = 0), 1,000 deaths.
Episodes split fit / held by episode id (fit = sha256(id) % 5 != 0); every probe / head is fitted on fit, scored on held.
Per event (true frames): scroll 14 -> 15 (scroll.estimate), nearest zombie distance to the player in frame 14 (teval's
Probes.zombie > 0.3 on cells within Manhattan 2 of token 31).

Stage `extract`: the subset's tokens / labels -> artifacts/eda/health_chain_v1/subset.{f16,pt}.
Stage `world <ckpt> [--window W]`: one frozen world, teacher-forced on its trained context (W = time rows - 1), predicting frame 15
from frames 15-W..14. Saves, at that output position: the emitted HUD (18 tokens), the corr mixture weights and generated
candidate at token 63, backbone h at tokens 63, 31 and the player's 4 neighbours, the emitted tokens of those 4 neighbours,
and the scroll estimated between true frame 14 and the emitted frame 15.  -> artifacts/eda/health_chain_v1/<world>.pt
Readouts are computed by health_chain_read.py.
`--ext` (diagnosis 2, 2026-10-08) also saves h_ext: backbone h at 63, 31 and the 12 cells within Manhattan 2 of the player, plus
the action token's output (backbone_full; corr heads), [N, 15, 256] -> <world>__w<W>__ext.pt.
E21 worlds (`--event`) also save the event head's logit at token 63 (event63).
Usage: health_chain.py extract | health_chain.py world <world.pt> [--window W] [--ext]
"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
ROOT = HERE.parents[2]
POOL = ROOT / 'artifacts/eda/levers_e20_v1/pool'
OUT = ROOT / 'artifacts/eda/health_chain_v1'
SEED = 20261008
NEAR = [22, 30, 32, 40]                                   # up, left, right, down of the player token 31


def fit_split(ids):
    return torch.tensor([int(hashlib.sha256(e.encode()).hexdigest(), 16) % 5 != 0 for e, _ in ids])


def extract():
    import teval as T
    from scroll import estimate
    OUT.mkdir(parents=True, exist_ok=True)
    lab = torch.load(POOL / 'labels.pt', weights_only=False)
    n = len(lab['classes'])
    tok = np.memmap(POOL / 'tokens.f16', dtype=np.float16, mode='r', shape=(n, 16, 81, 192))
    cls, fresh = lab['classes'], lab['fresh']
    g = torch.Generator().manual_seed(SEED)
    pick = lambda m, k: torch.where(m)[0][torch.randperm(int(m.sum()), generator=g)[:k]]
    rows = torch.cat([torch.where((cls == 1) & fresh)[0], pick((cls == 1) & ~fresh, 5000), pick(cls == 2, 8000),
                      pick(cls == 0, 1000)]).sort().values
    mm = np.memmap(OUT / 'subset.f16', dtype=np.float16, mode='w+', shape=(len(rows), 16, 81, 192))
    for i in range(0, len(rows), 256):
        mm[i:i + 256] = tok[rows[i:i + 256].numpy()]
    mm.flush()
    x = torch.from_numpy(np.asarray(mm))
    scroll = torch.cat([estimate(x[i:i + 512, 14].float(), x[i:i + 512, 15].float()) for i in range(0, len(rows), 512)])
    meta, train_roots, train_seeds = T.split()
    probes = T.Probes(T.build_cache('raw', torch.device('cpu')), meta, train_roots, train_seeds)
    cells = torch.tensor([r * 9 + c for r in range(7) for c in range(9) if 0 < abs(r - 3) + abs(c - 4) <= 2])
    dist = torch.tensor([abs(int(c) // 9 - 3) + abs(int(c) % 9 - 4) for c in cells])
    zdist = torch.full((len(rows),), 9)
    for i in range(0, len(rows), 512):
        z = probes.zombie(x[i:i + 512, 14, cells].float().flatten(0, 1))[:, 0].view(-1, len(cells)) > 0.3
        zdist[i:i + 512] = torch.where(z, dist[None], torch.tensor(9)).amin(1)
    ids = [lab['ids'][i] for i in rows.tolist()]
    sub = {'rows': rows, 'ids': ids, 'classes': cls[rows], 'fresh': fresh[rows], 'dh': lab['dh'][rows],
           'health': lab['health'][rows], 'actions': lab['actions'][rows], 'fit': fit_split(ids),
           'scroll': scroll, 'zombie_distance': zdist, 'seed': SEED, 'shape': list(mm.shape)}
    torch.save(sub, OUT / 'subset.pt')
    c = sub['classes']
    print(json.dumps({'events': len(rows), 'by_class': torch.bincount(c, minlength=4).tolist(),
                      'fresh_hits': int(((c == 1) & sub['fresh']).sum()), 'fit_share': float(sub['fit'].float().mean()),
                      'fresh_scroll_share': float((scroll[(c == 1) & sub['fresh']] != 0).float().mean()),
                      'hits_scroll_share': float((scroll[c == 1] != 0).float().mean())}), flush=True)


@torch.no_grad()
def world(path, window, ext=False):
    import teval as T
    from d4mj.config import config_from_dict
    from d4mj.train import autocast_context
    from scroll import estimate
    import spatial as S
    device = torch.device('cuda')
    config = config_from_dict(torch.load(S.CHECKPOINT, map_location='cpu', weights_only=False)['config'])
    sub = torch.load(OUT / 'subset.pt', weights_only=False)
    x = torch.from_numpy(np.asarray(np.memmap(OUT / 'subset.f16', dtype=np.float16, mode='r', shape=tuple(sub['shape']))))
    w, st = T.load_world(Path(path), device)
    W = window or w.time.shape[0] - 1
    bs = 8 if getattr(w, 'backbone_kind', 'full') in ('fmamba', 'fcanvas') else 16
    N = len(x)
    rec = {k: [] for k in ('hud', 'weights63', 'gen63', 'h', 'near_out', 'pred_scroll', 'h_ext', 'event63')}
    cells = [r * 9 + c for r in range(7) for c in range(9) if 0 < abs(r - 3) + abs(c - 4) <= 2]
    for i in range(0, N, bs):
        s = x[i:i + bs, 15 - W:15].float().to(device)
        a = sub['actions'][i:i + bs, 15 - W:15].to(device)
        with autocast_context(config):
            out, h, gen = w(s, a)
        out, h, gen = out[:, -1].float(), h[:, -1].float(), gen[:, -1].float()
        rec['hud'].append(out[:, 63:81].half().cpu())
        if hasattr(w, 'last_weights'):                                  # corr heads only (direct / residual: none)
            rec['weights63'].append(w.last_weights[:, -1, 63].float().cpu())
        rec['gen63'].append(gen[:, 63].half().cpu())
        rec['h'].append(h[:, [63, 31] + NEAR].half().cpu())
        rec['near_out'].append(out[:, NEAR].half().cpu())
        rec['pred_scroll'].append(estimate(s[:, -1].cpu(), out.cpu()))
        if hasattr(w, 'event_head'):                                    # E21: the event head's logit at token 63
            rec['event63'].append(w.event_head(h[:, 63]).float()[:, 0].cpu())
        if ext:
            with autocast_context(config):
                hb, ha = w.backbone_full(s, a)
            assert (hb[:, -1].float() - h).abs().max() < 1e-2           # same backbone pass as forward's h
            rec['h_ext'].append(torch.cat([hb[:, -1][:, [63, 31] + cells].float(), ha[:, -1, None].float()], 1).half().cpu())
    res = {k: torch.cat(v) for k, v in rec.items() if v}
    res.update({'world': st['name'], 'window': W, 'checkpoint': str(path)})
    torch.save(res, OUT / f"{st['name']}__w{W}{'__ext' if ext else ''}.pt")
    print(json.dumps({'world': st['name'], 'window': W, 'events': N}), flush=True)


if __name__ == '__main__':
    sys.path.insert(0, str(ROOT / 'artifacts/experiments/20260926_diagnosis'))
    sys.path.insert(0, str(ROOT / 'artifacts/experiments/20260921_readout_ladder'))
    args = sys.argv[1:]
    if args[0] == 'extract':
        extract()
    else:
        world(args[1], int(args[args.index('--window') + 1]) if '--window' in args else None, '--ext' in args)
