"""Positive controls and the layer-wise phase probe for the health diagnosis (2026-10-08, diagnosis 6). Same subset, episode
split, inner split and probe (health_evidence.weighted_probe, uniform weights, held AUC) as health_evidence.py.
  adjacency   is the post-move adjacency (strata.pt adj_post) recoverable from the RAW input the world sees at the output
              position: frame-14 tokens of the 13 cells within Manhattan 2 of the player + the action (one-hot), with and without
              the TRUE scroll 14 -> 15; vs M16 s7's h63. All events, scroll steps, scroll steps with a zombie within 2.
  layers      the cooldown phase (time since the last health drop: = 6, 1..4, >= 7) at slot 63 of M16 s7, output position, after
              the input embedding and after each of the 6 factored layers (residual stream, forward hooks on world.layers).
  false       (`health_controls.py false`, CPU) where the emitted false drops sit: held unchanged events by context (zombie
              distance in frame 14, adj_post, time since the last drop, current health), for M16 s7 / s8 and E20 A
              -> false_drops.json
  pool        (`health_controls.py pool`) the adjacency control at 5x the data: every event of the E20 pool (71,918; frames t-14..t+1)
              with M16 s7's h63 at the output position (window 15), scroll 14 -> 15 (scroll.estimate), adj_post from Probes.zombie
              > 0.3 on frame 14 beside the post-move cell, zombie within 2 of the player in frame 14; probes as `adjacency`, fit /
              held by episode (health_chain.fit_split) -> pool_adjacency.json (h63 / labels cached in pool_h63.pt)
Usage: health_controls.py [false | pool]  -> controls.json | false_drops.json | pool_adjacency.json in artifacts/eda/health_chain_v1
"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
ROOT = HERE.parents[2]
OUT = ROOT / 'artifacts/eda/health_chain_v1'
M16 = ROOT / 'artifacts/eda/levers_tworlds_v1/corrt_rawlong_teacher_s7_fmamba_L16b40_from36000.pt'


def main():
    import teval as T
    import health_evidence as HE
    from d4mj.config import config_from_dict
    from d4mj.train import autocast_context
    import spatial as S
    sub = torch.load(OUT / 'subset.pt', weights_only=False)
    st = torch.load(OUT / 'strata.pt', weights_only=False)
    mm = np.memmap(OUT / 'subset.f16', dtype=np.float16, mode='r', shape=tuple(sub['shape']))
    N = len(sub['classes']); allm = torch.ones(N, dtype=torch.bool)
    fit = sub['fit']
    inner = fit & torch.tensor([int(hashlib.sha256((e + '/inner').encode()).hexdigest(), 16) % 4 == 0 for e, _ in sub['ids']])
    held = ~fit
    probe = lambda X, y: HE.weighted_probe(X, y, torch.ones(N), allm & fit & ~inner, allm & inner, steps=2000)
    res = {'adjacency': {}, 'layers': {}}
    # adjacency positive control
    y = st['adj_post']; scroll = sub['scroll'] != 0; near2 = sub['zombie_distance'] <= 2
    cells = [r * 9 + c for r in range(7) for c in range(9) if abs(r - 3) + abs(c - 4) <= 2]
    raw = torch.from_numpy(np.array(mm[:, 14][:, cells])).float().flatten(1)
    act = F.one_hot(sub['actions'][:, 14].clamp(max=16), 17).float()
    h63 = torch.load(OUT / 'corrt_rawlong_teacher_s7_fmamba_L16b40_from36000__w15__ext.pt', weights_only=False)['h_ext'][:, 0].float()
    for name, X in (('raw_cells13_action', torch.cat([raw, act], 1)),
                    ('raw_cells13_action_true_scroll', torch.cat([raw, act, F.one_hot(sub['scroll'], 5).float()], 1)),
                    ('m16_s7_h63', h63)):
        pr = probe(X, y)
        res['adjacency'][name] = {'all': round(HE.auc(pr[held], y[held]), 4), 'scroll': round(HE.auc(pr[held & scroll], y[held & scroll]), 4),
                                  'scroll_zombie_within_2': round(HE.auc(pr[held & scroll & near2], y[held & scroll & near2]), 4)}
        print(json.dumps({name: res['adjacency'][name]}), flush=True)
    # layer-wise phase at slot 63
    config = config_from_dict(torch.load(S.CHECKPOINT, map_location='cpu', weights_only=False)['config'])
    w, _ = T.load_world(M16, torch.device('cuda'))
    acts = {}
    hooks = [layer.register_forward_hook(lambda m, i, o, k=k: acts.__setitem__(k, o)) for k, layer in enumerate(w.layers)]
    feats = {k: [] for k in range(-1, 6)}
    with torch.no_grad():
        for i in range(0, N, 8):
            s = torch.from_numpy(np.array(mm[i:i + 8, 0:15])).float().cuda(); a = sub['actions'][i:i + 8, 0:15].cuda()
            with autocast_context(config):
                x0 = w.inputs(s, a)                                  # [b, T, 82, D], the action token first
                w.backbone_full(s, a)
            feats[-1].append(x0[:, -1, 64].float().cpu())            # slot 63 = index 64
            for k in range(6):
                feats[k].append(acts[k][:, -1, 64].float().cpu())
    for hk in hooks:
        hk.remove()
    since = st['since']
    for k in range(-1, 6):
        X = torch.cat(feats[k]); name = 'input_embedding' if k < 0 else f'layer_{k}'
        res['layers'][name] = {nm: round(HE.auc(probe(X, yy)[held], yy[held]), 4) for nm, yy in
                               (('since_eq6', since == 6), ('since_1to4', (since >= 1) & (since <= 4)), ('since_ge7', since >= 7))}
        print(json.dumps({name: res['layers'][name]}), flush=True)
    (OUT / 'controls.json').write_text(json.dumps(res, indent=1))


def false_drops():
    import teval as T
    sub = torch.load(OUT / 'subset.pt', weights_only=False)
    st = torch.load(OUT / 'strata.pt', weights_only=False)
    mm = np.memmap(OUT / 'subset.f16', dtype=np.float16, mode='r', shape=tuple(sub['shape']))
    meta, train_roots, train_seeds = T.split()
    P = T.Probes(T.build_cache('raw', torch.device('cpu')), meta, train_roots, train_seeds)
    read = lambda hud: (P.hud(hud.float().flatten(-2))[..., 0] * 9).float()
    N = len(sub['classes']); u = (sub['classes'] == 2) & ~sub['fit']
    cur = torch.cat([read(torch.from_numpy(np.array(mm[i:i + 2048, 14, 63:81]))) for i in range(0, N, 2048)])
    zd, since, health = sub['zombie_distance'], st['since'], sub['health'][:, 14]
    contexts = {'all': torch.ones(N, dtype=torch.bool), 'zombie_within_1': zd <= 1, 'zombie_at_2': zd == 2, 'no_zombie_within_2': zd > 2,
                'adj_post': st['adj_post'], 'since_1to5': (since >= 1) & (since <= 5), 'no_drop_in_window': since == 0,
                'health_9': health >= 9, 'health_le_3': health <= 3}
    res = {}
    for name in ('corrt_rawlong_teacher_s7_fmamba_L16b40_from36000__w15', 'corrt_rawlong_teacher_s8_fmamba_L16b40_from36000__w15',
                 'e20_A_s7_fmamba_fromM16__w15__ext'):
        em = read(torch.load(OUT / f'{name}.pt', weights_only=False)['hud']) < cur - 1.5
        res[name] = {k: [int((em & u & m).sum()), int((u & m).sum())] for k, m in contexts.items()}
        print(json.dumps({name: res[name]}), flush=True)
    (OUT / 'false_drops.json').write_text(json.dumps(res, indent=1))


def pool_adjacency():
    import teval as T
    import health_evidence as HE
    from health_chain import POOL, fit_split
    from scroll import estimate
    from d4mj.config import config_from_dict
    from d4mj.train import autocast_context
    import spatial as S
    lab = torch.load(POOL / 'labels.pt', weights_only=False)
    N = len(lab['classes'])
    tok = np.memmap(POOL / 'tokens.f16', dtype=np.float16, mode='r', shape=(N, 16, 81, 192))
    cache = OUT / 'pool_h63.pt'
    if not cache.exists():
        meta, train_roots, train_seeds = T.split()
        P = T.Probes(T.build_cache('raw', torch.device('cpu')), meta, train_roots, train_seeds)
        config = config_from_dict(torch.load(S.CHECKPOINT, map_location='cpu', weights_only=False)['config'])
        w, _ = T.load_world(M16, torch.device('cuda'))
        coords = torch.tensor([[c // 9, c % 9] for c in range(63)])
        shifts = torch.tensor(((0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)))
        d_now = (coords - torch.tensor([3, 4])).abs().sum(-1)
        h63, scroll, adj, near2 = [], [], [], []
        with torch.no_grad():
            for i in range(0, N, 8):
                x = torch.from_numpy(np.array(tok[i:i + 8])).float()
                with autocast_context(config):
                    h = w(x[:, :15].cuda(), lab['actions'][i:i + 8, :15].cuda())[1]
                h63.append(h[:, -1, 63].half().cpu())
                sc = estimate(x[:, 14], x[:, 15])
                z = P.zombie(x[:, 14, :63].flatten(0, 1))[:, 0].view(len(x), 63) > 0.3
                d_post = (coords[None] - (torch.tensor([3, 4])[None] + shifts[sc])[:, None]).abs().sum(-1)
                scroll.append(sc); adj.append((z & (d_post == 1)).any(1)); near2.append((z & (d_now <= 2)[None]).any(1))
        torch.save({'h63': torch.cat(h63), 'scroll': torch.cat(scroll), 'adj_post': torch.cat(adj), 'zombie_within_2': torch.cat(near2)}, cache)
    c = torch.load(cache, weights_only=False)
    fit = fit_split(lab['ids'])
    inner = fit & torch.tensor([int(hashlib.sha256((e + '/inner').encode()).hexdigest(), 16) % 4 == 0 for e, _ in lab['ids']])
    held = ~fit; allm = torch.ones(N, dtype=torch.bool)
    y, sc, near2 = c['adj_post'], c['scroll'] != 0, c['zombie_within_2']
    cells = [r * 9 + col for r in range(7) for col in range(9) if abs(r - 3) + abs(col - 4) <= 2]
    raw = torch.cat([torch.from_numpy(np.array(tok[i:i + 4096, 14][:, cells])).float().flatten(1).half() for i in range(0, N, 4096)])
    act = F.one_hot(lab['actions'][:, 14].clamp(max=16), 17).float()
    res = {'events': N, 'held': int(held.sum()), 'adj_post_rate': float(y.float().mean()),
           'held_scroll_zombie_within_2': [int((held & sc & near2 & y).sum()), int((held & sc & near2).sum())]}
    for name, X in (('raw_cells13_action', torch.cat([raw.float(), act], 1)),
                    ('raw_cells13_action_true_scroll', torch.cat([raw.float(), act, F.one_hot(c['scroll'], 5).float()], 1)),
                    ('m16_s7_h63', c['h63'].float())):
        pr = HE.weighted_probe(X, y, torch.ones(N), allm & fit & ~inner, allm & inner, steps=4000)
        res[name] = {'all': round(HE.auc(pr[held], y[held]), 4), 'scroll': round(HE.auc(pr[held & sc], y[held & sc]), 4),
                     'scroll_zombie_within_2': round(HE.auc(pr[held & sc & near2], y[held & sc & near2]), 4)}
        print(json.dumps({name: res[name]}), flush=True)
    (OUT / 'pool_adjacency.json').write_text(json.dumps(res, indent=1))


if __name__ == '__main__':
    sys.path.insert(0, str(ROOT / 'artifacts/experiments/20260926_diagnosis'))
    sys.path.insert(0, str(ROOT / 'artifacts/experiments/20260921_readout_ladder'))
    {'false': false_drops, 'pool': pool_adjacency}.get(sys.argv[1] if sys.argv[1:] else '', main)()
