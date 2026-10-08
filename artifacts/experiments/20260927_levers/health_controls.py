"""Positive controls and the layer-wise phase probe for the health diagnosis (2026-10-08, diagnosis 6). Same subset, episode
split, inner split and probe (health_evidence.weighted_probe, uniform weights, held AUC) as health_evidence.py.
  adjacency   is the post-move adjacency (strata.pt adj_post) recoverable from the RAW input the world sees at the output
              position: frame-14 tokens of the 13 cells within Manhattan 2 of the player + the action (one-hot), with and without
              the TRUE scroll 14 -> 15; vs M16 s7's h63. All events, scroll steps, scroll steps with a zombie within 2.
  layers      the cooldown phase (time since the last health drop: = 6, 1..4, >= 7) at slot 63 of M16 s7, output position, after
              the input embedding and after each of the 6 factored layers (residual stream, forward hooks on world.layers).
Usage: health_controls.py  -> artifacts/eda/health_chain_v1/controls.json
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


if __name__ == '__main__':
    sys.path.insert(0, str(ROOT / 'artifacts/experiments/20260926_diagnosis'))
    sys.path.insert(0, str(ROOT / 'artifacts/experiments/20260921_readout_ladder'))
    main()
