"""10-10 PROGRAM Part II, B4 (prediction P-C): the anatomy of a world's training gradient by token category. GPU.
Held windows from the diagnosis futures with the simulator state of every frame: [root, factual future 1..L-1] (L = the world's
frames: 6 for the stage-1 worlds, 16 for L16), the factual actions, roots alive through the window (sample 0). Teacher-forced as in
training (tworld.rollout_losses' teacher term), autocast as training. Every target token is put in one of budget.CATS (root -> key-0
successor categories applied to every transition of the window: entering, own state changed, player sprite left, neighbour changed,
sleep changed, no state change, HUD changed / same, player token).
The objective L (`l1`: mean |e|; `deltairis`: mean e^2 + 0.1 mean |e| + 0.01 mean over frames of the frame's worst element e^2, the
worst element's term assigned to its token) is split EXACTLY into per-category terms L_c (sum_c L_c = L). Per category and batch, the
gradient g_c of L_c on the BACKBONE (every parameter except the corrt head's proj / choose / frame / target_gate) is taken.
Reported per category (over the batches): loss share; gradient norm ratio |g_c| / |g| (mean); projection share <g_c, g> / |g|^2
(mean; additive: sums to 1 over categories); consistency |sum_b g_c| / sum_b |g_c| (1 = the same direction in every batch,
~1 / sqrt(batches) = noise); and the mean cosine between categories within a batch (conflict when negative).
Both objectives are scored on every world (`--loss l1,deltairis`), so what the world IS and what the LOSS allocates are separated.
Check: on the first batch, sum_c L_c must equal tworld.rollout_losses(world, s, a, 'teacher', recon=...) (printed, relative error).
--device cpu: the fmamba layers on the reference backend (tests).
Usage: gradanat.py <world.pt> ... [--loss l1,deltairis] [--batches N] [--device cpu]  -> artifacts/eda/health_chain_v1/gradanat_<world>.json
"""
import json
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE))
OUT = ROOT / 'artifacts/eda/health_chain_v1'
HEAD = ('proj.', 'choose.', 'frame.', 'target_gate.', 'event_head.')


def main():
    import teval as T
    import budget as BG
    from d4mj.config import config_from_dict
    from d4mj.train import autocast_context
    import spatial as Sp
    args = sys.argv[1:]
    losses = (args[args.index('--loss') + 1] if '--loss' in args else 'l1,deltairis').split(',')
    nb_max = int(args[args.index('--batches') + 1]) if '--batches' in args else 10 ** 9
    paths = [a for i, a in enumerate(args) if not a.startswith('--') and (i == 0 or args[i - 1] not in ('--loss', '--batches', '--device'))]
    dev = torch.device(args[args.index('--device') + 1] if '--device' in args else 'cuda')
    config = config_from_dict(torch.load(Sp.CHECKPOINT, map_location='cpu', weights_only=False)['config'])
    meta, _, _ = T.split()
    cache = T.build_cache('raw', torch.device('cpu'))
    R = len(meta['seed'])
    root, fut, fa = cache['ctx'][:, -1], cache['fut'], cache['fut_a']
    vis = torch.cat([meta['root_visible'][:, None], meta['future_visible'][:, 0]], 1)                # [R,17,1534]
    toks = torch.cat([root[:, None], fut], 1)                                                         # [R,17,81,192]
    alive = ~meta['future_dead'][:, 0].cumsum(1).bool()
    ncat = len(BG.CATS)
    for path in paths:
        world, st = T.load_world(Path(path), dev)
        world.eval()
        if dev.type == 'cpu':
            import dataclasses
            for layer in getattr(world, 'layers', []):
                layer.mix.settings = dataclasses.replace(layer.mix.settings, backend='reference')
        L = world.time.shape[0]
        ok = alive[:, L - 2]                                                                          # alive through future frame L-1
        idx = torch.where(ok)[0]
        # categories of every transition in the window (frame t -> t+1, t = 0..L-2)
        cats = torch.empty(len(idx), L - 1, 81, dtype=torch.long)
        for t in range(L - 1):
            c, _ = BG.categories(vis[idx, t], vis[idx, t + 1], toks[idx, t].float(), toks[idx, t + 1].float())
            cats[:, t] = c
        params = [p for n, p in world.named_parameters() if not n.startswith(HEAD) and p.requires_grad]
        bs = 16 if L <= 6 else 4
        res = {'world': st['name'], 'frames': L, 'windows': int(len(idx)), 'batch': bs, 'backbone_params': sum(p.numel() for p in params)}
        for loss in losses:
            acc = {c: {'loss': 0.0, 'norm_ratio': 0.0, 'proj': 0.0, 'gsum': None, 'nsum': 0.0} for c in BG.CATS}
            cos = torch.zeros(ncat, ncat, dtype=torch.float64)
            gtot_sum, gtot_n, nb = None, 0.0, 0
            for j in range(0, len(idx), bs):
                if nb >= nb_max:
                    break
                ii = idx[j:j + bs]
                s = toks[ii, :L].float().to(dev)
                a = torch.nn.functional.pad(fa[ii, :L - 1], (0, 1)).to(dev)
                with autocast_context(config):
                    pred = world(s, a)[0]
                e = pred[:, :L - 1].float() - s[:, 1:]                                                # [b,L-1,81,192]
                n_el = e.numel()
                ct = cats[j:j + bs].to(dev)                                                           # [b,L-1,81]
                if loss == 'l1':
                    per_tok = e.abs().sum(-1) / n_el                                                  # [b,L-1,81]
                else:
                    e2 = e ** 2
                    per_tok = (e2.sum(-1) + 0.1 * e.abs().sum(-1)) / n_el
                    worst = e2.flatten(2).max(-1)                                                     # per frame [b,L-1]
                    wtok = worst.indices // 192
                    add = torch.zeros_like(per_tok)
                    add.scatter_(2, wtok[..., None], (0.01 * worst.values / (e.shape[0] * e.shape[1]))[..., None])
                    per_tok = per_tok + add
                total = per_tok.sum()
                if nb == 0:
                    import tworld as TW
                    with torch.no_grad(), autocast_context(config):
                        ref = float(TW.rollout_losses(world, s, a[:, :L - 1], 'teacher', recon='l1' if loss == 'l1' else 'deltairis'))
                    res.setdefault('objective_check', {})[loss] = {'sum_of_terms': float(total), 'rollout_losses': ref,
                                                                   'rel_err': abs(float(total) - ref) / ref}
                g_tot = torch.autograd.grad(total, params, retain_graph=True, allow_unused=True)
                g_tot = torch.cat([(g if g is not None else torch.zeros_like(p)).flatten() for g, p in zip(g_tot, params)]).float()
                gn = float(g_tot.norm()) + 1e-12
                gs = []
                for ci, c in enumerate(BG.CATS):
                    m = ct == ci
                    lc = (per_tok * m).sum()
                    if m.any():
                        g = torch.autograd.grad(lc, params, retain_graph=True, allow_unused=True)
                        g = torch.cat([(gg if gg is not None else torch.zeros_like(p)).flatten() for gg, p in zip(g, params)]).float()
                    else:
                        g = torch.zeros_like(g_tot)
                    gs.append(g)
                    A = acc[c]
                    A['loss'] += float(lc); A['norm_ratio'] += float(g.norm()) / gn; A['proj'] += float(g @ g_tot) / gn ** 2
                    A['gsum'] = g.clone() if A['gsum'] is None else A['gsum'] + g; A['nsum'] += float(g.norm())
                G = torch.stack(gs)                                                                   # [ncat, P]
                Gn = G / G.norm(dim=1, keepdim=True).clamp(min=1e-12)
                cos += (Gn @ Gn.T).double().cpu()
                gtot_sum = g_tot.clone() if gtot_sum is None else gtot_sum + g_tot; gtot_n += gn
                nb += 1
                del G, Gn, gs, g_tot, pred, e
            tot_loss = sum(A['loss'] for A in acc.values())
            out = {}
            for c, A in acc.items():
                out[c] = {'loss_share': round(A['loss'] / tot_loss, 4), 'grad_norm_ratio': round(A['norm_ratio'] / nb, 4),
                          'projection_share': round(A['proj'] / nb, 4),
                          'consistency': round(float(A['gsum'].norm()) / A['nsum'], 4) if A['nsum'] > 0 else None}
            res[loss] = {'batches': nb, 'noise_consistency': round(nb ** -0.5, 4),
                         'total_consistency': round(float(gtot_sum.norm()) / gtot_n, 4), 'categories': out,
                         'cosine_matrix': {BG.CATS[i]: {BG.CATS[k]: round(float(cos[i, k] / nb), 3) for k in range(ncat)} for i in range(ncat)}}
            print(json.dumps({st['name']: {loss: {c: [v['loss_share'], v['projection_share'], v['consistency']] for c, v in out.items()}}}), flush=True)
            for A in acc.values():
                A['gsum'] = None
        (OUT / f"gradanat_{st['name']}.json").write_text(json.dumps(res, indent=1))
        del world; torch.cuda.empty_cache()


if __name__ == '__main__':
    sys.path.insert(0, str(ROOT / 'artifacts/experiments/20260926_diagnosis'))
    sys.path.insert(0, str(ROOT / 'artifacts/experiments/20260921_readout_ladder'))
    main()
