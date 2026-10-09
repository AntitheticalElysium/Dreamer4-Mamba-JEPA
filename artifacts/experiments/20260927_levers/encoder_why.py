"""10-10 PROGRAM, Part II-A follow-up: WHY is the Raw token target not game state? CPU, pixel interventions on the frozen encoder.
encoder_diag measured WHAT the target carries (own state R^2 0.59; day scroll re-encodes unchanged cells x25; HUD tokens move 9-12 per
step at unchanged values; night static). This asks WHERE it comes from by editing the encoder's input pixels. The diagnosis frames are
stored as uint8 63 x 63 x 3: 9 x 9 patches of 7 px, rows 0-6 the map, rows 7-8 the HUD, which the renderer draws after (and without)
the night static and tint. Every layer's patch tokens are read layer-normed, like the world's target; 'final' = the world's exact
target (LN of the backbone's last_hidden_state).
  E1 locality: in a root frame, replace ONE map patch p (62, not the player's) by the same patch of another root's frame (donor = a
     fixed permutation). The change of every token q, by Chebyshev distance d(p, q) (0..3, 4+) and the HUD, per layer. Context share
     = the share of the edit's total token change that lands on tokens other than p. Day / night; own state differs or not.
  E2 scroll re-encoding, decomposed. Consecutive frames of the factual future (root -> 16, both alive, scroll from the true tile maps
     as in budget.categories, DAY only). For map cells whose state did not change (budget.categories' no_state_change):
       real    |x_{t+1}(c) - x_t(src c)|^2
       transl  the same with frame t+1 replaced by frame t's pixels rigidly translated one patch (entering patches and the HUD from
               frame t+1): what re-encoding a pure translation costs
       pos     transl's frame encoded with the position embeddings permuted so every translated patch carries its SOURCE position
               (entering patches and the HUD keep theirs): what remains is context (the entering / leaving rows, the HUD)
     Per layer, by distance from the entering edge (1..4, 5+); also the pixel difference between the translated and the real
     frame on those cells.
  E3 HUD jitter: the same pairs where the HUD values and the inventory did not change. Is the HUD's pixel block identical? The HUD
     tokens' change (18 summed) per layer, still / scroll x day / night, split by pixel-identical HUD (= change caused by the map alone).
  E4 night static: depths where all five futures are alive with IDENTICAL visible states at night (light < 0.5); futures 0 and 1
     (same state, different static). Per cell q: full |x1(q) - x0(q)|^2, own (frame 0 with only q's patch from frame 1) and context
     (frame 1 with only q's patch from frame 0), by ring around the player; the pixel noise per ring; the HUD tokens.
  E5 per layer: held R^2 of the cell token from own state (S0) and own state + position + lighting / sleep (S2), with encoder_diag's
     designs, on all frames and split day / night (light < 0.5). Ridge from sufficient statistics (intercept, no standardization; lambda on the validation fifth of the train seeds, R^2
     on TEST seeds), on the root + factual-future frames. This is encoder_diag --encoder raw's frame set, so its S0 0.592 / S2 0.888
     are the check.
Check: the final tokens of the root frames equal teval's cache (max |diff| reported; fp16 rounding only).
Usage: encoder_why.py [--roots N] [--encoder raw|ldad1|ldad10|tc]  -> artifacts/eda/health_chain_v1/encoder_why[_NAME].json
"""
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE))
OUT = ROOT / 'artifacts/eda/health_chain_v1'
SHIFTS = ((0, 0), (-1, 0), (1, 0), (0, -1), (0, 1))
MAPC = [c for c in range(63) if c != 31]
LAYERS = ['emb'] + [f'L{j}' for j in range(1, 13)] + ['final']
L = len(LAYERS)
CHEB = torch.tensor([[min(max(abs(p // 9 - q // 9), abs(p % 9 - q % 9)), 4) if q < 63 else 5 for q in range(81)] for p in range(63)])
RING = torch.tensor([max(abs(c // 9 - 3), abs(c % 9 - 4)) for c in range(63)])


def patches(x):
    """uint8 [n,63,63,3] -> [n,81,7,7,3] (row-major patch order, as the ViT's)"""
    return x.reshape(-1, 9, 7, 9, 7, 3).permute(0, 1, 3, 2, 4, 5).reshape(-1, 81, 7, 7, 3)


def frames_of(p):
    return p.reshape(-1, 9, 9, 7, 7, 3).permute(0, 1, 3, 2, 4, 5).reshape(-1, 63, 63, 3)


def source(dr, dc):
    """[63]: the patch index in frame t of map cell c's content in frame t+1 (scroll.estimate's convention), -1 if entering"""
    return torch.tensor([(c // 9 + dr) * 9 + c % 9 + dc if 0 <= c // 9 + dr < 7 and 0 <= c % 9 + dc < 9 else -1 for c in range(63)])


def load_encoder(name):
    import teval as T
    from d4mj.checkpoint import read_lewm_bundle
    from d4mj.config import config_from_dict
    from d4mj.world_api import ModelBundle
    raw = torch.load(T.ENCODERS[name], map_location='cpu', weights_only=False)
    payload = read_lewm_bundle(T.ENCODERS[name]) if 'format' in raw else raw
    bundle = ModelBundle.create(config_from_dict(payload['config']))
    bundle.encoder.load_state_dict(payload['modules']['encoder'])
    return bundle.encoder.to('cpu').freeze()


@torch.no_grad()
def run(enc, x, pos=None, bs=256):
    """uint8 [n,63,63,3] -> layer-normed patch tokens of every layer [L, n, 81, 192]; pos: position embeddings to use instead"""
    pe = enc.backbone.embeddings.position_embeddings
    keep = pe.data.clone()
    if pos is not None:
        pe.data.copy_(pos)
    out = []
    try:
        for i in range(0, len(x), bs):
            px = (x[i:i + bs].permute(0, 3, 1, 2).float() / 255.0 - enc.pixel_mean) / enc.pixel_std
            o = enc.backbone(px, interpolate_pos_encoding=True, output_hidden_states=True)
            out.append(torch.stack([F.layer_norm(h[:, 1:].float(), (192,)) for h in o.hidden_states]
                                   + [F.layer_norm(o.last_hidden_state[:, 1:].float(), (192,))]))
    finally:
        pe.data.copy_(keep)
    return torch.cat(out, 1)


def main():
    import teval as T
    import encoder_diag as ED
    import budget as BU
    from scroll import estimate
    from rollouts import STORE
    torch.set_num_threads(8)
    t_start = time.time()
    args = sys.argv[1:]
    name = args[args.index('--encoder') + 1] if '--encoder' in args else 'raw'
    meta, train_roots, train_seeds = T.split()
    R = int(args[args.index('--roots') + 1]) if '--roots' in args else len(meta['seed'])
    val_roots = torch.isin(meta['seed'], train_seeds[:len(train_seeds) // 5])
    part = torch.where(~train_roots, 2, torch.where(val_roots, 1, 0))[:R]
    enc = load_encoder(name)
    pe0 = enc.backbone.embeddings.position_embeddings.data.clone()
    rootx, futx = [], []
    for f in sorted(STORE.glob('seed-*.pt')):
        if len(rootx) >= R:
            break
        for r in torch.load(f, weights_only=False):
            if len(rootx) < R:
                rootx.append(r['context'][-1]); futx.append(r['future_frames'])
    rootx, futx = torch.stack(rootx), torch.stack(futx)                                               # [R,63,63,3], [R,5,16,63,63,3]
    rootv, futv = meta['root_visible'][:R], meta['future_visible'][:R]
    alive = ~meta['future_dead'][:R].cumsum(2).bool()                                                 # [R,5,16]
    cache = T.build_cache(name, torch.device('cpu'))
    chk = run(enc, rootx[:64])[-1] - cache['ctx'][:min(64, R), -1].float()
    res = {'encoder': name, 'roots': R, 'layers': LAYERS,
           'check_final_vs_cache': {'max_abs': round(float(chk.abs().max()), 5), 'mean_abs': round(float(chk.abs().mean()), 6)}}
    print(json.dumps(res), flush=True)

    # E1 locality
    donor = torch.randperm(R, generator=torch.Generator().manual_seed(20261010))
    tiles_r, mobs_r = ED.frame_state(rootv)[:2]
    e1 = torch.zeros(L, 24, 2, dtype=torch.float64)                                                    # layer x (night, differs, bucket) x [sum, n]
    e1own = torch.zeros(L, 4, 2, dtype=torch.float64)                                                  # layer x (night, differs) x [own, all]
    mi = torch.tensor(MAPC)
    for i in range(0, min(R, 500), 4):
        b = min(4, min(R, 500) - i)
        x = rootx[i:i + b]
        base = run(enc, x)
        E = patches(x)[:, None].expand(b, 62, 81, 7, 7, 3).clone()
        E[:, torch.arange(62), mi] = patches(rootx[donor[i:i + b]])[:, mi]
        d = ((run(enc, frames_of(E)).view(L, b, 62, 81, 192) - base[:, :, None]) ** 2).sum(-1)       # [L,b,62,81]
        dj = donor[i:i + b]
        differs = ((tiles_r[i:i + b].flatten(1) != tiles_r[dj].flatten(1)) | (mobs_r[i:i + b].flatten(1) != mobs_r[dj].flatten(1)))[:, mi]
        key = (rootv[i:i + b, 1521].float() < 0.5).long()[:, None] * 2 + differs.long()                # [b,62]
        idx = (key[:, :, None] * 6 + CHEB[mi][None]).flatten()
        own = d[:, :, torch.arange(62), mi]                                                            # [L,b,62]
        for l in range(L):
            e1[l, :, 0] += torch.bincount(idx, weights=d[l].flatten().double(), minlength=24)
            e1[l, :, 1] += torch.bincount(idx, minlength=24).double()
            e1own[l, :, 0] += torch.bincount(key.flatten(), weights=own[l].flatten().double(), minlength=4)
            e1own[l, :, 1] += torch.bincount(key.flatten(), weights=d[l].sum(-1).flatten().double(), minlength=4)
    res['E1_locality'] = {}
    for kv in range(4):
        nm = ('day', 'night')[kv // 2] + ('/same_state', '/state_differs')[kv % 2]
        row = {}
        for l in range(L):
            s = e1[l, kv * 6:(kv + 1) * 6]
            row[LAYERS[l]] = {'mean_dx2_by_distance': [round(float(s[j, 0] / s[j, 1]), 4) if s[j, 1] else None for j in range(5)],
                              'mean_dx2_hud_token': round(float(s[5, 0] / s[5, 1]), 4) if s[5, 1] else None,
                              'context_share': round(float(1 - e1own[l, kv, 0] / e1own[l, kv, 1]), 4) if e1own[l, kv, 1] else None}
        res['E1_locality'][nm] = {'edits': int(e1[0, kv * 6, 1]), 'layers': row}
    print(json.dumps({'E1_final': {k: v['layers']['final'] for k, v in res['E1_locality'].items()}}), f'{time.time() - t_start:.0f}s', flush=True)

    # E2 / E3 / E5 over root + factual-future frames
    nc_id = BU.CATS.index('no_state_change')
    e2 = torch.zeros(3, L, 5, 2, dtype=torch.float64)                                                  # real / transl / pos x layer x edge dist x [sum, n]
    e2pix = torch.zeros(2, dtype=torch.float64)
    e3 = torch.zeros(L, 8, 2, dtype=torch.float64)                                                     # layer x (scroll, night, pixel-identical) x [sum, n]
    e3n = torch.zeros(2, dtype=torch.long)
    D = 372
    S = torch.zeros(6, D, D, dtype=torch.float64); XY = torch.zeros(6, L, D, 192, dtype=torch.float64)          # (part, night)
    YY = torch.zeros(6, L, dtype=torch.float64); SY = torch.zeros(6, L, 192, dtype=torch.float64); NN = torch.zeros(6, dtype=torch.float64)
    for i in range(0, R, 8):
        b = min(8, R - i)
        seq = torch.cat([rootx[i:i + b, None], futx[i:i + b, 0]], 1)                                    # [b,17,63,63,3]
        vis = torch.cat([rootv[i:i + b, None], futv[i:i + b, 0]], 1)
        al = torch.cat([torch.ones(b, 1, dtype=torch.bool), alive[i:i + b, 0]], 1)                     # [b,17]
        tok = run(enc, seq.flatten(0, 1)).view(L, b, 17, 81, 192)
        fm = al.flatten()
        tl, mc, lt, sl = ED.frame_state(vis.flatten(0, 1)[fm])
        X = torch.cat([torch.ones(len(tl) * 62, 1), ED.design(tl, mc, lt, sl)[:, :D - 1]], 1).double()
        pr = (part[i:i + b, None].expand(b, 17).flatten()[fm] * 2 + (lt < 0.5).long()).repeat_interleave(62)
        Y = tok.flatten(1, 2)[:, fm][:, :, MAPC].flatten(1, 2).double()                                # [L, m*62, 192]
        for p in range(6):
            s = pr == p
            if s.any():
                Xs, Ys = X[s], Y[:, s]
                S[p] += Xs.T @ Xs; XY[p] += torch.einsum('nd,lnk->ldk', Xs, Ys)
                YY[p] += (Ys ** 2).sum((1, 2)); SY[p] += Ys.sum(1); NN[p] += int(s.sum())
        del X, Y
        bi, ti = torch.where(al[:, :-1] & al[:, 1:])
        v0, v1, x0, x1 = vis[bi, ti], vis[bi, ti + 1], seq[bi, ti], seq[bi, ti + 1]
        k0, k1 = tok[:, bi, ti], tok[:, bi, ti + 1]                                                    # [L,n,81,192]
        a0, a1 = ED.frame_state(v0)[0], ED.frame_state(v1)[0]
        frac = []
        for dr, dc in SHIFTS:
            src = ED.neighbour(a0, dr, dc, -1); ov = src >= 0
            frac.append(((src != a1) & ov).flatten(1).sum(1).float() / ov.flatten(1).sum(1).float())
        sh = (torch.stack(frac, 1) + 1e-3 * (torch.arange(5)[None] != estimate(k0[-1], k1[-1])[:, None]).float()).argmin(1)
        night = (v0[:, 1521].float() < 0.5) | (v1[:, 1521].float() < 0.5)
        hud_same = (v0[:, 1512:1516] == v1[:, 1512:1516]).all(1) & (v0[:, 1522:1534] == v1[:, 1522:1534]).all(1)
        pix_same = (x0[:, 49:] == x1[:, 49:]).flatten(1).all(1)
        dh = ((k1[:, :, 63:] - k0[:, :, 63:]) ** 2).sum((-1, -2))                                       # [L,n]
        key = ((sh != 0).long() * 2 + night.long()) * 2 + pix_same.long()
        for l in range(L):
            e3[l, :, 0] += torch.bincount(key[hud_same], weights=dh[l][hud_same].double(), minlength=8)
            e3[l, :, 1] += torch.bincount(key[hud_same], minlength=8).double()
        e3n += torch.tensor([int(hud_same.sum()), int((hud_same & pix_same).sum())])
        cat = BU.categories(v0, v1, k0[-1], k1[-1])[0]
        for si in range(1, 5):
            sel = (sh == si) & ~night
            if not sel.any():
                continue
            dr, dc = SHIFTS[si]
            src = source(dr, dc)
            pp = patches(x0[sel]).clone()
            xt = patches(x1[sel]).clone()
            xt[:, :63][:, src >= 0] = pp[:, src[src >= 0]]
            xt = frames_of(xt)
            pos = pe0.clone()
            pos[0, 1 + torch.where(src >= 0)[0]] = pe0[0, 1 + src[src >= 0]]
            tt, tp = run(enc, xt), run(enc, xt, pos)
            nc = (cat[sel][:, :63] == nc_id) & (src >= 0)[None]                                       # [m,63]
            r_, c_ = torch.arange(63) // 9, torch.arange(63) % 9
            dist = {1: 6 - r_, -1: r_}[dr] if dr else {1: 8 - c_, -1: c_}[dc]
            bk = (dist.clamp(1, 5) - 1)[None].expand_as(nc)[nc]
            ref = k0[:, sel][:, :, src.clamp(min=0)]
            for j, cand in enumerate((k1[:, sel][:, :, :63], tt[:, :, :63], tp[:, :, :63])):
                dd = ((cand - ref) ** 2).sum(-1)[:, nc]                                                # [L, cells]
                for l in range(L):
                    e2[j, l, :, 0] += torch.bincount(bk, weights=dd[l].double(), minlength=5)
                    e2[j, l, :, 1] += torch.bincount(bk, minlength=5).double()
            e2pix += torch.tensor([float(((patches(xt)[:, :63].float() - patches(x1[sel])[:, :63].float()) ** 2).sum((-1, -2, -3))[nc].sum()),
                                   float(nc.sum())], dtype=torch.float64)
        if i % 160 == 0:
            print(f'E2/E3/E5 roots {i + b}/{R}', f'{time.time() - t_start:.0f}s', flush=True)
    res['E2_scroll_reencoding_day'] = {'cells': int(e2[0, 0, :, 1].sum()),
                                       'pixel_dx2_translated_vs_real_per_cell': round(float(e2pix[0] / e2pix[1]), 2) if e2pix[1] else None,
                                       'layers': {}}
    for l in range(L):
        row = {}
        for j, jn in enumerate(('real', 'transl', 'pos_matched')):
            row[jn] = round(float(e2[j, l, :, 0].sum() / e2[j, l, :, 1].sum()), 4)
            row[jn + '_by_edge_distance'] = [round(float(e2[j, l, q, 0] / e2[j, l, q, 1]), 4) if e2[j, l, q, 1] else None for q in range(5)]
        res['E2_scroll_reencoding_day']['layers'][LAYERS[l]] = row
    res['E3_hud_jitter'] = {'unchanged_hud_pairs': int(e3n[0]), 'pixel_identical_share': round(float(e3n[1] / e3n[0]), 4), 'layers': {}}
    for l in range(L):
        row = {}
        for kv in range(8):
            if e3[l, kv, 1]:
                row[f"{('still', 'scroll')[kv // 4]}/{('day', 'night')[kv // 2 % 2]}/{('pixels_differ', 'pixels_identical')[kv % 2]}"] = \
                    {'pairs': int(e3[l, kv, 1]), 'mean_dx2_18tokens': round(float(e3[l, kv, 0] / e3[l, kv, 1]), 4)}
        res['E3_hud_jitter']['layers'][LAYERS[l]] = row
    res['E5_R2_by_layer'] = {}
    for nm, cols in (('S0_own_state', 86), ('S2_own_position_lighting', D)):
        for sub, nv in (('all', (0, 1)), ('day', (0,)), ('night', (1,))):
            g = lambda T_, p: sum(T_[p * 2 + v] for v in nv)                                          # statistics of part p, this subset
            if min(float(g(NN, p)) for p in range(3)) == 0:
                continue
            row = {}
            for l in range(L):
                best = None
                for lam in (1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2):
                    W = torch.linalg.solve(g(S, 0)[:cols, :cols] + lam * g(NN, 0) * torch.eye(cols, dtype=torch.float64), g(XY, 0)[l, :cols])
                    sse = [float(g(YY, p)[l] - 2 * (W * g(XY, p)[l, :cols]).sum() + (W * (g(S, p)[:cols, :cols] @ W)).sum()) for p in (1, 2)]
                    if best is None or sse[0] < best[0]:
                        best = (sse[0], sse[1], lam)
                mu = g(SY, 0)[l] / g(NN, 0)
                sst = float(g(YY, 2)[l] - 2 * (mu * g(SY, 2)[l]).sum() + g(NN, 2) * (mu ** 2).sum())
                row[LAYERS[l]] = round(1 - best[1] / sst, 4) if sst > 0 else None
            res['E5_R2_by_layer'][f'{nm}/{sub}'] = row
    print(json.dumps({k: res[k] for k in ('E5_R2_by_layer',)}), f'{time.time() - t_start:.0f}s', flush=True)

    # E4 night static: own patch vs context
    cases = [(i, k) for i in range(R) for k in range(16)
             if alive[i, :, k].all() and bool((futv[i, 1:, k] == futv[i, :1, k]).all()) and float(futv[i, 0, k, 1521]) < 0.5]
    pick = torch.randperm(len(cases), generator=torch.Generator().manual_seed(20261010))[:250].tolist()
    cases = [cases[j] for j in sorted(pick)]
    e4 = torch.zeros(4, 6, 2, dtype=torch.float64)                                                     # full / own / ctx / pixel x (ring 0..4, HUD) x [sum, n]
    hud_pix_same = 0
    for c0 in range(0, len(cases), 4):
        cs = cases[c0:c0 + 4]
        x0 = torch.stack([futx[i, 0, k] for i, k in cs]); x1 = torch.stack([futx[i, 1, k] for i, k in cs])
        b = len(cs)
        hud_pix_same += int((x0[:, 49:] == x1[:, 49:]).flatten(1).all(1).sum())
        t0, t1 = run(enc, x0)[-1], run(enc, x1)[-1]
        P0, P1 = patches(x0), patches(x1)
        Hh = P0[:, None].expand(b, 63, 81, 7, 7, 3).clone(); Hh[:, torch.arange(63), torch.arange(63)] = P1[:, :63]
        Gg = P1[:, None].expand(b, 63, 81, 7, 7, 3).clone(); Gg[:, torch.arange(63), torch.arange(63)] = P0[:, :63]
        th = run(enc, frames_of(Hh))[-1].view(b, 63, 81, 192)[:, torch.arange(63), torch.arange(63)]
        tg = run(enc, frames_of(Gg))[-1].view(b, 63, 81, 192)[:, torch.arange(63), torch.arange(63)]
        full = ((t1 - t0) ** 2).sum(-1)                                                                # [b,81]
        own, ctx = ((th - t0[:, :63]) ** 2).sum(-1), ((tg - t0[:, :63]) ** 2).sum(-1)
        pix = ((P1[:, :63].float() - P0[:, :63].float()) ** 2).sum((-1, -2, -3))
        for j, v in enumerate((full[:, :63], own, ctx, pix)):
            e4[j, :5, 0] += torch.bincount(RING, weights=v.sum(0).double(), minlength=5)
            e4[j, :5, 1] += torch.bincount(RING, minlength=5).double() * b
        e4[0, 5] += torch.tensor([float(full[:, 63:].sum(1).sum()), b], dtype=torch.float64)
    res['E4_night_static'] = {'cases': len(cases), 'hud_pixels_identical': hud_pix_same,
                              'by_ring': {f'ring{g}': {n_: round(float(e4[j, g, 0] / e4[j, g, 1]), 4)
                                                       for j, n_ in enumerate(('full_dx2', 'own_patch_only_dx2', 'context_only_dx2', 'pixel_dx2'))}
                                          for g in range(5)},
                              'hud_18tokens_full_dx2': round(float(e4[0, 5, 0] / e4[0, 5, 1]), 4) if e4[0, 5, 1] else None}
    res['seconds'] = round(time.time() - t_start)
    OUT.mkdir(parents=True, exist_ok=True)
    out = OUT / f"encoder_why{'' if name == 'raw' else '_' + name}{'' if R == len(meta['seed']) else f'_r{R}'}.json"
    out.write_text(json.dumps(res, indent=1))
    print(json.dumps({k: res[k] for k in ('E4_night_static', 'E3_hud_jitter')})[:3000], flush=True)
    print('wrote', out, flush=True)


if __name__ == '__main__':
    sys.path.insert(0, str(ROOT / 'artifacts/experiments/20260926_diagnosis'))
    sys.path.insert(0, str(ROOT / 'artifacts/experiments/20260921_readout_ladder'))
    main()
