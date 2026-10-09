"""10-10 PROGRAM, Part II-A (encoder; prediction P-A), CPU only. What does a frozen Raw token carry, and how much of it -- and of its
frame-to-frame change -- is the game state of its own cell?
Data: every diagnosis frame with its simulator state (1,002 roots, 143 episodes): the root frame (teval cache) and the five 16-step
futures (stochdiag cache), alive frames only; 81,162 frames. Split = teval's (70% / 30% of episode seeds; ridge lambda chosen on a
held-out fifth of the train seeds, as teval.ridge). Map cells only (63, the player cell 31 handled separately).
Cell state: tile class (17), mob category (none / zombie / skeleton / cow / arrow, first present in that order); global: light level
(10 bins), sleeping; position (63); 4-neighbour and diagonal-neighbour tile classes (+ out of view) and mob categories.
  A2 static composition: held R^2 of the cell token (192 dims pooled) from nested linear one-hot designs
       S0 own state (17 x 5) | S1 + position | S2 + global (light x sleep, tile x sleep, tile x light) | S3 + 4 neighbours |
       S4 + diagonals; and an MLP on S4's inputs (upper bound for "what the visible 3 x 3 + global state explains")
  A2 change composition: consecutive alive frames (root -> 1, k -> k+1, each future), scroll from the TRUE tile maps (the shift with
       fewest tile mismatches); every map cell at t+1 with a source at t (entering cells excluded), dx = x_{t+1}(cell) - x_t(source).
       Energy sum |dx|^2 and per-cell mean by cause, exclusive in this order: own state changed | source was the player cell (the
       sprite leaves the tile) | a 4-neighbour's state changed | sleep changed | light changed (> 0.02) | nothing in the state changed.
       Also split by sleeping / awake.
  A3 currency: per tile class (no mob, awake), the class mean (train) and within-class squared spread (test); for every class pair,
       |mu_a - mu_b|^2 (the squared token error of drawing class a where b is) vs the within-class spreads; compared with the per-cell
       |dx|^2 of the non-state causes above (what the loss pays for rendering / context changes).
  A1 information ceilings: held probes of tile class and each mob from the own token / the 3 x 3 tokens (ridge; MLP), HUD values and
       inventory from the 18 HUD tokens, facing and sleep from the player token, light from the mean map token.
Added in testing (10-10 09:3x, before the full run): the L1 currency of every change share (sum |dx|; the L1 loss pays this, not
|dx|^2); unchanged cells split by Chebyshev distance to the nearest change and scroll / still step; the share of unchanged cells'
scroll re-encoding explained by S4 (state, position, global, neighbours; 15% sample); HUD-token and player-token change when the HUD
values (+ inventory) / the facing and sleep are unchanged, scroll vs still.
Usage: encoder_diag.py [--roots N (test)]  -> artifacts/eda/health_chain_v1/encoder_diag.json
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
NB4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
NBD = ((-1, -1), (-1, 1), (1, -1), (1, 1))
LAMS = (1e-4, 1e-3, 1e-2, 1e-1, 1.0)


def mob_cat(mobs):
    """[..., 7] mob channels (zombie, cow, skeleton, arrows x4) -> 0 none, 1 zombie, 2 skeleton, 3 cow, 4 arrow (first present)"""
    m = mobs > 0
    cat = torch.zeros(m.shape[:-1], dtype=torch.long)
    for val, ch in ((4, m[..., 3:].any(-1)), (3, m[..., 1]), (2, m[..., 2]), (1, m[..., 0])):
        cat = torch.where(ch, torch.full_like(cat, val), cat)
    return cat


def frame_state(vis):
    """[n, 1534] -> tiles [n,7,9], mobcat [n,7,9], light [n], sleep [n]"""
    v = vis.float()
    return (v[:, :1071].view(-1, 7, 9, 17).argmax(-1), mob_cat(v[:, 1071:1512].view(-1, 7, 9, 7)), v[:, 1521], v[:, 1520] > 0.5)


def neighbour(grid, dr, dc, fill):
    """[n,7,9] -> value of the (r + dr, c + dc) neighbour, `fill` off-view"""
    out = torch.full_like(grid, fill)
    rs, re = max(0, -dr), 7 - max(0, dr)
    cs, ce = max(0, -dc), 9 - max(0, dc)
    out[:, rs:re, cs:ce] = grid[:, rs + dr:re + dr, cs + dc:ce + dc]
    return out


def design(tiles, mobc, light, sleep, upto):
    """one-hot design per map cell (MAPC order): [n * 62, D] float32, nested sets S0..S4"""
    n = len(tiles)
    lb = (light.clamp(0, 0.999) * 10).long()                                                       # [n]
    cell = lambda g: g.flatten(1)[:, MAPC]                                                         # [n, 62]
    own = cell(tiles) * 5 + cell(mobc)                                                             # 85
    parts = [F.one_hot(own, 85)]
    if upto >= 1:
        parts.append(F.one_hot(torch.arange(62).expand(n, 62), 62))
    if upto >= 2:
        sl = sleep.long()[:, None].expand(n, 62)
        lbe = lb[:, None].expand(n, 62)
        parts += [F.one_hot(lbe * 2 + sl, 20), F.one_hot(cell(tiles) * 2 + sl, 34), F.one_hot(cell(tiles) * 10 + lbe, 170)]
    if upto >= 3:
        for dr, dc in NB4:
            parts += [F.one_hot(cell(neighbour(tiles, dr, dc, 17)), 18), F.one_hot(cell(neighbour(mobc, dr, dc, 0)), 5)]
    if upto >= 4:
        for dr, dc in NBD:
            parts += [F.one_hot(cell(neighbour(tiles, dr, dc, 17)), 18), F.one_hot(cell(neighbour(mobc, dr, dc, 0)), 5)]
    return torch.cat(parts, -1).flatten(0, 1).float()


def ridge_solve(stats, lam):
    xx, xy = stats['xx'], stats['xy']
    d = xx.shape[0]
    return torch.linalg.solve(xx + lam * stats['n'] * torch.eye(d, dtype=xx.dtype), xy)


def main():
    import teval as T
    import stochdiag as SD
    torch.set_num_threads(6)
    t0 = time.time()
    args = sys.argv[1:]
    meta, train_roots, train_seeds = T.split()
    R = int(args[args.index('--roots') + 1]) if '--roots' in args else len(meta['seed'])
    val_roots = torch.isin(meta['seed'], train_seeds[:len(train_seeds) // 5])
    part = torch.where(~train_roots, 2, torch.where(val_roots, 1, 0))[:R]                           # 0 fit, 1 val, 2 test
    cache = T.build_cache('raw', torch.device('cpu'))
    fut5, _ = SD.token_cache(torch.device('cpu'))
    rootv, futv = meta['root_visible'][:R], meta['future_visible'][:R]
    alive = ~meta['future_dead'][:R].cumsum(2).bool()                                               # [R,5,16]

    def frames(i, j):
        """roots i..j: tokens [m,81,192] float32, visible [m,1534], part [m], plus consecutive-pair indices (a, b)"""
        toks, vis, prt, pairs = [], [], [], []
        base = 0
        for r in range(i, j):
            toks.append(cache['ctx'][r, -1:].float()); vis.append(rootv[r:r + 1]); prt.append(part[r:r + 1])
            idx0 = base; base += 1
            for s in range(5):
                prev = idx0
                for k in range(16):
                    if not alive[r, s, k]:
                        break
                    toks.append(fut5[r, s, k:k + 1].float()); vis.append(futv[r, s, k:k + 1]); prt.append(part[r:r + 1])
                    pairs.append((prev, base)); prev = base; base += 1
        return torch.cat(toks), torch.cat(vis), torch.cat(prt), torch.tensor(pairs)

    res = {'frames': 0}
    sets = range(5)
    stats = {u: {p: {'xx': 0, 'xy': 0, 'yy': 0., 'ysum': 0, 'n': 0} for p in range(3)} for u in sets}
    change = {}
    cls_sum = torch.zeros(17, 192, dtype=torch.float64); cls_n = torch.zeros(17, dtype=torch.float64)
    cls_sq = {p: torch.zeros(17, dtype=torch.float64) for p in range(3)}
    sub = {'x': [], 'y': [], 'p': []}                                                              # MLP subsample
    reenc = {'Xb': [], 'Xa': [], 'd': [], 'p': []}                                               # scroll re-encoding sample
    hudch = {}                                                                                   # (b) HUD / player token change
    gsub = {'hud': [], 'hudy': [], 'pl': [], 'ply': [], 'mean': [], 'meany': [], 'own': [], 'nb': [], 'celly': [], 'p': [], 'gp': []}
    gen = torch.Generator().manual_seed(20261010)
    for i in range(0, R, 32):
        x, vis, prt, pairs = frames(i, min(R, i + 32))
        tiles, mobc, light, sleep = frame_state(vis)
        y = x[:, MAPC].flatten(0, 1).double()                                                       # [m*62, 192]
        pc = prt[:, None].expand(-1, 62).flatten()
        res['frames'] += len(x)
        for u in sets:
            X = design(tiles, mobc, light, sleep, u).double()
            for p in range(3):
                mk = pc == p
                Xp, yp = X[mk], y[mk]
                st = stats[u][p]
                st['xx'] = st['xx'] + Xp.T @ Xp; st['xy'] = st['xy'] + Xp.T @ yp
                st['yy'] += float((yp ** 2).sum()); st['ysum'] = st['ysum'] + yp.sum(0); st['n'] += len(yp)
        # A3 class statistics: no mob, awake cells
        tc, mc = tiles.flatten(1)[:, MAPC].flatten(), mobc.flatten(1)[:, MAPC].flatten()
        awake = (~sleep)[:, None].expand(-1, 62).flatten()
        clean = (mc == 0) & awake
        for p in range(3):
            mk = clean & (pc == p)
            if p == 0:
                cls_sum.index_add_(0, tc[mk], y[mk]); cls_n.index_add_(0, tc[mk], torch.ones(int(mk.sum()), dtype=torch.float64))
        # change composition over consecutive pairs. Shift convention = scroll.estimate's: content at (r, c) in frame t+1 was at
        # (r + dr, c + dc) in frame t. Chosen by the TRUE tile maps (fraction of mismatching tiles over the overlap), ties broken by
        # scroll.estimate on the tokens.
        from scroll import estimate
        a, b = pairs[:, 0], pairs[:, 1]
        ta, tb = tiles[a], tiles[b]
        frac = []
        for dr, dc in SHIFTS:
            src = neighbour(ta, dr, dc, -1)
            ov = src >= 0
            frac.append(((src != tb) & ov).flatten(1).sum(1).float() / ov.flatten(1).sum(1).float())
        est = estimate(x[a], x[b])
        sh = (torch.stack(frac, 1) + 1e-3 * (torch.arange(5)[None] != est[:, None]).float()).argmin(1)
        res.setdefault('shift_counts', [0] * 5)
        for si in range(5):
            res['shift_counts'][si] += int((sh == si).sum())
        res.setdefault('shift_agree_with_token_estimate', [0, 0])
        res['shift_agree_with_token_estimate'][0] += int((sh == est).sum()); res['shift_agree_with_token_estimate'][1] += len(sh)
        keys, energies, l1s = [], [], []
        for si, (dr, dc) in enumerate(SHIFTS):
            sel = torch.where(sh == si)[0]
            if not len(sel):
                continue
            aa, bb = a[sel], b[sel]
            n_ = len(sel)
            xa = x[aa, :63].view(n_, 7, 9, 192); xb = x[bb, :63].view(n_, 7, 9, 192)
            src_x = torch.zeros_like(xa)
            rs0, re0, cs0, ce0 = max(0, -dr), 7 - max(0, dr), max(0, -dc), 9 - max(0, dc)
            src_x[:, rs0:re0, cs0:ce0] = xa[:, rs0 + dr:re0 + dr, cs0 + dc:ce0 + dc]
            has_src = torch.zeros(7, 9, dtype=torch.bool); has_src[rs0:re0, cs0:ce0] = True
            src_is_player = torch.zeros(7, 9, dtype=torch.bool)
            if 0 <= 3 - dr < 7 and 0 <= 4 - dc < 9:
                src_is_player[3 - dr, 4 - dc] = True                                                   # the cell whose source is (3, 4)
            st_t, st_m = neighbour(tiles[aa], dr, dc, -1), neighbour(mobc[aa], dr, dc, -1)
            own_ch = (st_t != tiles[bb]) | (st_m != mobc[bb])                                            # [n,7,9] (no source: True)
            ch_any = (own_ch & has_src[None]) | src_is_player[None]                                      # a real state / sprite change
            # Chebyshev distance to the nearest changed cell (0 = itself), by repeated 3x3 dilation; 9 = none in view
            dist = torch.full((n_, 7, 9), 9, dtype=torch.long)
            cur = ch_any.float()
            for d_ in range(0, 9):
                hit = (cur > 0) & (dist == 9)
                dist[hit] = d_
                cur = F.max_pool2d(cur[:, None], 3, 1, 1)[:, 0]
            nb_ch = torch.zeros_like(own_ch)
            for ndr, ndc in NB4:
                nb_ch |= neighbour(own_ch & has_src[None], ndr, ndc, 0).bool()
            dxv = xb - src_x
            dx2, dx1 = (dxv ** 2).sum(-1).double(), dxv.abs().sum(-1).double()
            cause = torch.full((n_, 7, 9), 5, dtype=torch.long)
            for val, m in ((4, ((light[bb] - light[aa]).abs() > 0.02)[:, None, None].expand(-1, 7, 9)),
                           (3, (sleep[bb] != sleep[aa])[:, None, None].expand(-1, 7, 9)), (2, nb_ch),
                           (1, src_is_player[None].expand(n_, -1, -1)), (0, own_ch)):
                cause = torch.where(m, torch.full_like(cause, val), cause)
            valid = has_src[None].expand(n_, -1, -1).clone(); valid[:, 3, 4] = False
            k_ = cause * 6 + sleep[bb].long()[:, None, None] * 3 + prt[bb][:, None, None]
            keys.append(k_[valid]); energies.append(dx2[valid]); l1s.append(dx1[valid])
            if si != 0:                                                                                  # scroll re-encoding sample
                u5 = valid & (cause == 5)
                ii, rr, cc = torch.where(u5)
                keep = torch.rand(len(ii), generator=gen) < 0.15
                ii, rr, cc = ii[keep], rr[keep], cc[keep]
                if len(ii):
                    Db = design(tiles[bb][ii], mobc[bb][ii], light[bb][ii], sleep[bb][ii], 4).view(len(ii), 62, -1)
                    Da = design(tiles[aa][ii], mobc[aa][ii], light[aa][ii], sleep[aa][ii], 4).view(len(ii), 62, -1)
                    mi = torch.tensor(MAPC)
                    pos_b = rr * 9 + cc; pos_a = (rr + dr) * 9 + (cc + dc)
                    jb = (mi[None] == pos_b[:, None]).float().argmax(1); ja = (mi[None] == pos_a[:, None]).float().argmax(1)
                    ar = torch.arange(len(ii))
                    reenc['Xb'].append(Db[ar, jb].half()); reenc['Xa'].append(Da[ar, ja].half())
                    reenc['d'].append(dxv[ii, rr, cc].half()); reenc['p'].append(prt[bb][ii])
            # unchanged-state cells (cause 5): by distance to the nearest change, and scroll vs no scroll
            u = valid & (cause == 5)
            dk = dist.clamp(max=5) * 2 + int(si != 0)                                                   # distance bin x scrolled
            e_ = torch.bincount(dk[u], weights=dx2[u], minlength=12); l_ = torch.bincount(dk[u], weights=dx1[u], minlength=12)
            c_ = torch.bincount(dk[u], minlength=12)
            for j in range(12):
                acc = res.setdefault('unchanged_by_distance', {}).setdefault(str(j), [0.0, 0.0, 0])
                acc[0] += float(e_[j]); acc[1] += float(l_[j]); acc[2] += int(c_[j])
        keys, energies, l1s = torch.cat(keys), torch.cat(energies), torch.cat(l1s)
        e_k = torch.bincount(keys, weights=energies, minlength=36); n_k = torch.bincount(keys, minlength=36)
        l_k = torch.bincount(keys, weights=l1s, minlength=36)
        for kk in range(36):
            if n_k[kk] > 0:
                cv, rest = divmod(kk, 6); sv, pp = divmod(rest, 3)
                e, n, l1 = change.get((cv, sv, pp), (0.0, 0, 0.0))
                change[(cv, sv, pp)] = (e + float(e_k[kk]), n + int(n_k[kk]), l1 + float(l_k[kk]))
        hud_same = (vis[a, 1512:1516] == vis[b, 1512:1516]).all(1) & (vis[a, 1522:1534] == vis[b, 1522:1534]).all(1)   # HUD values + inventory
        fac_same = (vis[a, 1516:1520] == vis[b, 1516:1520]).all(1) & (vis[a, 1520] == vis[b, 1520])
        hud2 = ((x[b, 63:81] - x[a, 63:81]) ** 2).sum((-1, -2)).double()
        pl2 = ((x[b, 31] - x[a, 31]) ** 2).sum(-1).double()
        for nm, val, same in (('hud18', hud2, hud_same), ('player', pl2, fac_same)):
            for scr in (0, 1):
                for sm in (0, 1):
                    mk = ((sh != 0).long() == scr) & (same.long() == sm)
                    acc = hudch.setdefault(f'{nm}/{"state_same" if sm else "state_changed"}/{"scroll" if scr else "still"}', [0.0, 0])
                    acc[0] += float(val[mk].sum()); acc[1] += int(mk.sum())
        # subsamples for MLP probes and A1
        mk_sub = torch.rand(len(y), generator=gen) < 0.08
        sub['x'].append(design(tiles, mobc, light, sleep, 4)[mk_sub].half()); sub['y'].append(y[mk_sub].half()); sub['p'].append(pc[mk_sub])
        gsub['hud'].append(x[:, 63:81].flatten(1).half()); gsub['hudy'].append(vis[:, 1512:1516].float() * 9)
        gsub['gp'].append(prt)
        gsub['pl'].append(x[:, 31].half()); gsub['ply'].append(torch.cat([vis[:, 1516:1520].float().argmax(-1, keepdim=True).float(), vis[:, 1520:1521].float(), vis[:, 1522:1534].float() * 9], 1))
        gsub['mean'].append(x[:, MAPC].mean(1).half()); gsub['meany'].append(light[:, None])
        mk_c = torch.rand(len(y), generator=gen) < 0.03
        g3 = F.pad(x[:, :63].view(-1, 7, 9, 192).permute(0, 3, 1, 2), (1, 1, 1, 1)).unfold(2, 3, 1).unfold(3, 3, 1)   # [m,192,7,9,3,3]
        nb = g3.permute(0, 2, 3, 4, 5, 1).reshape(len(x), 63, 9 * 192)[:, MAPC].flatten(0, 1)
        gsub['own'].append(y[mk_c].half()); gsub['nb'].append(nb[mk_c].half())
        gsub['celly'].append(torch.stack([tc, (mc == 1).long(), (mc == 2).long(), (mc == 3).long(), (mc == 4).long()], 1)[mk_c])
        gsub['p'].append(pc[mk_c])
        print(json.dumps({'roots_done': min(R, i + 32), 'frames': res['frames'], 'seconds': round(time.time() - t0)}), flush=True)

    # A2 static composition: select lambda on val, report test R^2
    res['A2_static_R2'] = {}
    for u in sets:
        fit, val, test = stats[u][0], stats[u][1], stats[u][2]
        def sse(st, W):
            return st['yy'] - 2 * float((W * st['xy']).sum()) + float((W * (st['xx'] @ W)).sum())
        best = min(LAMS, key=lambda lam: sse(val, ridge_solve(fit, lam)))
        W = ridge_solve(fit, best)
        mu = fit['ysum'] / fit['n']
        sst = test['yy'] - 2 * float((test['ysum'] * mu).sum()) + test['n'] * float((mu ** 2).sum())
        res['A2_static_R2'][f'S{u}'] = {'R2_test': round(1 - sse(test, W) / sst, 4), 'lambda': best, 'dims': int(fit['xx'].shape[0])}
    print(json.dumps({'A2_static_R2': res['A2_static_R2']}), flush=True)
    res['A2_hud_player_change'] = {k: {'transitions': v[1], 'mean_dx2': round(v[0] / max(v[1], 1), 3)} for k, v in hudch.items()}
    print(json.dumps({'A2_hud_player_change': res['A2_hud_player_change']}), flush=True)
    if reenc['d']:
        fit4 = stats[4][0]
        W4 = ridge_solve(fit4, res['A2_static_R2']['S4']['lambda']).float()
        Xb, Xa, D, Pq = torch.cat(reenc['Xb']).float(), torch.cat(reenc['Xa']).float(), torch.cat(reenc['d']).float(), torch.cat(reenc['p'])
        Dhat = (Xb - Xa) @ W4
        te = Pq == 2
        res['A2_scroll_reencoding'] = {'cells': int(len(D)), 'cells_test': int(te.sum()),
                                      'mean_dx2_test': round(float((D[te] ** 2).sum(-1).mean()), 3),
                                      'explained_by_S4_state_position_global': round(float(1 - ((D[te] - Dhat[te]) ** 2).sum() / (D[te] ** 2).sum()), 4)}
        print(json.dumps({'A2_scroll_reencoding': res['A2_scroll_reencoding']}), flush=True)
    # MLP upper bound on S4 inputs
    X = torch.cat(sub['x']).float(); Y = torch.cat(sub['y']).float(); P = torch.cat(sub['p'])
    torch.manual_seed(0)
    mlp = torch.nn.Sequential(torch.nn.Linear(X.shape[1], 512), torch.nn.GELU(), torch.nn.Linear(512, 512), torch.nn.GELU(), torch.nn.Linear(512, 192))
    opt = torch.optim.Adam(mlp.parameters(), 1e-3)
    tr = torch.where(P == 0)[0]
    for ep in range(6):
        perm = tr[torch.randperm(len(tr))]
        for j in range(0, len(perm), 1024):
            bi = perm[j:j + 1024]
            loss = ((mlp(X[bi]) - Y[bi]) ** 2).sum(-1).mean()
            opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        te = P == 2
        pred = torch.cat([mlp(X[te][j:j + 8192]) for j in range(0, int(te.sum()), 8192)])
        mu = Y[P == 0].mean(0)
        res['A2_static_R2']['MLP_S4'] = {'R2_test': round(float(1 - ((pred - Y[te]) ** 2).sum() / ((Y[te] - mu) ** 2).sum()), 4),
                                          'cells_fit': int(len(tr)), 'cells_test': int(te.sum())}
    print(json.dumps({'MLP_S4': res['A2_static_R2']['MLP_S4']}), flush=True)
    # A2 change composition
    names = ('own_state_changed', 'source_was_player', 'neighbour_changed', 'sleep_changed', 'light_changed', 'no_state_change')
    comp = {}
    for p, pname in ((None, 'all'), (2, 'test')):
        sel = {k: v for k, v in change.items() if p is None or k[2] == p}
        tot = sum(v[0] for v in sel.values()); tot1 = sum(v[2] for v in sel.values())
        rows = {}
        for cv, nm in enumerate(names):
            for sv, sn in ((0, 'awake'), (1, 'asleep')):
                vv = [v for k, v in sel.items() if k[0] == cv and k[1] == sv]
                e, n, l1 = sum(v[0] for v in vv), sum(v[1] for v in vv), sum(v[2] for v in vv)
                rows[f'{nm}/{sn}'] = {'L2_energy_share': round(e / tot, 4), 'L1_mass_share': round(l1 / tot1, 4), 'cells': n,
                                      'mean_dx2': round(e / max(n, 1), 3), 'mean_abs_dx': round(l1 / max(n, 1), 3)}
        comp[pname] = rows
    res['A2_change'] = comp
    ub = res.pop('unchanged_by_distance')
    res['A2_unchanged_by_distance'] = {f"dist{min(int(j) // 2, 5)}{'+' if int(j) // 2 == 5 else ''}_{'scroll' if int(j) % 2 else 'still'}":
        {'cells': v[2], 'mean_dx2': round(v[0] / max(v[2], 1), 3), 'mean_abs_dx': round(v[1] / max(v[2], 1), 3)} for j, v in ub.items() if v[2]}
    print(json.dumps({'A2_unchanged_by_distance': res['A2_unchanged_by_distance']}), flush=True)
    print(json.dumps({'A2_change_all': comp['all']}), flush=True)
    # A3 currency
    mu = cls_sum / cls_n.clamp(min=1)[:, None]
    present = torch.where(cls_n > 200)[0]
    dist = ((mu[present, None] - mu[None, present]) ** 2).sum(-1)
    res['A3'] = {'classes': present.tolist(), 'class_counts': cls_n[present].long().tolist(),
                 'mean_dist2': dist.tolist(),
                 'min_offdiag_dist2': float(dist[~torch.eye(len(present), dtype=torch.bool)].min()),
                 'median_offdiag_dist2': float(dist[~torch.eye(len(present), dtype=torch.bool)].median())}
    # A1 probes (ridge on the cached subsamples; MLP for cells)
    def ridge_fit(Xa, Ya, pa, kind):
        Xf, Yf, Xv, Yv, Xt, Yt = Xa[pa == 0], Ya[pa == 0], Xa[pa == 1], Ya[pa == 1], Xa[pa == 2], Ya[pa == 2]
        m_, s_ = Xf.mean(0), Xf.std(0).clamp(min=1e-6)
        z = lambda q: torch.cat([(q - m_) / s_, torch.ones(len(q), 1)], 1)
        A = z(Xf)
        AtA, AtB = (A.T @ A).double(), (A.T @ Yf.float()).double()                                   # float32 products, float64 solve
        best = None
        for lam in (1e-3, 1e-2, 1e-1, 1, 10, 100):
            Wt = torch.linalg.solve(AtA + lam * len(A) * torch.eye(A.shape[1], dtype=torch.float64), AtB).float()
            e = float(((z(Xv) @ Wt - Yv.float()) ** 2).mean())
            if best is None or e < best[0]:
                best = (e, Wt)
        return (z(Xt) @ best[1]).float(), Yt.float()
    a1 = {}
    hud, hudy, gp = torch.cat(gsub['hud']).float(), torch.cat(gsub['hudy']), torch.cat(gsub['gp'])
    ph, yh = ridge_fit(hud, hudy, gp, 'reg')
    for j, nm in enumerate(('health', 'food', 'drink', 'energy')):
        a1[nm] = {'R2': round(float(1 - ((ph[:, j] - yh[:, j]) ** 2).sum() / ((yh[:, j] - yh[:, j].mean()) ** 2).sum()), 4),
                  'exact_after_rounding': round(float((ph[:, j].round() == yh[:, j]).float().mean()), 4)}
    pl, ply = torch.cat(gsub['pl']).float(), torch.cat(gsub['ply'])
    pp_, yp_ = ridge_fit(hud, ply[:, 2:], gp, 'reg')
    a1['inventory_from_hud'] = {'R2_mean': round(float((1 - ((pp_ - yp_) ** 2).sum(0) / ((yp_ - yp_.mean(0)) ** 2).sum(0).clamp(min=1e-9)).mean()), 4),
                                'exact_after_rounding': round(float((pp_.round() == yp_).float().mean()), 4)}
    pf, yf = ridge_fit(pl, F.one_hot(ply[:, 0].long(), 4).float(), gp, 'cls')
    a1['facing_from_player_token'] = round(float((pf.argmax(1) == yf.argmax(1)).float().mean()), 4)
    ps_, ys_ = ridge_fit(pl, ply[:, 1:2], gp, 'reg')
    a1['sleep_from_player_token_acc'] = round(float(((ps_ > 0.5) == (ys_ > 0.5)).float().mean()), 4)
    pm, ym = ridge_fit(torch.cat(gsub['mean']).float(), torch.cat(gsub['meany']), gp, 'reg')
    a1['light_from_mean_map_token_R2'] = round(float(1 - ((pm - ym) ** 2).sum() / ((ym - ym.mean()) ** 2).sum()), 4)
    own, nbt, cy, cp = torch.cat(gsub['own']).float(), torch.cat(gsub['nb']).float(), torch.cat(gsub['celly']), torch.cat(gsub['p'])
    for feat_name, feat in (('own_token', own), ('3x3_tokens', nbt)):
        pt, yt = ridge_fit(feat, F.one_hot(cy[:, 0], 17).float(), cp, 'cls')
        r = {'tile_acc': round(float((pt.argmax(1) == yt.argmax(1)).float().mean()), 4)}
        for j, mn in enumerate(('zombie', 'skeleton', 'cow', 'arrow')):
            pm_, ym_ = ridge_fit(feat, cy[:, 1 + j:2 + j].float(), cp, 'cls')
            from compound import auc
            r[f'{mn}_auc'] = round(auc(pm_[:, 0], ym_[:, 0] > 0.5), 4)
            r[f'{mn}_positives_test'] = int((ym_ > 0.5).sum())
        a1[feat_name] = r
    res['A1'] = a1
    res['seconds'] = round(time.time() - t0)
    print(json.dumps({'A1': a1, 'A3_min_median': [res['A3']['min_offdiag_dist2'], res['A3']['median_offdiag_dist2']]}), flush=True)
    name = 'encoder_diag.json' if '--roots' not in args else 'encoder_diag_test.json'
    (OUT / name).write_text(json.dumps(res, indent=1))


if __name__ == '__main__':
    sys.path.insert(0, str(ROOT / 'artifacts/experiments/20260926_diagnosis'))
    sys.path.insert(0, str(ROOT / 'artifacts/experiments/20260921_readout_ladder'))
    main()
