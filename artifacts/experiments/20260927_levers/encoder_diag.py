"""10-10 PROGRAM, Part II-A (encoder; prediction P-A), CPU only. What does a frozen token carry, and how much of it -- and of its
frame-to-frame change -- is the game state of its own cell?
v2 (10:1x, after the first full run was OOM-killed at 09:28 holding 12.3 GB; its printed results are in the NOTEBOOK): one S4 design
per chunk (the nested sets are prefix blocks of its statistics), float32 chunk products with float64 accumulation, smaller subsamples;
and, after reading craftax_classic/renderer.py (night: every map pixel blended with fresh uniform static from state_rng, intensity
2 (0.5 - daylight) x (1 - exp(-0.5 r^2 / 0.25)) radial from the player), a day / night split everywhere plus a direct noise floor.
Data: every diagnosis frame with its simulator state (1,002 roots, 143 episodes): the root frame (teval cache) and the five 16-step
futures (stochdiag cache), alive frames only. Split = teval's (70% / 30% of episode seeds; lambda on a held-out fifth of the train
seeds). Map cells only (62; the player cell 31 handled separately).
Cell state: tile class (17), mob category (none / zombie / skeleton / cow / arrow, first present in that order); global: light level
(10 bins), sleeping; position (62); 4-neighbour and diagonal-neighbour tile classes (+ out of view) and mob categories.
  A2 static composition: held R^2 of the cell token (192 dims pooled) from nested linear one-hot designs
       S0 own state (17 x 5) | S1 + position | S2 + global (light x sleep, tile x sleep, tile x light) | S3 + 4 neighbours |
       S4 + diagonals; and an MLP on S4's inputs (upper bound for "what the visible 3 x 3 + global state explains")
  A2 change composition: consecutive alive frames (root -> 1, k -> k+1, each future), scroll from the TRUE tile maps (fraction of
       mismatching tiles over the overlap; ties by scroll.estimate); every map cell at t+1 with a source at t (entering cells excluded),
       dx = x_{t+1}(cell) - x_t(source). Sum |dx|^2 (L2 currency) and sum |dx| (L1 currency) by cause, exclusive in this order: own
       state changed | the source was the player cell (sprite leaves the tile) | a 4-neighbour's state changed | sleep changed | light
       changed by > 0.02 | nothing in the state changed; each split asleep / awake and night (light < 0.5, static rendered) / day.
       Unchanged cells (last cause) by Chebyshev ring around the player x night x scroll.
  A2 noise floor: for every root and depth where all five futures are alive AND their full visible states are identical, the
       per-token unbiased variance over the five futures: what differs is only rendering randomness (the night static) -- by ring x
       night for map tokens, and for the HUD and player tokens.
  A2 scroll re-encoding: unchanged cells on scroll steps: share of dx explained by S4's state / position / global / neighbour terms.
  A2 HUD / player token change at unchanged HUD values (+ inventory) / facing + sleep, scroll vs still, night vs day.
  A3 currency: per tile class (no mob, awake), class mean (fit) and the pairwise squared distances between class means.
  A1 information ceilings: held probes of tile class and each mob from the own token / the 3 x 3 tokens (ridge), HUD values and
       inventory from the 18 HUD tokens, facing and sleep from the player token, light from the mean map token.
--encoder raw|ldad1|ldad10|tc: the same analyses on teval's token cache of that encoder, root + the factual future (sample 0) only (no
       noise floor: one future), so encoders are compared on identical frames.
Usage: encoder_diag.py [--roots N (test)] [--encoder NAME]  -> artifacts/eda/health_chain_v1/encoder_diag[_NAME].json
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
RING = torch.tensor([max(abs(c // 9 - 3), abs(c % 9 - 4)) for c in range(63)])                     # Chebyshev ring around (3, 4)
SET_DIMS = (85, 147, 371, 463, 555)                                                               # S0..S4 = prefixes of the S4 design


def mob_cat(mobs):
    """[..., 7] mob channels (zombie, cow, skeleton, arrows x4) -> 0 none, 1 zombie, 2 skeleton, 3 cow, 4 arrow (first present)"""
    m = mobs > 0
    cat = torch.zeros(m.shape[:-1], dtype=torch.long)
    for val, ch in ((4, m[..., 3:].any(-1)), (3, m[..., 1]), (2, m[..., 2]), (1, m[..., 0])):
        cat = torch.where(ch, torch.full_like(cat, val), cat)
    return cat


def frame_state(vis):
    v = vis.float()
    return (v[:, :1071].view(-1, 7, 9, 17).argmax(-1), mob_cat(v[:, 1071:1512].view(-1, 7, 9, 7)), v[:, 1521], v[:, 1520] > 0.5)


def neighbour(grid, dr, dc, fill):
    """[n,7,9] -> out[r, c] = grid[r + dr, c + dc], `fill` off-view"""
    out = torch.full_like(grid, fill)
    rs, re = max(0, -dr), 7 - max(0, dr)
    cs, ce = max(0, -dc), 9 - max(0, dc)
    out[:, rs:re, cs:ce] = grid[:, rs + dr:re + dr, cs + dc:ce + dc]
    return out


def design(tiles, mobc, light, sleep):
    """S4 one-hot design per map cell (MAPC order): [n * 62, 555] float32; S0..S3 are its column prefixes (SET_DIMS)"""
    n = len(tiles)
    lb = (light.clamp(0, 0.999) * 10).long()
    cell = lambda g: g.flatten(1)[:, MAPC]
    own = cell(tiles) * 5 + cell(mobc)
    sl = sleep.long()[:, None].expand(n, 62); lbe = lb[:, None].expand(n, 62)
    parts = [F.one_hot(own, 85), F.one_hot(torch.arange(62).expand(n, 62), 62),
             F.one_hot(lbe * 2 + sl, 20), F.one_hot(cell(tiles) * 2 + sl, 34), F.one_hot(cell(tiles) * 10 + lbe, 170)]
    for dr, dc in NB4 + NBD:
        parts += [F.one_hot(cell(neighbour(tiles, dr, dc, 17)), 18), F.one_hot(cell(neighbour(mobc, dr, dc, 0)), 5)]
    return torch.cat(parts, -1).flatten(0, 1).float()


def ridge_solve(xx, xy, n, lam):
    return torch.linalg.solve(xx + lam * n * torch.eye(xx.shape[0], dtype=xx.dtype), xy)


def ridge_fit(Xa, Ya, pa):
    """standardized ridge (teval.ridge's lambda grid / selection on part 1), predictions on part 2"""
    Xf, Yf, Xv, Yv, Xt, Yt = Xa[pa == 0], Ya[pa == 0], Xa[pa == 1], Ya[pa == 1], Xa[pa == 2], Ya[pa == 2]
    m_, s_ = Xf.mean(0), Xf.std(0).clamp(min=1e-6)
    z = lambda q: torch.cat([(q - m_) / s_, torch.ones(len(q), 1)], 1)
    A = z(Xf)
    AtA, AtB = (A.T @ A).double(), (A.T @ Yf.float()).double()
    del A
    best = None
    for lam in (1e-3, 1e-2, 1e-1, 1, 10, 100):
        Wt = torch.linalg.solve(AtA + lam * len(Xf) * torch.eye(AtA.shape[0], dtype=torch.float64), AtB).float()
        e = float(((z(Xv) @ Wt - Yv.float()) ** 2).mean())
        if best is None or e < best[0]:
            best = (e, Wt)
    return (z(Xt) @ best[1]).float(), Yt.float()


def main():
    import teval as T
    import stochdiag as SD
    from scroll import estimate
    from compound import auc
    torch.set_num_threads(6)
    t0 = time.time()
    args = sys.argv[1:]
    meta, train_roots, train_seeds = T.split()
    R = int(args[args.index('--roots') + 1]) if '--roots' in args else len(meta['seed'])
    val_roots = torch.isin(meta['seed'], train_seeds[:len(train_seeds) // 5])
    part = torch.where(~train_roots, 2, torch.where(val_roots, 1, 0))[:R]                           # 0 fit, 1 val, 2 test
    enc = args[args.index('--encoder') + 1] if '--encoder' in args else None
    cache = T.build_cache(enc or 'raw', torch.device('cpu'))
    fut5 = SD.token_cache(torch.device('cpu'))[0] if enc is None else cache['fut'][:, None]          # [R, S, 16, 81, 192]
    S5 = fut5.shape[1]
    rootv, futv = meta['root_visible'][:R], meta['future_visible'][:R]
    alive = ~meta['future_dead'][:R].cumsum(2).bool()                                               # [R,5,16]

    def frames(i, j):
        toks, vis, prt, pairs = [], [], [], []
        base = 0
        for r in range(i, j):
            toks.append(cache['ctx'][r, -1:].float()); vis.append(rootv[r:r + 1]); prt.append(part[r:r + 1])
            idx0 = base; base += 1
            for s in range(S5):
                prev = idx0
                for k in range(16):
                    if not alive[r, s, k]:
                        break
                    toks.append(fut5[r, s, k:k + 1].float()); vis.append(futv[r, s, k:k + 1]); prt.append(part[r:r + 1])
                    pairs.append((prev, base)); prev = base; base += 1
        return torch.cat(toks), torch.cat(vis), torch.cat(prt), torch.tensor(pairs)

    res = {'frames': 0, 'encoder': enc or 'raw (five futures)', 'shift_counts': [0] * 5, 'shift_agree_with_token_estimate': [0, 0]}
    st = {p: {'xx': torch.zeros(555, 555, dtype=torch.float64), 'xy': torch.zeros(555, 192, dtype=torch.float64), 'yy': 0.0,
              'ysum': torch.zeros(192, dtype=torch.float64), 'n': 0} for p in range(3)}
    NK = 6 * 2 * 2 * 3                                                                             # cause x asleep x night x part
    chg = torch.zeros(NK, 3, dtype=torch.float64)                                                  # [energy L2, mass L1, cells]
    ring = torch.zeros(5 * 2 * 2, 3, dtype=torch.float64)                                          # unchanged: ring x night x scroll
    hudch = torch.zeros(2 * 2 * 2 * 2, 2, dtype=torch.float64)                                     # token(hud/player) x same x scroll x night
    cls_sum = torch.zeros(17, 192, dtype=torch.float64); cls_n = torch.zeros(17, dtype=torch.float64)
    sub = {'x': [], 'y': [], 'p': []}
    reenc = {'Xb': [], 'Xa': [], 'd': [], 'p': []}
    gsub = {'hud': [], 'hudy': [], 'gp': [], 'pl': [], 'ply': [], 'mean': [], 'meany': [], 'own': [], 'nb': [], 'celly': [], 'p': []}
    gen = torch.Generator().manual_seed(20261010)
    for i in range(0, R, 32):
        x, vis, prt, pairs = frames(i, min(R, i + 32))
        tiles, mobc, light, sleep = frame_state(vis)
        m = len(x)
        y = x[:, MAPC].flatten(0, 1)                                                                # [m*62, 192] float32
        pc = prt[:, None].expand(-1, 62).flatten()
        res['frames'] += m
        X = design(tiles, mobc, light, sleep)                                                      # [m*62, 555]
        for p in range(3):
            mk = pc == p
            Xp, yp = X[mk], y[mk]
            st[p]['xx'] += (Xp.T @ Xp).double(); st[p]['xy'] += (Xp.T @ yp).double()
            st[p]['yy'] += float((yp.double() ** 2).sum()); st[p]['ysum'] += yp.double().sum(0); st[p]['n'] += len(yp)
        tc, mc = tiles.flatten(1)[:, MAPC].flatten(), mobc.flatten(1)[:, MAPC].flatten()
        clean = (mc == 0) & (~sleep)[:, None].expand(-1, 62).flatten() & (pc == 0)
        cls_sum.index_add_(0, tc[clean], y[clean].double()); cls_n.index_add_(0, tc[clean], torch.ones(int(clean.sum()), dtype=torch.float64))
        mk_sub = torch.rand(len(y), generator=gen) < 0.05
        sub['x'].append(X[mk_sub].half()); sub['y'].append(y[mk_sub].half()); sub['p'].append(pc[mk_sub])
        # change composition
        a, b = pairs[:, 0], pairs[:, 1]
        ta, tb = tiles[a], tiles[b]
        frac = []
        for dr, dc in SHIFTS:
            src = neighbour(ta, dr, dc, -1)
            ov = src >= 0
            frac.append(((src != tb) & ov).flatten(1).sum(1).float() / ov.flatten(1).sum(1).float())
        est = estimate(x[a], x[b])
        sh = (torch.stack(frac, 1) + 1e-3 * (torch.arange(5)[None] != est[:, None]).float()).argmin(1)
        for si in range(5):
            res['shift_counts'][si] += int((sh == si).sum())
        res['shift_agree_with_token_estimate'][0] += int((sh == est).sum()); res['shift_agree_with_token_estimate'][1] += len(sh)
        night = light < 0.5
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
                src_is_player[3 - dr, 4 - dc] = True
            own_ch = (neighbour(tiles[aa], dr, dc, -1) != tiles[bb]) | (neighbour(mobc[aa], dr, dc, -1) != mobc[bb])
            nb_ch = torch.zeros_like(own_ch)
            for ndr, ndc in NB4:
                nb_ch |= neighbour(own_ch & has_src[None], ndr, ndc, 0).bool()
            dxv = xb - src_x
            dx2, dx1 = (dxv ** 2).sum(-1).double(), dxv.abs().sum(-1).double()
            cause = torch.full((n_, 7, 9), 5, dtype=torch.long)
            for val, msk in ((4, ((light[bb] - light[aa]).abs() > 0.02)[:, None, None].expand(-1, 7, 9)),
                             (3, (sleep[bb] != sleep[aa])[:, None, None].expand(-1, 7, 9)), (2, nb_ch),
                             (1, src_is_player[None].expand(n_, -1, -1)), (0, own_ch)):
                cause = torch.where(msk, torch.full_like(cause, val), cause)
            valid = has_src[None].expand(n_, -1, -1).clone(); valid[:, 3, 4] = False
            bc = lambda t3: t3[:, None, None].expand(-1, 7, 9)
            k_ = ((cause * 2 + bc(sleep[bb].long())) * 2 + bc(night[bb].long())) * 3 + bc(prt[bb])
            for col, w in ((0, dx2), (1, dx1), (2, torch.ones_like(dx2))):
                chg[:, col] += torch.bincount(k_[valid], weights=w[valid], minlength=NK)
            u = valid & (cause == 5)
            rk = ((RING.view(7, 9)[None].expand(n_, -1, -1) * 2 + bc(night[bb].long())) * 2 + int(si != 0))
            for col, w in ((0, dx2), (1, dx1), (2, torch.ones_like(dx2))):
                ring[:, col] += torch.bincount(rk[u], weights=w[u], minlength=20)
            if si != 0:                                                                                  # scroll re-encoding sample
                ii, rr, cc = torch.where(u)
                keep = torch.rand(len(ii), generator=gen) < 0.15
                ii, rr, cc = ii[keep], rr[keep], cc[keep]
                if len(ii):
                    col_of = {c: j for j, c in enumerate(MAPC)}
                    jb = torch.tensor([col_of[int(r_) * 9 + int(c_)] for r_, c_ in zip(rr, cc)])
                    ja = torch.tensor([col_of[(int(r_) + dr) * 9 + int(c_) + dc] for r_, c_ in zip(rr, cc)])
                    reenc['Xb'].append(X[bb[ii] * 62 + jb].half()); reenc['Xa'].append(X[aa[ii] * 62 + ja].half())
                    reenc['d'].append(dxv[ii, rr, cc].half()); reenc['p'].append(prt[bb][ii])
        hud_same = (vis[a, 1512:1516] == vis[b, 1512:1516]).all(1) & (vis[a, 1522:1534] == vis[b, 1522:1534]).all(1)
        fac_same = (vis[a, 1516:1520] == vis[b, 1516:1520]).all(1) & (vis[a, 1520] == vis[b, 1520])
        hud2 = ((x[b, 63:81] - x[a, 63:81]) ** 2).sum((-1, -2)).double()
        pl2 = ((x[b, 31] - x[a, 31]) ** 2).sum(-1).double()
        for ti, (val, same) in enumerate(((hud2, hud_same), (pl2, fac_same))):
            kk = ((ti * 2 + same.long()) * 2 + (sh != 0).long()) * 2 + night[b].long()
            hudch[:, 0] += torch.bincount(kk, weights=val, minlength=16); hudch[:, 1] += torch.bincount(kk, minlength=16)
        # A1 subsamples
        fm = torch.rand(m, generator=gen) < 0.3
        gsub['hud'].append(x[fm, 63:81].flatten(1).half()); gsub['hudy'].append(torch.cat([vis[fm, 1512:1516], vis[fm, 1522:1534]], 1).float() * 9)
        gsub['gp'].append(prt[fm])
        gsub['pl'].append(x[fm, 31].half()); gsub['ply'].append(torch.cat([vis[fm, 1516:1520].float().argmax(-1, keepdim=True).float(), vis[fm, 1520:1521].float()], 1))
        gsub['mean'].append(x[fm][:, MAPC].mean(1).half()); gsub['meany'].append(light[fm][:, None])
        ci = torch.where(torch.rand(len(y), generator=gen) < 0.015)[0]
        fi, cj = ci // 62, torch.tensor(MAPC)[ci % 62]
        pad = F.pad(x[:, :63].view(m, 7, 9, 192), (0, 0, 1, 1, 1, 1))                                   # [m, 9, 11, 192]
        rr_, cc_ = cj // 9, cj % 9
        nbt = torch.stack([pad[fi, rr_ + 1 + dr, cc_ + 1 + dc] for dr in (-1, 0, 1) for dc in (-1, 0, 1)], 1).flatten(1)
        gsub['own'].append(y[ci].half()); gsub['nb'].append(nbt.half())
        gsub['celly'].append(torch.stack([tc[ci], (mc[ci] == 1).long(), (mc[ci] == 2).long(), (mc[ci] == 3).long(), (mc[ci] == 4).long()], 1))
        gsub['p'].append(pc[ci])
        del X, y, x
        print(json.dumps({'roots_done': min(R, i + 32), 'frames': res['frames'], 'seconds': round(time.time() - t0)}), flush=True)

    # noise floor: identical visible states across the five futures
    if S5 == 5:
        nf = torch.zeros(5 * 2 + 2 * 2, 2, dtype=torch.float64)                                     # map rings x night, hud x night, player x night
        n_ident = 0; n_cases = 0
        for i in range(0, R, 32):
            for r in range(i, min(R, i + 32)):
                for k in range(16):
                    if not alive[r, :, k].all():
                        continue
                    n_cases += 1
                    v = futv[r, :, k]
                    if not (v == v[:1]).all():
                        continue
                    n_ident += 1
                    xk = fut5[r, :, k].float()                                                           # [5,81,192]
                    s2 = ((xk - xk.mean(0)) ** 2).sum((0, -1)) / 4                                      # [81]
                    nt = int(v[0, 1521] < 0.5)
                    for c in MAPC:
                        nf[RING[c] * 2 + nt, 0] += float(s2[c]); nf[RING[c] * 2 + nt, 1] += 1
                    nf[10 + nt, 0] += float(s2[63:81].sum()); nf[10 + nt, 1] += 1
                    nf[12 + nt, 0] += float(s2[31]); nf[12 + nt, 1] += 1
        res['A2_noise_floor_identical_states'] = {'cases_all_alive': n_cases, 'cases_identical_state': n_ident}
        for ring_ in range(5):
            for nt, nn in ((0, 'day'), (1, 'night')):
                e, c = nf[ring_ * 2 + nt]
                if c:
                    res['A2_noise_floor_identical_states'][f'map_ring{ring_}_{nn}'] = {'tokens': int(c), 'var_per_token': round(float(e / c), 4)}
        for j, nm in ((10, 'hud18_sum'), (12, 'player')):
            for nt, nn in ((0, 'day'), (1, 'night')):
                e, c = nf[j + nt]
                if c:
                    res['A2_noise_floor_identical_states'][f'{nm}_{nn}'] = {'frames': int(c), 'var': round(float(e / c), 4)}
        print(json.dumps({'A2_noise_floor_identical_states': res['A2_noise_floor_identical_states']}), flush=True)

    # A2 static composition (prefix blocks of the S4 statistics)
    res['A2_static_R2'] = {}
    mu = st[0]['ysum'] / st[0]['n']
    sst = st[2]['yy'] - 2 * float((st[2]['ysum'] * mu).sum()) + st[2]['n'] * float((mu ** 2).sum())
    for u, dmax in enumerate(SET_DIMS):
        blk = lambda s_: (s_['xx'][:dmax, :dmax], s_['xy'][:dmax])
        sse = lambda s_, W: s_['yy'] - 2 * float((W * blk(s_)[1]).sum()) + float((W * (blk(s_)[0] @ W)).sum())
        best = min(LAMS, key=lambda lam: sse(st[1], ridge_solve(*blk(st[0]), st[0]['n'], lam)))
        W = ridge_solve(*blk(st[0]), st[0]['n'], best)
        res['A2_static_R2'][f'S{u}'] = {'R2_test': round(1 - sse(st[2], W) / sst, 4), 'lambda': best, 'dims': dmax}
        if u == 4:
            W4 = W.float()
    print(json.dumps({'A2_static_R2': res['A2_static_R2']}), flush=True)
    Xs, Ys, Ps = torch.cat(sub['x']).float(), torch.cat(sub['y']).float(), torch.cat(sub['p'])
    torch.manual_seed(0)
    mlp = torch.nn.Sequential(torch.nn.Linear(555, 512), torch.nn.GELU(), torch.nn.Linear(512, 512), torch.nn.GELU(), torch.nn.Linear(512, 192))
    opt = torch.optim.Adam(mlp.parameters(), 1e-3)
    tr = torch.where(Ps == 0)[0]
    for ep in range(6):
        perm = tr[torch.randperm(len(tr))]
        for j in range(0, len(perm), 1024):
            bi = perm[j:j + 1024]
            loss = ((mlp(Xs[bi]) - Ys[bi]) ** 2).sum(-1).mean()
            opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        te = torch.where(Ps == 2)[0]
        pred = torch.cat([mlp(Xs[te[j:j + 8192]]) for j in range(0, len(te), 8192)])
        res['A2_static_R2']['MLP_S4'] = {'R2_test': round(float(1 - ((pred - Ys[te]) ** 2).sum() / ((Ys[te] - Ys[tr].mean(0)) ** 2).sum()), 4),
                                          'cells_fit': int(len(tr)), 'cells_test': int(len(te))}
    del Xs, Ys, Ps, sub
    print(json.dumps({'MLP_S4': res['A2_static_R2']['MLP_S4']}), flush=True)
    # A2 change composition
    names = ('own_state_changed', 'source_was_player', 'neighbour_changed', 'sleep_changed', 'light_changed', 'no_state_change')
    comp = {}
    for pname, parts_ in (('all', (0, 1, 2)), ('test', (2,))):
        rows = {}
        idx = [k for k in range(NK) if k % 3 in parts_]
        tot2, tot1 = float(chg[idx, 0].sum()), float(chg[idx, 1].sum())
        for cv, nm in enumerate(names):
            for sv, sn in ((0, 'awake'), (1, 'asleep')):
                for nv, nn in ((0, 'day'), (1, 'night')):
                    kk = [((cv * 2 + sv) * 2 + nv) * 3 + p for p in parts_]
                    e2, e1, n = (float(chg[kk, c].sum()) for c in range(3))
                    if n:
                        rows[f'{nm}/{sn}/{nn}'] = {'L2_share': round(e2 / tot2, 4), 'L1_share': round(e1 / tot1, 4), 'cells': int(n),
                                                   'mean_dx2': round(e2 / n, 3), 'mean_abs_dx': round(e1 / n, 3)}
        comp[pname] = rows
    res['A2_change'] = comp
    res['A2_unchanged_by_ring'] = {}
    for rg in range(5):
        for nv, nn in ((0, 'day'), (1, 'night')):
            for sc, sn in ((0, 'still'), (1, 'scroll')):
                e2, e1, n = (float(v) for v in ring[(rg * 2 + nv) * 2 + sc])
                if n:
                    res['A2_unchanged_by_ring'][f'ring{rg}_{nn}_{sn}'] = {'cells': int(n), 'mean_dx2': round(e2 / n, 3), 'mean_abs_dx': round(e1 / n, 3)}
    res['A2_hud_player_change'] = {}
    for ti, tn in ((0, 'hud18'), (1, 'player')):
        for sm, smn in ((0, 'state_changed'), (1, 'state_same')):
            for sc, sn in ((0, 'still'), (1, 'scroll')):
                for nv, nn in ((0, 'day'), (1, 'night')):
                    e, n = (float(v) for v in hudch[((ti * 2 + sm) * 2 + sc) * 2 + nv])
                    if n:
                        res['A2_hud_player_change'][f'{tn}/{smn}/{sn}/{nn}'] = {'transitions': int(n), 'mean_dx2': round(e / n, 3)}
    print(json.dumps({'A2_change_all': comp['all'], 'A2_unchanged_by_ring': res['A2_unchanged_by_ring'],
                      'A2_hud_player_change': res['A2_hud_player_change']}), flush=True)
    if reenc['d']:
        Xb, Xa, D, Pq = torch.cat(reenc['Xb']).float(), torch.cat(reenc['Xa']).float(), torch.cat(reenc['d']).float(), torch.cat(reenc['p'])
        Dhat = (Xb - Xa) @ W4
        te = Pq == 2
        res['A2_scroll_reencoding'] = {'cells': int(len(D)), 'cells_test': int(te.sum()), 'mean_dx2_test': round(float((D[te] ** 2).sum(-1).mean()), 3),
                                      'explained_by_S4': round(float(1 - ((D[te] - Dhat[te]) ** 2).sum() / (D[te] ** 2).sum()), 4)}
        print(json.dumps({'A2_scroll_reencoding': res['A2_scroll_reencoding']}), flush=True)
    # A3
    mu_c = cls_sum / cls_n.clamp(min=1)[:, None]
    present = torch.where(cls_n > 200)[0]
    dist = ((mu_c[present, None] - mu_c[None, present]) ** 2).sum(-1)
    off = dist[~torch.eye(len(present), dtype=torch.bool)]
    res['A3'] = {'classes': present.tolist(), 'class_counts': cls_n[present].long().tolist(), 'mean_dist2': dist.tolist(),
                 'min_offdiag_dist2': float(off.min()), 'median_offdiag_dist2': float(off.median())}
    # A1
    a1 = {}
    gp = torch.cat(gsub['gp'])
    ph, yh = ridge_fit(torch.cat(gsub['hud']).float(), torch.cat(gsub['hudy']), gp)
    r2 = lambda p_, y_: float(1 - ((p_ - y_) ** 2).sum() / ((y_ - y_.mean()) ** 2).sum().clamp_min(1e-9))
    for j, nm in enumerate(('health', 'food', 'drink', 'energy')):
        a1[nm] = {'R2': round(r2(ph[:, j], yh[:, j]), 4), 'exact_after_rounding': round(float((ph[:, j].round() == yh[:, j]).float().mean()), 4)}
    inv = [r2(ph[:, 4 + j], yh[:, 4 + j]) for j in range(12) if yh[:, 4 + j].var() > 0]
    a1['inventory_R2_mean'] = round(sum(inv) / max(len(inv), 1), 4)
    a1['inventory_exact_after_rounding'] = round(float((ph[:, 4:].round() == yh[:, 4:]).float().mean()), 4)
    pl, ply = torch.cat(gsub['pl']).float(), torch.cat(gsub['ply'])
    pf, yf = ridge_fit(pl, F.one_hot(ply[:, 0].long(), 4).float(), gp)
    a1['facing_from_player_token'] = round(float((pf.argmax(1) == yf.argmax(1)).float().mean()), 4)
    ps_, ys_ = ridge_fit(pl, ply[:, 1:2], gp)
    a1['sleep_from_player_token_acc'] = round(float(((ps_ > 0.5) == (ys_ > 0.5)).float().mean()), 4)
    pm, ym = ridge_fit(torch.cat(gsub['mean']).float(), torch.cat(gsub['meany']), gp)
    a1['light_from_mean_map_token_R2'] = round(r2(pm, ym), 4)
    cy, cp = torch.cat(gsub['celly']), torch.cat(gsub['p'])
    for feat_name, key in (('own_token', 'own'), ('3x3_tokens', 'nb')):
        feat = torch.cat(gsub[key]).float()
        pt, yt = ridge_fit(feat, F.one_hot(cy[:, 0], 17).float(), cp)
        r = {'tile_acc': round(float((pt.argmax(1) == yt.argmax(1)).float().mean()), 4)}
        for j, mn in enumerate(('zombie', 'skeleton', 'cow', 'arrow')):
            pm_, ym_ = ridge_fit(feat, cy[:, 1 + j:2 + j].float(), cp)
            r[f'{mn}_auc'] = round(auc(pm_[:, 0], ym_[:, 0] > 0.5), 4) if 0 < (ym_ > 0.5).sum() < len(ym_) else None
            r[f'{mn}_positives_test'] = int((ym_ > 0.5).sum())
        a1[feat_name] = r
        del feat
    res['A1'] = a1
    res['seconds'] = round(time.time() - t0)
    print(json.dumps({'A1': a1, 'A3_min_median': [res['A3']['min_offdiag_dist2'], res['A3']['median_offdiag_dist2']]}), flush=True)
    name = 'encoder_diag' + (f'_{enc}' if enc else '') + ('_test' if '--roots' in args else '') + '.json'
    (OUT / name).write_text(json.dumps(res, indent=1))


if __name__ == '__main__':
    sys.path.insert(0, str(ROOT / 'artifacts/experiments/20260926_diagnosis'))
    sys.path.insert(0, str(ROOT / 'artifacts/experiments/20260921_readout_ladder'))
    main()
