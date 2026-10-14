"""Stage 1 (2026-10-15): can retained-content transport express the training distribution, and is the Raw token local enough?
CPU only, evaluation-side simulator layouts (oracle_v1/layouts.npy, row-aligned with the training pool); nothing here trains.
  census    a uniform sample of eligible TRAIN windows (direct_generator/data windows: the exact training sampler population). For
            every step t -> t+1 of each window: camera shift from the tokens (parameter-free scroll.estimate), layouts aligned into
            the afterstate frame (cells whose source lies outside the 7x9 view are 'entering' and excluded). Per mob type, every
            appearance / disappearance is classified:
              move       disappearance with an adjacent same-type appearance (greedy one-to-one): expressible as a swap with the
                         emptied cell
              kill       disappearance at the faced cell when the action is DO (deterministic afterstate, not chance)
              edge_in / edge_out   unmatched, on the view border (could be a move across the border: not expressible)
              birth / death        unmatched, interior (spawn / despawn / other: not expressible)
            plus the share of steps whose camera shift is nonzero, and arrows separately (they fly several cells).
  locality  stationary steps from the same sample: (a) ridge probes from ONE Raw token to its own tile's mob occupancy and to
            each neighbour's (context leakage a renderer could exploit); (b) for real single-cow moves i -> j: is the arrival an
            additive transfer of the source's change? rel = |z'_j - (z_j + z_i - z'_i)| / |z'_j - z_j|, and cos(dz_j, -dz_i).
  tail      why some real cow moves are not additive transfers: the same single-cow moves, binned by the frozen bank readers'
            light estimate (mean map token), the tile class under source and destination (bank tile head on the cell's token
            without the cow: source at t+1, destination at t), and the step's static token change (mean |dz| over cells with no
            mob at t or t+1 and no mob neighbour; the night-static nuisance).
Usage: transport_stage1.py census|locality|tail [windows] -> artifacts/eda/oracle_v1/transport/stage1_<mode>.json
"""
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
BASE = ROOT / "artifacts/eda/oracle_v1"
OUT = BASE / "transport"
POOL = ROOT / "artifacts/eda/levers_mamba_long_pools_v1/raw"
TYPES = ("zombie", "cow", "skeleton", "arrow")
NB = ((-1, 0), (1, 0), (0, -1), (0, 1))
FACED = {1: (0, -1), 2: (0, 1), 3: (-1, 0), 4: (1, 0)}                 # Craftax-classic LEFT, RIGHT, UP, DOWN; DO = 5 faces last direction


def scroll_module():
    spec = importlib.util.spec_from_file_location("t1_scroll", ROOT / "artifacts/experiments/20260927_levers/scroll.py")
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


def sample(n, seed=20261015):
    windows = torch.load(BASE / "direct_generator/data/windows.pt", map_location="cpu", weights_only=False)
    windows = windows["windows"] if isinstance(windows, dict) else windows
    g = torch.Generator().manual_seed(seed)
    return windows[torch.randperm(len(windows), generator=g)[:n]]


def load():
    labels = torch.load(POOL / "labels.pt", map_location="cpu", weights_only=False, mmap=True)
    tokens = np.memmap(POOL / "tokens.f16", dtype=np.float16, mode="r", shape=(len(labels["terminal"]), 64, 81, 192))
    lay = np.load(BASE / "layouts.npy", mmap_mode="r"); valid = np.load(BASE / "layouts_valid.npy")
    return labels["actions"], tokens, lay, valid


def align(layout, dr, dc):
    """afterstate frame: successor cell (r,c) shows what was at (r+dr, c+dc); -1 where that lies outside the view"""
    out = -np.ones_like(layout)
    for r in range(7):
        for c in range(9):
            rr, cc = r + dr, c + dc
            if 0 <= rr < 7 and 0 <= cc < 9:
                out[r * 9 + c] = layout[rr * 9 + cc]
    return out


def census(n):
    scroll = scroll_module(); shifts = scroll.SHIFTS
    actions, tokens, lay, valid = load()
    win = sample(n)
    counts = {t: dict.fromkeys(("move", "kill", "edge_in", "edge_out", "birth", "death", "stay"), 0) for t in TYPES}
    steps = moving_steps = 0; skipped = 0; facing = {}
    for row, start in win.tolist():
        if not valid[row]:
            skipped += 1; continue
        tok = torch.from_numpy(np.array(tokens[row, start:start + 16])).float()
        idx = scroll.estimate(tok[:-1], tok[1:]).tolist()               # [15]
        L = np.array(lay[row, start:start + 16]).astype(np.int16)        # [16,63,4]
        facing.pop(row, None)
        for a0 in actions[row, :start].tolist():                         # facing persists from the row's earlier moves
            if a0 in FACED:
                facing[row] = FACED[a0]
        for t in range(15):
            a = int(actions[row, start + t])
            if a in FACED:
                facing[row] = FACED[a]
            dr, dc = shifts[idx[t]]
            steps += 1; moving_steps += (dr, dc) != (0, 0)
            prev = align((L[t] > 0).astype(np.int8), dr, dc); nxt = L[t + 1] > 0
            for k, name in enumerate(TYPES):
                p, q = prev[:, k], nxt[:, k]
                known = p >= 0
                gone = [x for x in range(63) if known[x] and p[x] == 1 and not q[x]]
                came = [x for x in range(63) if known[x] and p[x] == 0 and q[x]]
                counts[name]["stay"] += int(((p == 1) & q).sum())
                used = set()
                for x in gone:
                    r, c = divmod(x, 9)
                    match = [y for y in came if y not in used and abs(y // 9 - r) + abs(y % 9 - c) == 1]
                    if match:
                        used.add(match[0]); counts[name]["move"] += 1; continue
                    f = facing.get(row)
                    if a == 5 and f is not None and x == (3 + f[0]) * 9 + 4 + f[1]:
                        counts[name]["kill"] += 1; continue
                    edge = r in (0, 6) or c in (0, 8)
                    counts[name]["edge_out" if edge else "death"] += 1
                for y in came:
                    if y in used:
                        continue
                    r, c = divmod(y, 9)
                    counts[name]["edge_in" if r in (0, 6) or c in (0, 8) else "birth"] += 1
    res = {"windows_sampled": n, "windows_skipped_invalid_layouts": skipped, "steps": steps,
           "camera_moving_step_share": round(moving_steps / max(steps, 1), 4), "counts": counts}
    for name, c in counts.items():
        changes = sum(v for k, v in c.items() if k != "stay")
        res.setdefault("shares_of_changes", {})[name] = {k: round(v / max(changes, 1), 4) for k, v in c.items() if k != "stay"}
        res.setdefault("changes_per_step", {})[name] = round(changes / max(steps, 1), 4)
    return res


def ridge(x, y, lam=1.0):
    mu, sd = x.mean(0), x.std(0) + 1e-6
    z = torch.cat([(x - mu) / sd, torch.ones(len(x), 1)], 1).double()
    w = torch.linalg.solve(z.T @ z + lam * torch.eye(z.shape[1], dtype=torch.float64), z.T @ y.double())
    return lambda q: (torch.cat([(q - mu) / sd, torch.ones(len(q), 1)], 1).double() @ w)


def auc(score, label):
    s, l = score.numpy(), label.numpy().astype(bool)
    if l.all() or (~l).all():
        return None
    order = np.argsort(s); ranks = np.empty(len(s)); ranks[order] = np.arange(1, len(s) + 1)
    return float((ranks[l].sum() - l.sum() * (l.sum() + 1) / 2) / (l.sum() * (~l).sum()))


def locality(n):
    scroll = scroll_module()
    actions, tokens, lay, valid = load()
    win = sample(n, seed=20261016)
    X, own, nbr, moves = [], [], [], []
    for row, start in win.tolist():
        if not valid[row]:
            continue
        tok = torch.from_numpy(np.array(tokens[row, start:start + 16])).float()
        idx = scroll.estimate(tok[:-1], tok[1:]).tolist()
        L = torch.from_numpy(np.array(lay[row, start:start + 16]) > 0)
        for t in range(15):
            if idx[t] != 0:
                continue
            g = L[t].view(7, 9, 4)
            pad = torch.nn.functional.pad(g.permute(2, 0, 1).float(), (1, 1, 1, 1)).permute(1, 2, 0)
            nb = torch.stack([pad[1 + dr:8 + dr, 1 + dc:10 + dc] for dr, dc in NB], 2).view(63, 4, 4)   # [cell, neighbour, type]
            keep = [x for x in range(63) if x != 31]
            X.append(tok[t, keep]); own.append(L[t][keep].float()); nbr.append(nb[keep])
            # single-cow moves at interior cells, both endpoints non-player, stationary camera
            for x in range(63):
                if L[t][x, 1] and not L[t + 1][x, 1]:
                    r, c = divmod(x, 9)
                    dests = [(r + dr) * 9 + c + dc for dr, dc in NB if 0 <= r + dr < 7 and 0 <= c + dc < 9]
                    arr = [y for y in dests if L[t + 1][y, 1] and not L[t][y, 1]]
                    if len(arr) == 1 and 31 not in (x, arr[0]) and int(L[t][:, 1].sum()) == 1:
                        moves.append((tok[t, x], tok[t + 1, x], tok[t, arr[0]], tok[t + 1, arr[0]]))
    X, own, nbr = torch.cat(X), torch.cat(own), torch.cat(nbr)
    m = len(X); cut = int(.7 * m)
    res = {"stationary_tokens": m, "probe_auc": {}}
    for k, name in enumerate(TYPES[:3]):
        f = ridge(X[:cut], own[:cut, k:k + 1])
        res["probe_auc"][f"{name}_own_tile"] = round(auc(f(X[cut:])[:, 0].float(), own[cut:, k]), 4)
        anyn = nbr[:, :, k].amax(1)
        sel = own[:, k] == 0                                          # neighbour mob, own tile empty: what a renderer could leak
        f = ridge(X[:cut][sel[:cut]], anyn[:cut][sel[:cut]][:, None])
        res["probe_auc"][f"{name}_any_neighbour_given_own_empty"] = round(auc(f(X[cut:][sel[cut:]])[:, 0].float(), anyn[cut:][sel[cut:]]), 4)
        res["probe_auc"][f"{name}_any_neighbour_base_rate"] = round(float(anyn[sel].mean()), 4)
    if moves:
        zi, zi2, zj, zj2 = (torch.stack(x) for x in zip(*moves))
        dzi, dzj = zi2 - zi, zj2 - zj
        rel = (zj2 - (zj - dzi)).norm(dim=-1) / dzj.norm(dim=-1)
        cos = torch.nn.functional.cosine_similarity(dzj, -dzi, dim=-1)
        u = torch.nn.functional.normalize(dzj, dim=-1)
        res["cow_moves"] = {"n": len(moves), "additive_transfer_relative_error_median": round(float(rel.median()), 4),
                            "additive_transfer_relative_error_p10_p90": [round(float(x), 4) for x in torch.quantile(rel, torch.tensor([.1, .9]))],
                            "cos_arrival_vs_minus_departure_mean": round(float(cos.mean()), 4),
                            "arrival_direction_mean_pairwise_cos": round(float(((u @ u.T).sum() - len(u)) / (len(u) * (len(u) - 1))), 4),
                            "arrival_norm_over_token_norm": round(float((dzj.norm(dim=-1) / zj.norm(dim=-1)).mean()), 4)}
    return res


def tail(n):
    scroll = scroll_module()
    actions, tokens, lay, valid = load()
    bank = torch.load(BASE / "snap_bank_readers.pt", map_location="cpu", weights_only=False)["heads"]

    def lin(head, x):
        z = ((x.float() - head["mu"]) / head["sd"]).double()
        return (torch.cat([z, torch.ones(len(z), 1, dtype=torch.float64)], 1) @ head["w"].double()).float()
    win = sample(n, seed=20261016)
    rows = []
    for row, start in win.tolist():
        if not valid[row]:
            continue
        tok = torch.from_numpy(np.array(tokens[row, start:start + 16])).float()
        idx = scroll.estimate(tok[:-1], tok[1:]).tolist()
        L = torch.from_numpy(np.array(lay[row, start:start + 16]) > 0)
        for t in range(15):
            if idx[t] != 0 or int(L[t][:, 1].sum()) != 1:
                continue
            x = int(torch.where(L[t][:, 1])[0][0])
            if L[t + 1][x, 1]:
                continue
            r, c = divmod(x, 9)
            dests = [(r + dr) * 9 + c + dc for dr, dc in NB if 0 <= r + dr < 7 and 0 <= c + dc < 9]
            arr = [y for y in dests if L[t + 1][y, 1]]
            if len(arr) != 1 or 31 in (x, arr[0]):
                continue
            y = arr[0]
            dzi, dzj = tok[t + 1, x] - tok[t, x], tok[t + 1, y] - tok[t, y]
            rel = float((tok[t + 1, y] - (tok[t, y] - dzi)).norm() / dzj.norm())
            anymob = (L[t].any(-1) | L[t + 1].any(-1)).view(7, 9).float()
            near = torch.nn.functional.max_pool2d(anymob[None, None], 3, 1, 1)[0, 0].view(63) > 0
            static = [q for q in range(63) if not near[q] and q != 31]
            noise = float((tok[t + 1, static] - tok[t, static]).norm(dim=-1).mean()) if static else float("nan")
            light = float(lin(bank["light"], tok[t, :63].mean(0, keepdim=True))[0, 0])
            ts, td = (int(lin(bank["tile"], v[None]).argmax()) for v in (tok[t + 1, x], tok[t, y]))
            rows.append((rel, light, ts, td, noise))
    a = np.array(rows)
    rel, light, ts, td, noise = a.T
    res = {"moves": len(a), "rel_median": round(float(np.median(rel)), 4), "share_rel_above_0.3": round(float((rel > .3).mean()), 4)}

    def summary(mask):
        return {"n": int(mask.sum()), "rel_median": round(float(np.median(rel[mask])), 4) if mask.any() else None,
                "share_rel_above_0.3": round(float((rel[mask] > .3).mean()), 4) if mask.any() else None}
    res["by_light"] = {f"{lo}-{hi}": summary((light >= lo) & (light < hi)) for lo, hi in ((-9, .5), (.5, .8), (.8, 9))}
    q = np.nanquantile(noise, [.5, .9])
    res["static_noise_quantiles_p50_p90"] = [round(float(x), 4) for x in q]
    res["by_static_noise"] = {"low(<=p50)": summary(noise <= q[0]), "mid": summary((noise > q[0]) & (noise <= q[1])), "high(>p90)": summary(noise > q[1])}
    res["by_terrain"] = {"same_tile_class": summary(ts == td), "different_tile_class": summary(ts != td)}
    big = rel > .3
    res["among_rel_above_0.3"] = {"share_light_below_0.5": round(float((light[big] < .5).mean()), 4) if big.any() else None,
                                  "share_static_noise_above_p90": round(float((noise[big] > q[1]).mean()), 4) if big.any() else None,
                                  "share_terrain_differs": round(float((ts[big] != td[big]).mean()), 4) if big.any() else None}
    res["corr_rel_static_noise"] = round(float(np.corrcoef(rel[~np.isnan(noise)], noise[~np.isnan(noise)])[0, 1]), 4)
    return res


if __name__ == "__main__":
    mode = sys.argv[1]; n = int(sys.argv[2]) if len(sys.argv) > 2 else 3000
    OUT.mkdir(parents=True, exist_ok=True)
    res = {"census": census, "locality": locality, "tail": tail}[mode](n)
    (OUT / f"stage1_{mode}.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))
