"""D12. One step, all 17 actions: which kinds of transition does each world get wrong, and which events does
its imagined successor carry? From the D10 rollouts (depth-1 fan) and the simulator facts; nothing trained.

Class of each (root, action), from a visible-state passability rule (key 0), mutually exclusive in this order:
  moved        a move that changed the player's position (the view scrolls)
  blocked      a move predicted not to change player position (solid tile or mob)
  interact     another action that changed a view tile or the inventory (mine, place, craft, eat, drink)
  sleep        the player fell asleep
  idle         nothing of the above changed (NOOP-like; mobs, intrinsics and daylight still evolve)
Per class: share of transitions; error / V, aleatoric noise (4 keys) / V, bias vs the 4-key mean / V,
predictable signal / V, captured = 1 - bias / signal.
Events (AUC, ridge probes split by seed 70/30): moved (move actions), tile_changed and inventory_changed
(non-move actions), health_down, sleep_started (SLEEP rows), facing_changed -- from the true successor's
change (s' - root), the imagined change (g - root), the Mamba output h(a), and the root state with a
separate probe per action (what the input holds).
T only: share of its one-step error per region of the 9x9 token grid by class (revealed = the row or
column that scrolls into view on a successful move; hud = token rows 7-8).
"""
import json
import sys
from pathlib import Path

import torch

HERE = Path(__file__).parent
ROOT = Path("/home/antithetical/EPITA/PERSO/DynamicHorizons-Mamba-JEPA")
sys.path.insert(0, str(HERE))
from choices import move_table  # noqa: E402
from compound import auc  # noqa: E402

DATA = ROOT / "artifacts/eda/diagnosis_rollouts_v1"
WORLDS = ("H2", "Z", "sZ", "U", "W", "T")
CLASSES = ("moved", "blocked", "interact", "sleep", "idle")
N = 17


def classify(meta):
    rv = meta["root_visible"].float()
    ov = meta["onestep_visible"][:, 0].float()                      # key 0, [R,17,1534]
    cat, _ = move_table(rv)
    tiles_r = rv[:, :1071].reshape(-1, 1, 7, 9, 17).argmax(-1)
    tiles_s = ov[..., :1071].reshape(-1, N, 7, 9, 17).argmax(-1)
    tile_changed = (tiles_r != tiles_s).flatten(2).any(-1)
    inv_changed = (rv[:, None, 1522:1534] != ov[..., 1522:1534]).any(-1)
    sleep_started = (rv[:, None, 1520] < 0.5) & (ov[..., 1520] > 0.5)
    health_down = ov[..., 1512] < rv[:, None, 1512]
    facing_changed = (rv[:, None, 1516:1520].argmax(-1) != ov[..., 1516:1520].argmax(-1))
    is_move = torch.zeros(N, dtype=torch.bool); is_move[1:5] = True
    # move_table marks a move INTO lava as blocked (cat 1), but the player does enter lava (game_logic
    # move_player: lava is enterable). Read the target tile directly so lava entries count as moves.
    lava_move = torch.zeros_like(cat, dtype=torch.bool)
    for a, (dr, dc) in ((1, (0, -1)), (2, (0, 1)), (3, (-1, 0)), (4, (1, 0))):
        lava_move[:, a] = (tiles_r[:, 0, 3 + dr, 4 + dc] == 14) & (rv[:, 1520] < 0.5)
    moved = is_move[None] & ((cat == 0) | lava_move)
    cls = torch.full(cat.shape, 4)
    cls[~is_move[None].expand_as(cat) & (tile_changed | inv_changed)] = 2
    cls[sleep_started] = 3
    cls[is_move[None] & ~moved] = 1
    cls[moved] = 0
    events = {"moved": (moved, is_move[None].expand_as(cat)), "tile_changed": (tile_changed, ~is_move[None].expand_as(cat)),
              "inventory_changed": (inv_changed, ~is_move[None].expand_as(cat)), "health_down": (health_down, None),
              "sleep_started": (sleep_started, torch.zeros_like(cat, dtype=torch.bool) | (torch.arange(N) == 6)[None]),
              "facing_changed": (facing_changed, None)}
    return cls, events


def probe_auc(x, y, tr, te, lams=(1e-2, 1e-1, 1, 10, 100)):
    mu, sd = x[tr].mean(0), x[tr].std(0).clamp_min(1e-6)
    xs = ((x - mu) / sd).double()
    t = y.double() * 2 - 1
    idx = torch.where(tr)[0]
    half = idx[: len(idx) // 2], idx[len(idx) // 2:]
    best = None
    for lam in lams:
        w = torch.linalg.solve(xs[half[0]].T @ xs[half[0]] + lam * len(half[0]) * torch.eye(xs.shape[1], dtype=xs.dtype),
                               xs[half[0]].T @ t[half[0]])
        a = auc(xs[half[1]] @ w, y[half[1]])
        if best is None or a > best[0]:
            best = (a, lam)
    w = torch.linalg.solve(xs[tr].T @ xs[tr] + best[1] * int(tr.sum()) * torch.eye(xs.shape[1], dtype=xs.dtype), xs[tr].T @ t[tr])
    return auc(xs[te] @ w, y[te])


def main():
    meta = torch.load(DATA / "meta.pt")
    seeds = meta["seed"]
    useed = seeds.unique()
    train_seeds = useed[torch.randperm(len(useed), generator=torch.Generator().manual_seed(0))[: int(0.7 * len(useed))]]
    tr_root = torch.isin(seeds, train_seeds)
    cls, events = classify(meta)
    result = {"class_share": {c: float((cls == i).float().mean()) for i, c in enumerate(CLASSES)}, "worlds": {}}
    print(json.dumps(result["class_share"]), flush=True)
    for w in WORLDS:
        d = torch.load(DATA / f"{w}.pt")
        g, s, root = d["one"], d["one_true"], d["root"]                       # [R,17,D], [R,4,17,D], [R,D]
        V = float(torch.cov((d["true"][:, 0].flatten(0, 1)).T.double()).trace())
        mean = s.mean(1)
        noise = ((s - mean[:, None]) ** 2).sum(-1).sum(1) / 3                  # [R,17] unbiased tr Var
        err = ((g - s[:, 0]) ** 2).sum(-1)
        bias = ((g - mean) ** 2).sum(-1) - noise / 4
        signal = ((mean - root[:, None]) ** 2).sum(-1) - noise / 4
        per = {}
        for i, c in enumerate(CLASSES):
            m = cls == i
            per[c] = {"error": float(err[m].mean() / V), "noise": float(noise[m].mean() / V), "bias": float(bias[m].mean() / V),
                      "signal": float(signal[m].mean() / V), "captured": float(1 - bias[m].mean() / signal[m].mean()),
                      "share_of_total_error": float(err[m].sum() / err.sum())}
        ev = {}
        tr = tr_root[:, None].expand(-1, N)
        for name, (label, rows) in events.items():
            rows = torch.ones_like(label) if rows is None else rows
            if label[rows].float().mean() in (0, 1):
                continue
            r = rows.flatten()
            lab = label.flatten()[r]
            trr, ter = tr.flatten()[r], ~tr.flatten()[r]
            e = {"rate": float(lab.float().mean()),
                 "true_change": probe_auc((s[:, 0] - root[:, None]).flatten(0, 1)[r], lab, trr, ter),
                 "imagined_change": probe_auc((g - root[:, None]).flatten(0, 1)[r], lab, trr, ter)}
            if "one_h" in d:
                e["h"] = probe_auc(d["one_h"].float().flatten(0, 1)[r], lab, trr, ter)
            # the input: root state, one probe per action, pooled scores
            scores, labels = [], []
            for a in range(N):
                ra = rows[:, a]
                if ra.sum() < 50 or label[ra, a].float().mean() in (0, 1):
                    continue
                x = root[ra]
                ya = label[ra, a]
                tra, tea = tr_root[ra], ~tr_root[ra]
                if ya[tra].float().mean() in (0, 1):
                    continue
                mu, sd = x[tra].mean(0), x[tra].std(0).clamp_min(1e-6)
                xs = ((x - mu) / sd).double()
                wv = torch.linalg.solve(xs[tra].T @ xs[tra] + 10 * int(tra.sum()) * torch.eye(xs.shape[1], dtype=xs.dtype),
                                        xs[tra].T @ (ya[tra].double() * 2 - 1))
                scores.append(xs[tea] @ wv); labels.append(ya[tea])
            e["root_per_action"] = auc(torch.cat(scores), torch.cat(labels)) if scores else float("nan")
            ev[name] = e
        result["worlds"][w] = {"V": V, "by_class": per, "events": ev}
        print(w, json.dumps({c: {k: round(v, 3) for k, v in x.items()} for c, x in per.items()}), flush=True)
        print(w, "events", json.dumps({k: {kk: round(vv, 3) for kk, vv in v.items()} for k, v in ev.items()}), flush=True)
    # T per-token error map by class
    tf = torch.load(DATA / "T_fullspace.pt")
    tok = tf["one_err_tok"].float().view(-1, N, 9, 9)                         # [R,17,9,9]
    regions = {}
    rv = meta["root_visible"].float()
    for i, c in enumerate(CLASSES):
        m = cls == i
        e = tok[m]
        total = float(e.sum())
        hud = float(e[:, 7:].sum()) / total
        centre = float(e[:, 3, 4].sum()) / total
        near = float(sum(e[:, r, cc].sum() for r, cc in ((2, 4), (4, 4), (3, 3), (3, 5)))) / total
        entry = {"hud": hud, "player": centre, "neighbours": near}
        if c == "moved":
            acts = torch.arange(N)[None].expand_as(cls)[m]
            strip = torch.zeros_like(e)
            for a, (sl) in ((1, (slice(0, 7), slice(0, 1))), (2, (slice(0, 7), slice(8, 9))),
                            (3, (slice(0, 1), slice(0, 9))), (4, (slice(6, 7), slice(0, 9)))):
                strip[acts == a][:, sl[0], sl[1]] = 1
                rows_a = acts == a
                strip[rows_a] = 0
                tmp = torch.zeros(9, 9); tmp[sl[0], sl[1]] = 1
                strip[rows_a] = tmp
            entry["revealed_strip"] = float((e * strip).sum()) / total
            entry["revealed_strip_tile_share"] = float(strip.sum() / strip.numel() * 81 / 81)
        regions[c] = entry
    result["T_error_regions"] = regions
    print("T regions", json.dumps({c: {k: round(v, 3) for k, v in x.items()} for c, x in regions.items()}), flush=True)
    (HERE / "onestep.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
