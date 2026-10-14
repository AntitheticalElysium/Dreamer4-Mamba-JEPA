"""Audit (10-14), stage 5: independent recomputation of the direct-generator study (other agent, DIRECT_RESULTS.md) from its
saved samples, with code written separately from direct_coherence.py, plus milestone trajectories it did not report.
Inputs: the 104 bright-view DEV roots (data/existing_eval_selection.json), panel root frame 15, the 32-key real fan, exact camera
shifts (eval_reference/metadata.pt), the frozen real ridge reader (learned_innovation/eval_u3000_h01/mob_heads.pt, dev_f1
thresholds) and the existing parameter-free scroll reader. Sources are re-selected from the stated rule (stationary camera on all
32 real keys, interior, no same-type source within Manhattan 2, action != DO, no real creation/deletion in the 5-cell cross,
near the player and the cross inside the scored interior; reader-clean = real reader matches the true event on >=95% of real keys
and detects the root source). Per arm and milestone (0/500/1500/3000), over these sources x 64 samples:
  valid (one destination in the cross and camera registered), absent, multiple, source+other duplicate, camera mismatch;
  cow stay / move among all cow draws; distinct valid outcomes per source;
  exact independence baseline: validity if the 5 cross cells' decoded flags were independent with the same per-cell marginals
  (sum_j p_j prod_{i!=j} (1-p_i), registered draws), and the dependence = actual - independent;
  primary joint occupancy energy score on the scored 35 cells (first-power, 64 samples vs 32 real reader-decoded keys).
Mode "noise <arm>" (GPU, u3000, the 14 sources above, 256 fresh noise draws per root, own seed): the mechanism. Arms: energy (this
  study), direct_replicate (the paired study's direct control: same seeds/ledger, numerically diverged run), paired (PairedWorld).
  For paired, keep = 1 - gate_s (1 - route_self_s) (removal mass at the source) and move_j = gate_s route_s[j] (arrival mass the
  source's own packet sends to destination j); the last column is the mean gate over the cross instead of the generation weight.
  copy bookkeeping per draw: keep = routing weight of the source cell on itself; move_j = weight of destination j on the candidate
  that copies the source; generation weight at each cross cell; S = keep + sum_j move_j (physics needs one winner, S ~ 1) and the
  across-draw correlation of keep with sum_j move_j (conservation needs it strongly negative);
  noise influence: resample the 4 local noise dims at ONE tile (source / each destination / a far map tile), or the 4 shared dims
  only, everything else fixed; the probability each cross cell's decoded flag flips and the mean |change| of keep / move weights.
  Each tile's decision driven only by its own noise = no structural coordination.
Mode "paired" (audit S6): the same recomputation for the paired study's arms (paired_event/eval_exploratory_u*/pred_{direct,paired}.pt,
  same 104 roots, readers and sources) -> paired_event/audit_recompute.json.
Usage: audit_s5_direct.py [paired | noise energy|direct_replicate|paired] -> stdout + .../direct_generator/audit_recompute.json | audit_noise_<arm>.json
"""
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT / "artifacts/eda/oracle_v1"
OUT = BASE / "direct_generator"
INTERIOR = [r * 9 + c for r in range(1, 6) for c in range(1, 8)]
CROSS = ((0, 0), (-1, 0), (1, 0), (0, -1), (0, 1))
KINDS = ("zombie", "cow", "skeleton")                                   # layout channels 0, 1, 2
READER_CHANNEL = {"zombie": 0, "skeleton": 1, "cow": 2}                 # real ridge reader order: zombie, skeleton, cow, arrow


def load_scroll():
    spec = importlib.util.spec_from_file_location("audit_scroll", ROOT / "artifacts/experiments/20260927_levers/scroll.py")
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


def decoder(head):
    mu, sd, w = head["mu"].double(), head["sd"].double(), head["w"].double()
    thr = torch.tensor(head["thresholds"]["dev_f1"], dtype=torch.float64)

    def decode(tokens):                                                 # [...,81,192] -> bool [...,63,3] (zombie, cow, skeleton)
        x = tokens[..., :63, :].double()
        s = ((x - mu) / sd) @ w[:-1] + w[-1]
        f = s > thr
        out = torch.stack([f[..., READER_CHANNEL[k]] for k in KINDS], -1)
        out[..., 31, :] = False                                         # player patch is never a mob cell
        return out
    return decode


def event(flags):                                                       # [...,5] bool -> 0..4 one destination, 5 absent, 6 multiple
    n = flags.long().sum(-1)
    return torch.where(n == 0, 5, torch.where(n > 1, 6, flags.long().argmax(-1)))


def energy(x, y):                                                       # [n,K,D], [n,R,D] -> [n] first-power ES / sqrt(D)
    x, y = x.double(), y.double()
    k = x.shape[1]
    cross = torch.cdist(x, y).mean((1, 2))
    spread = torch.cdist(x, x).sum((1, 2)) / (k * (k - 1))
    return (cross - .5 * spread) / np.sqrt(x.shape[-1])


def main(study="direct"):
    sel = json.loads((OUT / "data/existing_eval_selection.json").read_text())["selected_indices"]
    panel = torch.load(BASE / "panel.pt", map_location="cpu", weights_only=False, mmap=True)
    fan = torch.load(BASE / "learned_innovation/real_futures/fan.pt", map_location="cpu", weights_only=False, mmap=True)
    meta = torch.load(OUT / "eval_reference/metadata.pt", map_location="cpu", weights_only=False)
    assert meta["selected_indices"] == sel
    root_tok = panel["tokens"][sel, 15].float(); root_lay = panel["layouts"][sel, 15].bool()
    fan_tok = fan["tokens"][sel].float(); fan_lay = fan["layouts"][sel].bool()
    act = panel["actions"][sel, 15]; shifts = meta["shifts"]
    decode = decoder(torch.load(BASE / "learned_innovation/eval_u3000_h01/mob_heads.pt", map_location="cpu", weights_only=False)["heads"]["real"])
    scroll = load_scroll()
    real_dec = decode(fan_tok); root_dec = decode(root_tok)
    n = len(sel)

    units = []
    for i in range(n):
        if not bool((shifts[i] == 0).all()) or int(act[i]) == 5:
            continue
        for kind in range(3):
            srcs = torch.where(root_lay[i, :, kind])[0].tolist()
            for s in srcs:
                r, c = divmod(s, 9)
                if not (0 < r < 6 and 0 < c < 8):
                    continue
                if any(o != s and abs(o // 9 - r) + abs(o % 9 - c) <= 2 for o in srcs):
                    continue
                pts = [(r + dr) * 9 + c + dc for dr, dc in CROSS]
                true_ev = event(fan_lay[i][:, pts, kind])
                if bool((true_ev >= 5).any()):
                    continue
                if not (abs(r - 3) + abs(c - 4) <= 2 and all(p in INTERIOR for p in pts)):
                    continue
                agree = float((event(real_dec[i][:, pts, kind]) == true_ev).float().mean())
                if agree >= .95 and bool(root_dec[i, s, kind]):
                    units.append({"root": i, "source": s, "kind": kind, "points": pts, "true_events": true_ev})
    res = {"sources": len(units), "by_kind": {k: sum(u["kind"] == j for u in units) for j, k in enumerate(KINDS)},
           "sources_list": [[u["root"], u["source"], KINDS[u["kind"]]] for u in units]}
    real_shift_ok = None

    def physical(dec, registered, tag):
        rows = {"draws": 0, "valid": 0, "absent": 0, "multiple": 0, "duplicate": 0, "camera_mismatch": 0,
                "independent_valid_expected": 0.0, "cow_draws": 0, "cow_valid_stay": 0, "cow_valid_move": 0, "distinct_valid_per_source": []}
        for u in units:
            i, pts, kind = u["root"], u["points"], u["kind"]
            f = dec[i][:, pts, kind]                                    # [K,5]
            reg = registered[i]                                         # [K]
            ev = event(f)
            ok = (ev < 5) & reg
            rows["draws"] += len(f); rows["valid"] += int(ok.sum())
            rows["absent"] += int((ev == 5).sum()); rows["multiple"] += int((ev == 6).sum())
            rows["duplicate"] += int((f[:, 0] & f[:, 1:].any(-1)).sum()); rows["camera_mismatch"] += int((~reg).sum())
            if reg.any():
                p = f[reg].double().mean(0)
                ind = sum(float(p[j] * torch.prod(torch.cat([1 - p[:j], 1 - p[j + 1:]]))) for j in range(5))
                rows["independent_valid_expected"] += ind * int(reg.sum())
            rows["distinct_valid_per_source"].append(len(torch.unique(ev[ok])))
            if kind == 1:
                rows["cow_draws"] += len(f); rows["cow_valid_stay"] += int((ok & (ev == 0)).sum())
                rows["cow_valid_move"] += int((ok & (ev > 0) & (ev < 5)).sum())
        d = rows["draws"]
        summary = {k: (round(v / d, 4) if isinstance(v, int) and k not in ("draws", "cow_draws") and not k.startswith("cow") else v) for k, v in rows.items()}
        summary["independent_valid_expected"] = round(rows["independent_valid_expected"] / d, 4)
        summary["dependence_pp"] = round(100 * (rows["valid"] - rows["independent_valid_expected"]) / d, 2)
        summary["counts"] = {k: rows[k] for k in ("draws", "valid", "absent", "multiple", "duplicate", "camera_mismatch")}
        if rows["cow_draws"]:
            summary["cow_valid_stay"] = round(rows["cow_valid_stay"] / rows["cow_draws"], 4)
            summary["cow_valid_move"] = round(rows["cow_valid_move"] / rows["cow_draws"], 4)
        summary["distinct_valid_per_source"] = round(float(np.mean(rows["distinct_valid_per_source"])), 3)
        return summary

    real_reg = torch.stack([(torch.as_tensor(scroll.SHIFTS)[scroll.estimate(root_tok[i:i + 1, None], fan_tok[i:i + 1])[0]] == shifts[i]).all(-1) for i in range(n)])
    res["real_fan"] = physical(real_dec, real_reg, "real")
    res["real_fan_truth_layouts"] = physical(fan_lay[..., :3].clone(), torch.ones(n, 32, dtype=torch.bool), "truth")
    split = energy(real_dec[:, :16, INTERIOR].flatten(2).float(), real_dec[:, 16:, INTERIOR].flatten(2).float())
    res["real_split_scored_ES_mean"] = round(float(split.mean()), 6)
    res["arms"] = {}
    directory, arms = (OUT, ("point", "energy")) if study == "direct" else (BASE / "paired_event", ("direct", "paired"))
    for update in (0, 500, 1500, 3000):
        for arm in arms:
            pred = torch.load(directory / f"eval_exploratory_u{update:04d}/pred_{arm}.pt", map_location="cpu", weights_only=False)["predictions"]
            dec = decode(pred)
            reg = torch.stack([(scroll.estimate(root_tok[i:i + 1, None], pred[i:i + 1])[0] == 0) for i in range(n)])
            es = energy(dec[:, :, INTERIOR].flatten(2).float(), real_dec[:, :, INTERIOR].flatten(2).float())
            row = physical(dec, reg, arm)
            row["scored_joint_ES_mean"] = round(float(es.mean()), 6)
            res["arms"][f"{arm}_u{update:04d}"] = row
            print(arm, update, json.dumps({k: v for k, v in row.items() if k != "counts"}), flush=True)
            del pred, dec
    (directory / "audit_recompute.json").write_text(json.dumps(res, indent=1))
    print(json.dumps({k: v for k, v in res.items() if k not in ("arms",)}), flush=True)


@torch.no_grad()
def noise(arm):
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import direct_model as M
    import paired_model as PM
    import direct_train as T
    import direct_eval as E
    from d4mj.train import autocast_context
    config = T.settings()
    data, _ = E.cohort("exploratory")
    units = json.loads((OUT / "audit_recompute.json").read_text())["sources_list"]
    world = (PM.PairedWorld() if arm == "paired" else M.DirectWorld()).cuda().eval()
    checkpoint = {"energy": OUT / "training/energy_u3000.pt", "direct_replicate": BASE / "paired_event/training/direct_u3000.pt",
                  "paired": BASE / "paired_event/training/paired_u3000.pt"}[arm]
    world.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=False)["world"])
    decode = decoder(torch.load(BASE / "learned_innovation/eval_u3000_h01/mob_heads.pt", map_location="cpu", weights_only=False)["heads"]["real"])
    offsets = [(0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)]                # candidate order: self + tworld.NEIGHBOURS
    K = 256
    g = torch.Generator().manual_seed(20261014)
    variants = ["base", "source", "dest_up", "dest_down", "dest_left", "dest_right", "far", "shared"]
    flips = {v: torch.zeros(5) for v in variants[1:]}; wchange = {v: torch.zeros(6) for v in variants[1:]}
    book = {"S": [], "keep": [], "move": [], "gen_cross": [], "corr_keep_move": [], "valid": [], "per_source": []}
    for root, source, kindname in units:
        kind = KINDS.index(kindname)
        r, c = divmod(source, 9)
        pts = [(r + dr) * 9 + c + dc for dr, dc in CROSS]
        slots = [0] + [offsets.index((r - p // 9, c - p % 9)) for p in pts[1:]]
        far = max(range(63), key=lambda q: abs(q // 9 - r) + abs(q % 9 - c) if q != 31 else -1)
        ctx = data["context_tokens"][root:root + 1].cuda().float(); ac = data["actions"][root:root + 1].cuda()
        frames = torch.cat([ctx, torch.zeros_like(ctx[:, :1])], 1).expand(K, -1, -1, -1)
        ac = torch.nn.functional.pad(ac, (0, 1)).expand(K, -1)
        with autocast_context(config):
            belief, ha = world.backbone_full(frames[:1], ac[:1])
        # completion is per time step: keep only the predicted step 14 (exact; avoids 16x memory)
        frames, ac, belief, ha = frames[:, 14:15], ac[:, 14:15], belief[:, 14:15].expand(K, -1, -1, -1), ha[:, 14:15].expand(K, -1, -1)   # action_belief is [b,t,d]
        base = M.draw_noise((K, 16), g)[0]                                  # [K,16,81,8]
        fresh = M.draw_noise((K, 16), g)[0]
        out = {}
        for v in variants:
            n = base.clone()
            tile = {"source": pts[0], "dest_up": pts[1], "dest_down": pts[2], "dest_left": pts[3], "dest_right": pts[4], "far": far}.get(v)
            if tile is not None:
                n[:, :, tile, :4] = fresh[:, :, tile, :4]
            elif v == "shared":
                n[..., 4:] = fresh[..., 4:]
            toks, ws = [], []
            for c0 in range(0, K, 64):
                with autocast_context(config):
                    if arm == "paired":
                        t_, routes, aux = world.complete(frames[c0:c0 + 64], ac[c0:c0 + 64], belief[c0:c0 + 64], ha[c0:c0 + 64], n[c0:c0 + 64, 14:15].cuda(), trace=True)
                        w_ = torch.cat([routes.float(), aux["gate"].float()], -1)          # [.,1,81,6]: 5 routes + gate
                    else:
                        t_, w_, _ = world.complete(frames[c0:c0 + 64], ac[c0:c0 + 64], belief[c0:c0 + 64], ha[c0:c0 + 64], n[c0:c0 + 64, 14:15].cuda())
                toks.append(t_[:, 0].float().cpu()); ws.append(w_[:, 0].float().cpu())
            tok, w = torch.cat(toks), torch.cat(ws)
            f = decode(tok)[:, pts, kind]                                   # [K,5]
            if arm == "paired":
                gate = w[:, pts[0], 5]
                ww = torch.stack([1 - gate * (1 - w[:, pts[0], 0])] + [gate * w[:, pts[0], j] for j in range(1, 5)] + [w[:, pts, 5].mean(-1)], -1)
            else:
                ww = torch.stack([w[:, pts[0], 0]] + [w[:, pts[j], slots[j]] for j in range(1, 5)] + [w[:, pts, 5].mean(-1)], -1)   # keep, 4 moves, mean gen
            out[v] = (f, ww)
        f0, w0 = out["base"]
        for v in variants[1:]:
            flips[v] += (out[v][0] != f0).float().mean(0) / len(units)
            wchange[v] += (out[v][1] - w0).abs().mean(0) / len(units)
        keep, move = w0[:, 0], w0[:, 1:5].sum(-1)
        ev = event(f0)
        book["S"].append(float((keep + move).mean())); book["keep"].append(float(keep.mean())); book["move"].append(float(move.mean()))
        book["gen_cross"].append(float(w0[:, 5].mean())); book["valid"].append(float((ev < 5).float().mean()))
        cc = float(torch.corrcoef(torch.stack([keep, move]))[0, 1]) if keep.std() > 0 and move.std() > 0 else None
        book["corr_keep_move"].append(cc)
        book["per_source"].append({"root": root, "source": source, "kind": kindname, "valid": round(book["valid"][-1], 3),
                                   "keep_mean": round(float(keep.mean()), 3), "keep_sd": round(float(keep.std()), 3),
                                   "move_sum_mean": round(float(move.mean()), 3), "move_sum_sd": round(float(move.std()), 3),
                                   "S_p05_p50_p95": [round(float(x), 3) for x in torch.quantile(keep + move, torch.tensor([.05, .5, .95]))],
                                   "corr_keep_move": None if cc is None else round(cc, 3),
                                   "gen_weight_cross_mean": round(float(w0[:, 5].mean()), 3),
                                   "event_frequencies": [round(float((ev == e).float().mean()), 3) for e in range(7)]})
        print(json.dumps(book["per_source"][-1]), flush=True)
    res = {"draws_per_root": K, "flip_probability_by_cross_cell[source,up,down,left,right]": {v: [round(float(x), 4) for x in flips[v]] for v in flips},
           "mean_abs_weight_change[keep,move_up,move_down,move_left,move_right,gen]": {v: [round(float(x), 4) for x in wchange[v]] for v in wchange},
           "mean_valid": round(float(np.mean(book["valid"])), 4),
           "mean_S": round(float(np.mean(book["S"])), 4), "mean_corr_keep_move": round(float(np.mean([x for x in book["corr_keep_move"] if x is not None])), 4),
           "per_source": book["per_source"]}
    res["arm"] = arm; res["checkpoint"] = str(checkpoint.relative_to(ROOT))
    (OUT / f"audit_noise_{arm}.json").write_text(json.dumps(res, indent=1))
    print(json.dumps({k: v for k, v in res.items() if k != "per_source"}, indent=1), flush=True)


if __name__ == "__main__":
    noise(sys.argv[2]) if sys.argv[1:2] == ["noise"] else main("paired" if sys.argv[1:] == ["paired"] else "direct")
