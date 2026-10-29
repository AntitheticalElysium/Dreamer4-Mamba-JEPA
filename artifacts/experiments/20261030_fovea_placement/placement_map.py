"""The guided-placement rule of the fovea fine-tune (PLAN.md): a privileged diagnostic built from simulator labels.

Target consequences: the five the field misses (generation diagnosis A, F, G, H), cells as loss_accounting.cell_classes
assigns them. The context of frame t is read from state t-1 only (it is in the denoiser's conditioning, never the
outcome): action a_{t-1}, the faced block, a zombie at Manhattan distance 1, sleeping. On the train split (monitor
episodes excluded), per context: frames n and, per window of the fixation prior's 49, the frames whose target cells
fall inside it. A context is an ATTEMPT context if n >= N_MIN and its best window covers a target consequence on more
than the base rate of target frames; its window is that best one (ties: the window centred nearest the context's
target cells). In the guided arm an attempt frame takes its window with probability Q (one fixed coin per frame),
otherwise the prior's draw; every other frame keeps the prior's draw.
Writes map.json (the rule and its statistics), place_train.npy (int8 [train frames, 2] window centre, -1 = prior;
data.frame_offsets order) and targets_train.npy (packed bits of each frame's target cells, for the manipulation check).
Usage (project venv, CPU): python -m artifacts.experiments.20261030_fovea_placement.placement_map
"""
import importlib, json
from pathlib import Path

import numpy as np
import yaml

from saccade import data as D

LA = importlib.import_module("artifacts.experiments.20261029_generation_diagnosis.loss_accounting")
ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
LABELS = ROOT / "artifacts/eda/s0/s0_train_labels.npz"
TARGETS = ("drink_up", "wood_up", "sapling_up", "table_placed", "zombie_hit")
N_MIN, Q, COIN_SEED, NB = 100, 0.5, 2030, 32
CENTRES = np.array([(r, c) for r in range(1, D.GRID - 1) for c in range(1, D.GRID - 1)])   # data.fixation_prior's support
COVER = np.zeros((len(CENTRES), 81), bool)                               # window x periphery cell
for _i, (_r, _c) in enumerate(CENTRES):
    for _dr in (-1, 0, 1):
        COVER[_i, (_r + _dr) * 9 + _c - 1:(_r + _dr) * 9 + _c + 2] = True


def offsets(L):
    return np.r_[0, np.cumsum(L["ep_T"] + 1)], np.r_[0, np.cumsum(L["ep_T"])]


def context_ids(L):
    """Context id of each state g, deciding frame g + 1 of the same episode; -1 at an episode's last state."""
    s_off, a_off = offsets(L)
    n = len(L["s_dir"])
    d = LA.DIRS[L["s_dir"]]
    face = L["s_view"][np.arange(n), 3 + d[:, 0], 4 + d[:, 1]].astype(np.int64)
    zadj = (L["s_zombies_m"] & (np.abs(L["s_zombies_p"].astype(int) - L["s_pos"].astype(int)[:, None]).sum(-1) == 1)).any(1)
    act = np.concatenate([np.r_[L["t_action"][a_off[e]:a_off[e + 1]].astype(np.int64), -1] for e in range(len(L["ep_T"]))])
    assert face.max() < NB and len(act) == n
    ids = ((act * NB + face) * 2 + zadj) * 2 + L["s_sleep"].astype(np.int64)
    return np.where(act < 0, -1, ids)


def describe(cid):
    a, rest = divmod(int(cid), NB * 4)
    face, rest = divmod(rest, 4)
    return {"action": a, "faced_block": face, "zombie_adjacent": bool(rest // 2), "sleeping": bool(rest % 2)}


def target_cells(L):
    """[states, 81] bool: the cells of frame g holding one of TARGETS (false at every episode's first frame)."""
    cls = LA.cell_classes(L)
    tgt = np.isin(cls, [LA.CLASSES.index(c) for c in TARGETS])
    tgt[offsets(L)[0][:-1]] = False                                      # cell_classes compares across episodes there
    return tgt, cls


def window_of_state(L, rule):
    """[states, 2]: the rule's window centre for frame g + 1 decided at state g (attempt contexts), else -1."""
    ids = context_ids(L)
    out = np.full((len(ids), 2), -1, np.int8)
    for cid, (r, c) in rule.items():
        out[ids == cid] = (r, c)
    return out


def main():
    cfg = yaml.safe_load((ROOT / "saccade/m0.yaml").read_text())
    L = dict(np.load(LABELS))
    split = D.load_split(cfg)
    s_off, _ = offsets(L)
    where = {(int(s), int(o)): e for e, (s, o) in enumerate(zip(L["ep_shard"], L["ep_slot"]))}
    order = [where[(ep.shard, ep.slot)] for ep in split["train"]]
    assert all(L["ep_T"][e] == ep.T for e, ep in zip(order, split["train"]))
    ids, (tgt, cls) = context_ids(L), target_cells(L)
    # frames t >= 1 of train episodes: global state g = s_off[e] + t, context at g - 1
    g = np.concatenate([np.arange(s_off[e] + 1, s_off[e + 1]) for e in order])
    ctx, tg = ids[g - 1], tgt[g]
    assert (ctx >= 0).all()
    any_t = tg.any(1)
    base_rate = float(any_t.mean())
    n_ctx = np.bincount(ctx, minlength=17 * NB * 4)
    hit = np.zeros((len(n_ctx), len(CENTRES)))
    cells = np.zeros((len(n_ctx), 81))
    sel = np.nonzero(any_t)[0]
    np.add.at(hit, ctx[sel], (tg[sel].astype(np.int32) @ COVER.T.astype(np.int32)) > 0)
    np.add.at(cells, ctx[sel], tg[sel])
    rule, rows = {}, []
    for cid in np.nonzero(n_ctx >= N_MIN)[0]:
        p = hit[cid] / n_ctx[cid]
        if p.max() <= base_rate:
            continue
        best = np.nonzero(hit[cid] == hit[cid].max())[0]
        cr, cc = np.divmod(np.arange(81), 9)
        dist = [(cells[cid] * ((cr - CENTRES[w, 0]) ** 2 + (cc - CENTRES[w, 1]) ** 2)).sum() for w in best]
        w = int(best[int(np.argmin(dist))])
        rule[int(cid)] = (int(CENTRES[w, 0]), int(CENTRES[w, 1]))
        rows.append({"context": int(cid), **describe(cid), "frames": int(n_ctx[cid]), "covered_rate": float(p[w]),
                     "window": rule[int(cid)], "ties": int(len(best)),
                     "target_frames_by_class": {c: int(((cls[g[ctx == cid]] == LA.CLASSES.index(c)).any(1)).sum()) for c in TARGETS}})
    rows.sort(key=lambda r: -r["frames"])
    # per-frame placement for every train frame (frame_offsets order; frame 0 never placed)
    off = D.frame_offsets(split["train"])
    win = window_of_state(L, rule)
    coin = np.random.default_rng(COIN_SEED).random(int(off[-1]))
    place = np.full((int(off[-1]), 2), -1, np.int8)
    bits = np.zeros((int(off[-1]), 81), bool)
    for i, e in enumerate(order):
        a, b = s_off[e], s_off[e + 1]
        w_ = win[a:b - 1]                                                # frame t decided at state t - 1
        take = (w_[:, 0] >= 0) & (coin[off[i] + 1:off[i + 1]] < Q)
        place[off[i] + 1:off[i + 1]][take] = w_[take]
        bits[off[i]:off[i + 1]] = tgt[a:b]
    np.save(HERE / "place_train.npy", place)
    np.save(HERE / "targets_train.npy", np.packbits(bits, axis=1))
    # predicted manipulation check on the train split: share of target frames whose cells the glimpse covers
    attempt = np.array([int(c) in rule for c in ctx])
    prior_cover = (tg.astype(np.int32) @ COVER.T.astype(np.int32) > 0).mean(1)          # P(a prior window covers)
    rw = np.array([rule.get(int(c), (-1, -1)) for c in ctx])
    widx = np.where(rw[:, 0] >= 0, (rw[:, 0] - 1) * 7 + rw[:, 1] - 1, 0)
    rule_cover = (tg & COVER[widx]).any(1) & attempt
    guided = np.where(attempt, Q * rule_cover + (1 - Q) * prior_cover, prior_cover)
    by_class = {}
    for c in TARGETS:
        m = (cls[g] == LA.CLASSES.index(c)).any(1)
        by_class[c] = {"frames": int(m.sum()), "in_attempt_contexts": float(attempt[m].mean()),
                       "covered_prior": float(prior_cover[m].mean()), "covered_guided": float(guided[m].mean())}
    res = {"labels": str(LABELS), "targets": TARGETS, "n_min": N_MIN, "q": Q, "coin_seed": COIN_SEED,
           "train_frames_t_ge_1": int(len(g)), "base_rate_target_frames": base_rate,
           "attempt_contexts": len(rule), "frames_in_attempt_contexts": float(attempt.mean()),
           "frames_placed": float((place[:, 0] >= 0).sum() / len(g)),
           "target_frames_in_attempt_contexts": float(attempt[any_t].mean()),
           "predicted_coverage": {"all_targets": {"prior": float(prior_cover[any_t].mean()), "guided": float(guided[any_t].mean())},
                                  **by_class},
           "rule": {str(k): v for k, v in rule.items()}, "contexts": rows}
    (HERE / "map.json").write_text(json.dumps(res, indent=1))
    print(json.dumps({k: v for k, v in res.items() if k not in ("rule", "contexts")}, indent=1))
    for r in rows[:25]:
        print(r)


if __name__ == "__main__":
    main()
