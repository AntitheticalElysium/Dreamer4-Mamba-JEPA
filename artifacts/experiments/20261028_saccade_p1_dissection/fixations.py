"""Exact metadata-only replay of training fixations, validated at every saved stream state; S1 exposure EDA."""
import hashlib
import json
import time

from .common import *


def accumulator():
    return {"frames": 0, "centres": np.zeros((7, 7), np.int64), "hud_seen": np.zeros(16, np.int64),
            "hud_value_n": np.zeros((16, 10), np.int64), "hud_value_seen": np.zeros((16, 10), np.int64),
            "labelled_frames": 0, "entity_n": np.zeros(4, np.int64), "entity_seen": np.zeros(4, np.int64),
            "entity_uniform_expected": np.zeros(4), "damage_n": 0, "damage_health_seen": 0,
            "night_n": 0, "night_hud_seen": np.zeros(16, np.int64)}


centres = np.array([(r, c) for r in range(1, 8) for c in range(1, 8)])
coverage = np.bincount(crop_cells(centres).ravel(), minlength=81).reshape(9, 9) / 49


def add(a, fix, L=None, indices=None, flat_grid=None):
    a["frames"] += len(fix)
    a["centres"] += np.bincount((fix[:, 0] - 1) * 7 + fix[:, 1] - 1, minlength=49).reshape(7, 7)
    hr, hc = HUD_CELLS // 9, HUD_CELLS % 9
    seen = (abs(fix[:, 0, None] - hr) <= 1) & (abs(fix[:, 1, None] - hc) <= 1)
    a["hud_seen"] += seen.sum(0)
    if L is None or not len(indices):
        return
    idx, mask = indices
    f, s = fix[mask], seen[mask]
    values = hud_values(L)[idx] if "_hud" not in L else L["_hud"][idx]
    a["labelled_frames"] += len(idx)
    for j in range(16):
        a["hud_value_n"][j] += np.bincount(values[:, j], minlength=10)
        a["hud_value_seen"][j] += np.bincount(values[s[:, j], j], minlength=10)
    night = L["s_light"][idx] < .5
    a["night_n"] += night.sum()
    a["night_hud_seen"] += s[night].sum(0)
    damage = L["_damage"][idx]
    a["damage_n"] += damage.sum()
    a["damage_health_seen"] += s[damage, 0].sum()
    gg = flat_grid[idx]
    cropped = np.take_along_axis(gg, crop_cells(f), 1)
    for k in range(4):
        a["entity_n"][k] += (gg == k + 1).sum()
        a["entity_seen"][k] += (cropped == k + 1).sum()
        a["entity_uniform_expected"][k] += ((gg == k + 1) * coverage.ravel()).sum()


def prepare(L):
    L["_hud"] = hud_values(L)
    first = np.r_[0, np.cumsum(L["ep_T"] + 1)[:-1]]
    L["_damage"] = np.r_[False, np.diff(L["s_body"][:, 0].astype(float)) < 0]
    L["_damage"][first] = False
    grid = np.concatenate([np.pad(ep["grid"], ((0, 0), (0, 2), (0, 0))) for ep in grids(L)]).reshape(-1, 81)
    return grid


def report(a):
    out = {k: v.tolist() if isinstance(v, np.ndarray) else int(v) for k, v in a.items()}
    out["hud_inclusion_rates"] = {name: float(a["hud_seen"][j] / a["frames"]) for j, name in enumerate(HUD)}
    out["analytic_hud_inclusion_rates"] = {name: float(coverage.ravel()[cell]) for name, cell in zip(HUD, HUD_CELLS)}
    out["entity_exposure"] = {k: {"observed": float(a["entity_seen"][i] / max(a["entity_n"][i], 1)),
                                  "uniform_expected": float(a["entity_uniform_expected"][i] / max(a["entity_n"][i], 1))}
                              for i, k in enumerate(KINDS[1:])}
    out["after_damage_health_inclusion"] = float(a["damage_health_seen"] / max(a["damage_n"], 1))
    return out


if __name__ == "__main__":
    start = time.time()
    cfg = TR.load_cfg(); eps = D.load_split(cfg)["train"]
    L = labels("train"); grid = prepare(L)
    lookup = {(int(sh), int(slot)): i for i, (sh, slot) in enumerate(zip(L["ep_shard"], L["ep_slot"]))}
    mapping = np.array([lookup.get((e.shard, e.slot), -1) for e in eps])
    offsets = np.r_[0, np.cumsum(L["ep_T"] + 1)]
    s = D.Streams(eps, cfg["data"]["streams"], cfg["data"]["context_prefix"], seed=1)
    original = D.gather
    D.gather = lambda episodes, records, offsets=None: records  # no random draws or stream mutation in gather
    a, checks, frames, update, next_checkpoint = accumulator(), [], 0, 0, 1
    try:
        while frames < cfg["p1"]["frames"]:
            steps = 64 if frames < cfg["p1"]["frames"] / 2 else 256
            records = s.next(steps)
            rr = np.array([r for row in records for r in row[-steps:]], np.int64)
            label_e = mapping[rr[:, 0]]; mask = label_e >= 0
            idx = offsets[label_e[mask]] + rr[mask, 2]
            add(a, rr[:, 3:5], L, (idx, mask), grid)
            frames += len(rr); update += 1
            if frames >= next_checkpoint * cfg["p1"]["frames"] * cfg["p1"]["checkpoint_every_fraction"]:
                ck = torch.load(RUN / f"ckpt_{next_checkpoint:02d}.pt", map_location="cpu", weights_only=False)
                observed = json.dumps(s.state(), sort_keys=True)
                assert observed == json.dumps(ck["streams"], sort_keys=True), next_checkpoint
                assert frames == ck["state"]["frames"] and update == ck["state"]["update"]
                checks.append({"checkpoint": next_checkpoint, "frames": frames, "update": update,
                               "stream_state_sha256": hashlib.sha256(observed.encode()).hexdigest(), "exact_match": True})
                print("training stream parity", next_checkpoint, frames, flush=True)
                next_checkpoint += 1
                del ck
    finally:
        D.gather = original
    save("training_fixations.json", {**report(a), "checks": checks, "labelled_physical_episodes": int((mapping >= 0).sum()),
         "analytic_cell_coverage": coverage.tolist(), "seconds": time.time() - start})
    del L, grid
    L = labels("dev"); grid = prepare(L); offsets = np.r_[0, np.cumsum(L["ep_T"] + 1)]
    a = accumulator()
    for i, ep in enumerate(s1.episodes(L)):
        seed = [cfg["p1"]["monitor"]["fixed_fixation_seed"], ep.shard, ep.slot]
        f = D.fixation_prior(np.random.default_rng(seed), (ep.T + 1,))
        if i < 2:
            rng = np.random.default_rng(seed)
            assert np.array_equal(f, np.array([D.fixation_prior(rng, ()) for _ in range(ep.T + 1)]))
        idx = np.arange(offsets[i], offsets[i + 1])
        add(a, f, L, (idx, np.ones(len(f), bool)), grid)
    save("dev_fixations.json", {**report(a), "analytic_cell_coverage": coverage.tolist(), "policy": "uniform random, not learned"})
