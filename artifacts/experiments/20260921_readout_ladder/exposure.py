"""Exposure audit: what did the old u->u world see in training, and where does it err on it?

Part 1 of the queue declared in `diagnose.py` (committed `e86e33e1`, thresholds fixed there). The old
u->u world trained on a 25,600-window pool drawn from `craftax_support_v2`; replaying its sampler
with seed 20260917 reproduces the cached action triples byte for byte, so the pool is known
exactly. The corpus stores frames, actions, rewards and terminal flags, no simulator state, so the
hazard labels come from what it stores, each validated first:

  health change   reward = achievements (integers) + 0.1 x health change; checked exactly against
                  the fork rows' stored health_delta before use
  zombie adjacent a per-tile classifier on the 7x7 tile pixels, trained on the FIT fork roots'
                  last frames (whose simulator state gives every zombie's tile), validated on the
                  fresh roots, day and night separately
  night           a classifier on map-region pixel statistics, same training and validation
  stay / move     LEFT, RIGHT, UP, DOWN move the player; every other action leaves it in place

Then: unique-transition exposure counts for the pool against the declared thresholds; the world's
one-step teacher error on the pool's logged transitions, overall and along the fatal-versus-safe
direction w the `critical` part fitted, stratified by hazard; and the same exposure counts over the
M4 training corpus (expert v1 + support v2, train split, uniform-eligible), which is what a next
world would train on.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

ROOT = Path("/home/antithetical/EPITA/PERSO/DynamicHorizons-Mamba-JEPA")
HERE = Path(__file__).parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

from d4mj.data import _sha256
from d4mj.lewm_diagnostics import FORK_STORE

from compactness import OLD, OLD_CHECKPOINT  # noqa: E402
from diagnose import MOVES, NEIGHBOURS, TILE  # noqa: E402
from observability import FIT_STATE, FRESH  # noqa: E402

SUPPORT = ROOT / "artifacts/craftax_support_v2"
EXPERT = ROOT / "artifacts/craftax_expert_store_v1"
RECIPE = ROOT / "artifacts/lewm_gates_20260916/paired_window/raw/resolved_recipe.json"
WORLD = ROOT / "artifacts/experiments/20260918_matched_10k/evidence/world_u_u.pt"
SPARSE = {"stay": 200, "move": 200, "damaged": 50}          # diagnose.py's declared thresholds


def health_change(reward):
    """reward = a + 0.1*dh with a a non-negative integer and dh in [-9, +1]."""
    reward = torch.as_tensor(reward, dtype=torch.float64)
    achievements = torch.ceil(reward - 0.1 - 1e-6).clamp(min=0)
    return torch.round((reward - achievements) / 0.1).long()


def tiles(frames, cells=None):
    """[n, 63, 63, 3] uint8 -> [n, k, 147] float for the given (row, col) map cells (default: all 63)."""
    cells = cells or [(r, c) for r in range(7) for c in range(9)]
    x = torch.as_tensor(np.asarray(frames)).float() / 255.0
    return torch.stack([x[:, r * TILE:(r + 1) * TILE, c * TILE:(c + 1) * TILE].flatten(1) for r, c in cells], 1)


def night_features(frames):
    x = torch.as_tensor(np.asarray(frames))[:, :49].float() / 255.0
    return torch.cat((x.mean((1, 2)), x.std((1, 2)), (x.mean(-1) < 0.15).float().mean((1, 2))[:, None]), 1)


def root_rows(store_paths, state_paths=None):
    """Last root frame and its visible-state vector, for the FIT (fork + state store) or fresh roots."""
    frames, visible = [], []
    for i, path in enumerate(store_paths):
        rows = {int(r["step"]): r for r in torch.load(path, weights_only=False)}
        feats = rows if state_paths is None else {int(r["step"]): r for r in torch.load(state_paths[i], weights_only=False)}
        for step, f in feats.items():
            frames.append(rows[step]["frames"][-1])
            visible.append(f["visible"])
    return torch.stack(frames), torch.stack(visible).float()


def train_classifiers(device):
    """Zombie-in-tile and night, fit on FIT roots, validated on fresh roots."""
    fit_seeds = json.loads((HERE / "evidence/root_partition.json").read_text())["fit_train"]["seeds"]
    fork = {int(p.stem.split("-")[1]): p for p in FORK_STORE.glob("seed-*.pt")}
    state = {int(p.stem.split("-")[1]): p for p in FIT_STATE.glob("seed-*.pt")}
    fit_frames, fit_vis = root_rows([fork[s] for s in sorted(fit_seeds)], [state[s] for s in sorted(fit_seeds)])
    val_frames, val_vis = root_rows(sorted(FRESH.glob("seed-*.pt")))
    zombie = lambda vis: vis[:, 1071:1071 + 441].reshape(-1, 7, 9, 7)[..., 0].reshape(len(vis), -1) > 0
    night = lambda vis: vis[:, 1521] < 0.5

    torch.manual_seed(0)
    xf, yf = tiles(fit_frames).reshape(-1, 147), zombie(fit_vis).reshape(-1).float()
    model = nn.Sequential(nn.Linear(147, 128), nn.ReLU(), nn.Linear(128, 1)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    pos = ((1 - yf).sum() / yf.sum().clamp_min(1)).to(device)
    gen = torch.Generator().manual_seed(0)
    for _ in range(4000):
        idx = torch.randint(len(xf), (2048,), generator=gen)
        loss = nn.functional.binary_cross_entropy_with_logits(model(xf[idx].to(device))[:, 0], yf[idx].to(device), pos_weight=pos)
        opt.zero_grad(); loss.backward(); opt.step()
    model.eval()
    night_model = nn.Linear(7, 1).to(device)
    nx, ny = night_features(fit_frames).to(device), night(fit_vis).float().to(device)
    mu, sd = nx.mean(0), nx.std(0).clamp_min(1e-6)
    nopt = torch.optim.LBFGS(night_model.parameters(), max_iter=500, line_search_fn="strong_wolfe")

    def closure():
        nopt.zero_grad()
        l = nn.functional.binary_cross_entropy_with_logits(night_model((nx - mu) / sd)[:, 0], ny)
        l.backward()
        return l
    nopt.step(closure)

    @torch.no_grad()
    def zombie_prob(frames, cells=None):
        x = tiles(frames, cells)
        return torch.sigmoid(model(x.reshape(-1, 147).to(device))[:, 0]).reshape(x.shape[:2]).cpu()

    @torch.no_grad()
    def night_prob(frames):
        return torch.sigmoid(night_model((night_features(frames).to(device) - mu) / sd)[:, 0]).cpu()

    # Validation on the fresh roots: per tile, and on the four cells that decide adjacency.
    truth = zombie(val_vis).reshape(len(val_vis), 7, 9)
    pred = zombie_prob(val_frames).reshape(len(val_vis), 7, 9) > 0.5
    is_night = night(val_vis)
    report = {}
    for name, mask in (("day", ~is_night), ("night", is_night), ("all", torch.ones_like(is_night))):
        t, p = truth[mask], pred[mask]
        nb_t = torch.stack([t[:, r, c] for r, c in NEIGHBOURS], 1).any(1)
        nb_p = torch.stack([p[:, r, c] for r, c in NEIGHBOURS], 1).any(1)
        tp, fp, fn = int((t & p).sum()), int((~t & p).sum()), int((t & ~p).sum())
        report[name] = {"frames": int(mask.sum()), "tile_precision": tp / max(tp + fp, 1), "tile_recall": tp / max(tp + fn, 1),
                        "adjacent_precision": int((nb_t & nb_p).sum()) / max(int(nb_p.sum()), 1),
                        "adjacent_recall": int((nb_t & nb_p).sum()) / max(int(nb_t.sum()), 1),
                        "adjacent_truth_rate": float(nb_t.float().mean())}
    report["night_accuracy"] = float(((night_prob(val_frames) > 0.5) == is_night).float().mean())
    return zombie_prob, night_prob, report


def label(frames, actions, rewards, terminated, zombie_prob, night_prob, batch=4096):
    """Per-transition labels for aligned arrays of frames at t, actions, rewards, terminal flags."""
    near, dark = [], []
    for i in range(0, len(frames), batch):
        near.append((zombie_prob(frames[i:i + batch], list(NEIGHBOURS)) > 0.5).any(1))
        dark.append(night_prob(frames[i:i + batch]) > 0.5)
    dh = health_change(rewards)
    return {"near_zombie": torch.cat(near), "night": torch.cat(dark),
            "move": torch.isin(torch.as_tensor(actions).long(), torch.tensor(MOVES)),
            "damage2": dh <= -2, "damage1": dh <= -1, "dead": torch.as_tensor(terminated).bool()}


def counts(lab, episode):
    """Unique-transition exposure, with the number of distinct episodes behind each count."""
    near = lab["near_zombie"]
    out = {"transitions": int(len(near)), "near_zombie": int(near.sum()),
           "near_zombie_episodes": len(set(np.asarray(episode)[near.numpy()].tolist())),
           "near_zombie_night": int((near & lab["night"]).sum())}
    for name, mask in (("stay", near & ~lab["move"]), ("move", near & lab["move"])):
        out[f"near_zombie_{name}"] = int(mask.sum())
        out[f"near_zombie_{name}_damage2"] = int((mask & lab["damage2"]).sum())
        out[f"near_zombie_{name}_dead"] = int((mask & lab["dead"]).sum())
        out[f"near_zombie_{name}_damage2_rate"] = float((mask & lab["damage2"]).sum() / max(int(mask.sum()), 1))
    out["near_zombie_damaged"] = int((near & lab["damage2"]).sum())
    out["damage2_anywhere"] = int(lab["damage2"].sum())
    out["dead_anywhere"] = int(lab["dead"].sum())
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=HERE / "evidence")
    args = parser.parse_args(argv)
    started = time.time()
    log = lambda **kw: print(json.dumps({**kw, "seconds": round(time.time() - started, 1)}), flush=True)
    device = torch.device("cuda")

    # ---- validate the health decomposition on rows that store health_delta directly ----
    check = [r for p in sorted(FORK_STORE.glob("seed-*.pt"))[:40] for r in torch.load(p, weights_only=False)]
    got = health_change(torch.cat([r["reward"] for r in check]))
    want = torch.cat([r["health_delta"] for r in check]).round().long()
    if not torch.equal(got, want):
        raise SystemExit(f"health decomposition fails on {int((got != want).sum())} of {len(want)} fork transitions")
    log(stage="health_decomposition_exact", transitions=len(want))

    zombie_prob, night_prob, validation = train_classifiers(device)
    log(stage="classifiers", validation=validation)

    # ---- the old u->u pool, reconstructed and verified ----
    from d4mj.config import load_recipe
    from d4mj.data import JointSampler, load_joint_corpus, load_episodes
    config = load_recipe(RECIPE)
    episodes, _ = load_joint_corpus(SUPPORT, config)
    sampler = JointSampler(episodes, config, torch.Generator().manual_seed(20260917))
    cache = torch.load(OLD / "evidence/state_cache.pt", weights_only=False, map_location="cpu")
    ids, starts, acts = [], [], []
    for _ in range(len(cache["actions"]) // config.joint.batch):
        b = sampler.sample()
        acts.append(b.actions); ids += list(b[2]); starts += b[3].tolist()
    if not torch.equal(torch.cat(acts), cache["actions"]):
        raise SystemExit("the reconstructed pool does not reproduce the cached actions")
    by_id = {e.episode_id: e for e in episodes}
    keys, frames, actions, rewards, terms = [], [], [], [], []
    for eid, s in zip(ids, starts):
        e = by_id[eid]
        for k in range(3):
            keys.append((eid, s + k))
            frames.append(np.asarray(e.observations[s + k]))
            actions.append(int(e.actions_taken[s + k])); rewards.append(float(e.rewards[s + k]))
            terms.append(bool(e.terminated[s + k]))
    frames = np.stack(frames)
    lab = label(frames, actions, rewards, terms, zombie_prob, night_prob)
    unique = {}
    for i, k in enumerate(keys):
        unique.setdefault(k, i)
    first = torch.tensor(sorted(unique.values()))
    pool = counts({k: v[first] for k, v in lab.items()}, [keys[i][0] for i in first.tolist()])
    pool["unique_transitions_of"] = len(keys)
    sparse = bool(pool["near_zombie_stay"] < SPARSE["stay"] or pool["near_zombie_move"] < SPARSE["move"]
                  or pool["near_zombie_damaged"] < SPARSE["damaged"])
    log(stage="pool", sparse=sparse, **{k: v for k, v in pool.items() if "rate" not in k})

    # ---- the world's one-step error on the logged pool transitions, overall and along w ----
    from d4mj.config import config_from_dict
    from d4mj.lewm import LeWMWorld
    stored = torch.load(OLD_CHECKPOINT, map_location="cpu", weights_only=False)
    world = LeWMWorld(config_from_dict(stored["config"])).to(device)
    del stored
    world.load_state_dict(torch.load(WORLD, map_location="cpu", weights_only=False)["state_dict"], strict=True)
    world.eval()
    w = torch.load(HERE / "evidence/critical_direction.pt", weights_only=False)["w"]
    err_all, err_w, var_next = [], [], []
    with torch.no_grad():
        for i in range(0, len(cache["u"]), 512):
            u = cache["u"][i:i + 512].to(device)
            pred = world.teacher(u[:, :, None], cache["actions"][i:i + 512].to(device)).predicted[:, :, 0].cpu()
            diff = pred - cache["u"][i:i + 512, 1:]
            err_all.append(diff.square().sum(-1).reshape(-1)); err_w.append((diff @ w).square().reshape(-1))
    err_all, err_w = torch.cat(err_all), torch.cat(err_w)
    nxt = cache["u"][:, 1:].reshape(-1, cache["u"].shape[-1])
    base_all, base_w = (nxt - nxt.mean(0)).square().sum(-1).mean(), ((nxt - nxt.mean(0)) @ w).square().mean()

    def stratum(mask):
        return {"transitions": int(mask.sum()),
                "normalized_error_all": float(err_all[mask].mean() / base_all),
                "normalized_error_w": float(err_w[mask].mean() / base_w)}
    everyone = torch.ones(len(err_all), dtype=torch.bool)
    near = lab["near_zombie"]
    errors = {"all": stratum(everyone), "not_near_zombie": stratum(~near), "near_zombie": stratum(near),
              "near_zombie_stay": stratum(near & ~lab["move"]), "near_zombie_move": stratum(near & lab["move"]),
              "near_zombie_damage2": stratum(near & lab["damage2"]), "near_zombie_no_damage": stratum(near & ~lab["damage1"]),
              "night": stratum(lab["night"]), "day": stratum(~lab["night"])}
    log(stage="pool_errors", near=errors["near_zombie"], damaged=errors["near_zombie_damage2"], rest=errors["not_near_zombie"])

    # ---- the M4 training corpus: exposure only ----
    corpus = {}
    for name, path in (("expert_v1", EXPERT), ("support_v2", SUPPORT)):
        eps = episodes if name == "support_v2" else load_episodes(path, verify=True)
        eps = [e for e in eps if e.split == "train" and e.uniform_eligible]
        parts, eid = [], []
        for e in eps:
            n = len(e.actions_taken)
            lab_e = label(np.asarray(e.observations[:n]), e.actions_taken, e.rewards, e.terminated, zombie_prob, night_prob)
            parts.append(lab_e); eid += [e.episode_id] * n
        merged = {k: torch.cat([p[k] for p in parts]) for k in parts[0]}
        corpus[name] = counts(merged, eid)
        corpus[name]["episodes"] = len(eps)
        log(stage="corpus", name=name, near_zombie=corpus[name]["near_zombie"], transitions=corpus[name]["transitions"])

    evidence = {"schema": "d4mj_exposure_v1", "script_sha256": _sha256(Path(__file__)),
                "diagnose_sha256": _sha256(HERE / "diagnose.py"), "world_sha256": _sha256(WORLD),
                "classifier_validation": validation, "pool": pool, "sparse_exposure": sparse,
                "thresholds": SPARSE, "pool_prediction_error": errors, "m4_corpus": corpus}
    (args.out / "exposure.json").write_text(json.dumps(evidence, indent=2) + "\n")
    log(status="exposure_complete", sparse_exposure=sparse)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
