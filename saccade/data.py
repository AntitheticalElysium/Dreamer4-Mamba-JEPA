"""The corpus as training streams (saccade/design/M0_CODE.md; CURRICULUM.md §1).

Frames come from the support-v2 store through artifacts/eda/replay.py (shared with the S0 harness). A stream plays
episodes in order in consecutive chunks; a chunk carries the previous P frames of its stream as context only. Per
frame t (CURRICULUM §1, alignment): the percept o_t arrives with the reward and termination of the transition
t-1 -> t (0 / False at an episode's first frame); a_t is the action taken at t (0 at an episode's last frame, which has
none: its output is never used, the next frame starts a new episode from the zero belief); prev_action = a_{t-1}
(-1 at a first frame: DreamerV3's reset, a zero action input).
Usage (project venv): python -m saccade.data targets OUT_DIR   (build step 2: refit the A9b teacher, write targets;
needs lightgbm on PYTHONPATH)
"""
import hashlib, json, sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "artifacts" / "eda"))
import replay as RP                                                      # noqa: E402

GRID, CELL = 9, 7
A9B = ROOT / "artifacts/eda/system_design_v1/a9"


@dataclass(frozen=True)
class Episode:
    shard: int
    slot: int
    T: int                                                               # actions; frames 0..T
    key: str


def load_split(cfg) -> dict:
    """train / monitor / dev / final episode lists in manifest order. Monitor = a fixed fraction of train episodes chosen
    by the sha256 of the episode id: the same for every arm and seed (CURRICULUM, Budgets)."""
    if (ROOT / cfg["data"]["corpus"]).resolve() != RP.STORE.resolve():
        raise ValueError("Configured corpus differs from the replay store")
    frac = cfg["data"]["monitor_fraction"]
    out = {"train": [], "monitor": [], "dev": [], "final": []}
    for sh in range(len(RP.manifest()["shards"])):
        for slot, e in enumerate(RP._shard(sh)["episodes"]):
            ep, split = Episode(sh, slot, len(e["actions_taken"]), e["episode_id"]), e["split"]
            if split == "train" and int(hashlib.sha256(ep.key.encode()).hexdigest(), 16) % 10_000 < frac * 10_000:
                split = "monitor"
            out[split].append(ep)
    return out


def frame_offsets(episodes) -> np.ndarray:
    """Row of each episode's frame 0 in a split-wide frame array (frames 0..T of every episode, in list order)."""
    return np.r_[0, np.cumsum([e.T + 1 for e in episodes])]


def fixation_prior(rng, shape) -> np.ndarray:
    """Window centres uniform over the 49 positions of a 3x3 window on the 9x9 screen, HUD included (STEP_SPEC 7)."""
    return rng.integers(1, GRID - 1, size=(*shape, 2))


class Streams:
    """n episode streams. A finished episode is followed by the next one of a shared queue, reshuffled at each pass
    (no padding). A fixation is drawn once per emitted frame and kept with it, so a frame reused as context keeps
    the glimpse its memory was computed with. Deterministic given the seed and the call sequence."""

    def __init__(self, episodes, n, prefix, seed, targets_offsets=None):
        self.eps, self.prefix, self.offsets = episodes, prefix, targets_offsets
        self.rng, self.queue, self.uid = np.random.default_rng(seed), [], 0
        self.cur = [self._pull() for _ in range(n)]
        self.t = [0] * n
        self.hist = [[] for _ in range(n)]                               # the last `prefix` emitted records

    def _pull(self):
        if not self.queue:
            self.queue = self.rng.permutation(len(self.eps)).tolist()
        self.uid += 1
        return self.queue.pop(), self.uid

    def state(self) -> dict:
        return {"rng": self.rng.bit_generator.state, "queue": list(self.queue), "uid": self.uid, "cur": list(self.cur),
                "t": list(self.t), "hist": [list(h) for h in self.hist]}

    def load_state(self, s):
        self.rng.bit_generator.state = s["rng"]
        self.queue, self.uid, self.cur, self.t = list(s["queue"]), s["uid"], [tuple(c) for c in s["cur"]], list(s["t"])
        self.hist = [[tuple(r) for r in h] for h in s["hist"]]

    def next(self, steps: int) -> dict:
        """The next chunk of every stream: arrays [n, prefix + steps, ...] (gather)."""
        recs = []
        for i in range(len(self.cur)):
            new = []
            for _ in range(steps):
                (e, uid), t = self.cur[i], self.t[i]
                new.append((e, uid, t, *(int(x) for x in fixation_prior(self.rng, ()))))
                if t == self.eps[e].T:
                    self.cur[i], self.t[i] = self._pull(), 0
                else:
                    self.t[i] = t + 1
            pad = [None] * (self.prefix - len(self.hist[i]))
            recs.append(pad + self.hist[i] + new)
            self.hist[i] = (self.hist[i] + new)[-self.prefix:]
        return gather(self.eps, recs, self.offsets)


def gather(episodes, recs, offsets=None) -> dict:
    """Records (episode index, uid, t, fixation row, fixation col) or None (no frame) -> chunk arrays."""
    n, L = len(recs), len(recs[0])
    c = {"frames": np.zeros((n, L, 63, 63, 3), np.uint8), "action": np.zeros((n, L), np.int64),
         "prev_action": np.full((n, L), -1, np.int64), "reward": np.zeros((n, L), np.float32),
         "terminated": np.zeros((n, L), bool), "first": np.zeros((n, L), bool), "episode": np.zeros((n, L), np.int64),
         "valid": np.zeros((n, L), bool), "fixation": np.ones((n, L, 2), np.int64), "target": np.full((n, L), -1, np.int64)}
    for i, row in enumerate(recs):
        j = 0
        while j < L:
            if row[j] is None:
                j += 1
                continue
            e, uid, t0 = row[j][:3]
            k = j
            while k + 1 < L and row[k + 1] is not None and row[k + 1][1] == uid:
                k += 1
            ts = np.array([r[2] for r in row[j:k + 1]])
            ep = episodes[e]
            f = RP.episode_fields(ep.shard, ep.slot)
            acts, rew, term = (f[x].numpy() for x in ("actions_taken", "rewards", "terminated"))
            c["frames"][i, j:k + 1] = f["observations"][ts[0]:ts[-1] + 1].numpy()
            c["action"][i, j:k + 1] = np.where(ts < ep.T, acts[np.minimum(ts, ep.T - 1)], 0)
            has_prev = ts > 0
            c["prev_action"][i, j:k + 1] = np.where(has_prev, acts[np.maximum(ts - 1, 0)], -1)
            c["reward"][i, j:k + 1] = np.where(has_prev, rew[np.maximum(ts - 1, 0)], 0.0)
            c["terminated"][i, j:k + 1] = has_prev & term[np.maximum(ts - 1, 0)]
            c["first"][i, j:k + 1] = ts == 0
            c["episode"][i, j:k + 1] = uid
            c["valid"][i, j:k + 1] = True
            c["fixation"][i, j:k + 1] = [r[3:5] for r in row[j:k + 1]]
            if offsets is not None:
                c["target"][i, j:k + 1] = np.where(has_prev, offsets[e] + ts, -1)
            j = k + 1
    return c


def select(chunk: dict, rows) -> dict:
    return {k: v[rows] for k, v in chunk.items()}


# ---- change-map targets (build step 2): the A9b label-free teacher, refit on its own data, frozen ----

def _a9b():
    sys.path.insert(0, str(ROOT / "artifacts/experiments/20261023_system_design"))
    import eda_change_night_v2 as A9
    return A9


def fit_teacher(out_dir: Path):
    """Refit A9b's corrupted-day detector exactly as eda_change_night_v2.fit does (same cells, episode split, Otsu
    threshold, LightGBM settings) and check it against A9b's saved held-out scores (simulator labels, evaluation only)."""
    import lightgbm as lgb
    from saccade.eval.gates import average_precision
    A9 = _a9b()
    D = np.load(A9B / "a9b_cells.npz")
    Xr, yr, gr, nr, lfr = D["real_X"], D["real_y"], D["real_g"], D["real_night"], D["real_lf"]
    Xc, gc, lfc = D["corrupt_X"], D["corrupt_g"], D["corrupt_lf"]
    perm = np.random.default_rng(0).permutation(np.unique(gr))
    tr_e, te_e = perm[: int(0.7 * len(perm))], perm[int(0.7 * len(perm)):]
    tau = A9.otsu(lfr[(~nr) & np.isin(gr, tr_e)])
    p = {"objective": "binary", "learning_rate": 0.1, "num_leaves": 63, "min_data_in_leaf": 50, "verbose": -1,
         "num_threads": 12, "seed": 0}
    mc = np.isin(gc, tr_e)
    X, y = Xc[mc], (lfc[mc] > tau).astype(int)
    idx = np.random.default_rng(1).permutation(len(y)); v, t = idx[: len(y) // 10], idx[len(y) // 10:]
    dt = lgb.Dataset(X[t], y[t]); dv = lgb.Dataset(X[v], y[v], reference=dt)
    model = lgb.train(p, dt, 400, valid_sets=[dv], callbacks=[lgb.early_stopping(30, verbose=False)])
    saved = json.loads((A9B / "a9b_results.json").read_text())
    report = {"tau": tau, "saved_tau": saved["label_free_day_threshold"], "trees": model.best_iteration,
              "ap_definition": "threshold_grouped", "saved_ap_definition": "legacy_item_ranked"}
    for name, night in (("day", False), ("night", True)):
        te = (nr == night) & np.isin(gr, te_e)
        s, yy = model.predict(Xr[te], num_iteration=model.best_iteration), yr[te]
        o = np.argsort(-s, kind="stable"); ys = yy[o]; tp = np.cumsum(ys)
        flag = s > 0.5
        report[name] = {"ap": average_precision(yy, s),
                        "legacy_item_rank_ap": float(np.sum(tp[ys == 1] / (np.nonzero(ys == 1)[0] + 1)) / max(ys.sum(), 1)),
                        "precision": float(yy[flag].mean()), "recall": float(flag[yy == 1].mean()),
                        "saved": {k: saved[name]["detector_corrupted_day_labelfree"][k]
                                  for k in ("ap", "labelfree_precision", "labelfree_recall")}}
    model.save_model(str(out_dir / "change_teacher.txt"), num_iteration=model.best_iteration)
    (out_dir / "change_teacher.json").write_text(json.dumps(report, indent=1))
    return model, report


def pair_features(frames: np.ndarray):
    """frames uint8 [n + 1, 63, 63, 3] of one episode -> (features [m, F], (pair, row, col) of each feature row) for the
    map cells of every consecutive pair whose aligned source (estimated camera shift, A9b) is in view; the player's
    cell and entering cells are excluded, as in A9b."""
    A9 = _a9b()
    c = frames[:, :49].astype(np.float32).reshape(-1, 7, 7, 9, 7, 3).transpose(0, 1, 3, 2, 4, 5)   # [n+1, 7, 9, 7, 7, 3]
    c0, c1 = c[:-1], c[1:]
    cost = []
    for dr, dc in A9.SHIFTS:
        r0, r1 = max(0, -dr), min(7, 7 - dr); q0, q1 = max(0, -dc), min(9, 9 - dc)
        cost.append(np.abs(c0[:, r0:r1, q0:q1] - c1[:, r0 + dr:r1 + dr, q0 + dc:q1 + dc]).mean((1, 2, 3, 4, 5)))
    shift = np.array(A9.SHIFTS)[np.argmin(np.stack(cost, 1), 1)]                                    # [n, 2]
    rr, qq = np.meshgrid(np.arange(7), np.arange(9), indexing="ij")
    sr, sq = rr[None] - shift[:, 0, None, None], qq[None] - shift[:, 1, None, None]                 # source of each cell
    ok = (sr >= 0) & (sr < 7) & (sq >= 0) & (sq < 9) & ~((sr == 3) & (sq == 4))
    ok[:, 3, 4] = False
    pi, ri, qi = np.nonzero(ok)
    return A9.feats(c1[pi, ri, qi], c0[pi, sr[pi, ri, qi], sq[pi, ri, qi]]), (pi, ri, qi)


def change_targets(episodes, out_dir: Path, model=None):
    """Soft change probabilities [frames, 7, 9] float16 for a split's frames (frame_offsets order); the target of frame
    t + 1 describes the pair (t, t + 1); NaN where undefined (an episode's first frame, the player's cell, cells whose
    aligned source is out of view). Written once to OUT_DIR/change_targets.npy and memory-mapped by training."""
    import lightgbm as lgb
    model = model or lgb.Booster(model_file=str(out_dir / "change_teacher.txt"))
    off = frame_offsets(episodes)
    out = np.lib.format.open_memmap(out_dir / "change_targets.npy", "w+", np.float16, (int(off[-1]), 7, 9))
    out[:] = np.nan
    for i, ep in enumerate(episodes):
        frames = RP.episode_fields(ep.shard, ep.slot)["observations"].numpy()
        X, (pi, ri, qi) = pair_features(frames)
        block = np.full((ep.T + 1, 7, 9), np.nan, np.float32)
        block[pi + 1, ri, qi] = model.predict(X)
        out[off[i]:off[i + 1]] = block
        if i % 500 == 0:
            print(f"targets {i}/{len(episodes)}", flush=True)
    out.flush()
    return out


def main():
    import yaml
    cfg = yaml.safe_load((ROOT / "saccade/m0.yaml").read_text())
    if sys.argv[1] == "targets":
        out_dir = Path(sys.argv[2]); out_dir.mkdir(parents=True, exist_ok=True)
        model, report = fit_teacher(out_dir)
        print(json.dumps(report, indent=1), flush=True)
        change_targets(load_split(cfg)["train"], out_dir, model)


if __name__ == "__main__":
    main()
