"""The corpus as training streams (saccade/design/M0_CODE.md; CURRICULUM.md §1).

Frames come from the support-v2 store through artifacts/eda/replay.py (shared with the S0 harness). A stream plays
episodes in order in consecutive chunks; a chunk carries the previous P frames of its stream as context only. Per
frame t (CURRICULUM §1, alignment): the percept o_t arrives with the reward and termination of the transition
t-1 -> t (0 / False at an episode's first frame); a_t is the action taken at t (0 at an episode's last frame, which has
none: its output is never used, the next frame starts a new episode from the zero belief); prev_action = a_{t-1}
(-1 at a first frame: DreamerV3's reset, a zero action input).
Usage (project venv): python -m saccade.data targets OUT_DIR   (refit the observation-derived teacher, write full-frame targets;
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
CHANGE_FORMAT = "full_frame_a9b_v2"
CHANGE_MODES = ("same_position", "dominant_translation")


def targets_dir(cfg) -> Path:
    return ROOT / cfg["p1"].get("change_targets_dir", str(Path(cfg["p1"]["out_dir"]) / "targets"))


def validate_change_targets(targets, episodes, directory: Path):
    if targets.shape != (int(frame_offsets(episodes)[-1]), GRID, GRID, len(CHANGE_MODES)):
        raise ValueError("Full-frame change targets must match the training split and cover all 9x9 cells; regenerate legacy map-only targets")
    meta = json.loads((directory / "change_targets.json").read_text())
    expected = hashlib.sha256("\n".join(e.key for e in episodes).encode()).hexdigest()
    if (meta.get("format") != CHANGE_FORMAT or meta.get("episodes_sha256") != expected
            or meta.get("semantic_exclusions") != [] or meta.get("modes") != list(CHANGE_MODES)):
        raise ValueError("Change target format or episode order differs from the configured training split")


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


# ---- Full-frame change grounding. A9b supplies the noise-tolerant pair detector, not semantic region masks. ----

def _a9b():
    sys.path.insert(0, str(ROOT / "artifacts/experiments/20261023_system_design"))
    import eda_change_night_v2 as A9
    return A9


def fit_teacher(out_dir: Path):
    """Refit A9b's corrupted-day detector exactly as eda_change_night_v2.fit does (same cells, episode split, Otsu
    threshold, LightGBM settings) and check it against A9b's saved held-out scores (simulator labels, evaluation only)."""
    if (out_dir / "change_targets.npy").exists():
        raise FileExistsError("Refusing to replace the teacher of an existing target store; choose a new directory")
    out_dir.mkdir(parents=True, exist_ok=True)
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
    report = {"format": CHANGE_FORMAT, "tau": tau, "saved_tau": saved["label_free_day_threshold"], "trees": model.best_iteration,
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


def change_correspondence(frames: np.ndarray):
    """Two full-frame comparison sources, with no region labels or local winner selection.

    Temporal variability supplies soft support for camera estimation at BOTH source and destination: a stationary
    overlay must not penalize a translation of content behind it. BOTH identity and that translation are supervised
    at every cell; choosing between them locally would erase some independent changes. This is offline target
    construction, not an inference input. Identity is available everywhere; translation has geometric validity only.
    """
    c = frames.astype(np.float32).reshape(-1, GRID, CELL, GRID, CELL, 3).transpose(0, 1, 3, 2, 4, 5)
    if len(c) < 2:
        return c, np.empty((0, GRID, GRID, 2, CELL, CELL, 3), np.float32), np.empty((0, GRID, GRID, 2), bool)
    prev, cur = c[:-1], c[1:]
    rr, qq = np.meshgrid(np.arange(GRID), np.arange(GRID), indexing="ij")
    w = (np.abs(np.diff(c, axis=0)) > 0).mean((0, 3, 4, 5)) + 0.01
    shifts = np.asarray(_a9b().SHIFTS)
    cost, valid, support = [], [], []
    for dr, dc in shifts:
        r, q = rr - dr, qq - dc
        ok = (r >= 0) & (r < GRID) & (q >= 0) & (q < GRID)
        cost.append(np.abs(cur - prev[:, r.clip(0, GRID - 1), q.clip(0, GRID - 1)]).mean((-1, -2, -3)))
        valid.append(ok)
        support.append(w * w[r.clip(0, GRID - 1), q.clip(0, GRID - 1)] * ok)
    cost, valid, support = np.stack(cost, 1), np.stack(valid), np.stack(support)
    scores = (cost * support).sum((-1, -2)) / support.sum((-1, -2))
    glob = scores.argmin(1)
    shift = shifts[glob]
    sr = rr[None] - shift[:, 0, None, None]
    sq = qq[None] - shift[:, 1, None, None]
    source = prev[np.arange(len(prev))[:, None, None], sr.clip(0, GRID - 1), sq.clip(0, GRID - 1)]
    sources = np.stack([prev, source], axis=3)
    validity = np.stack([np.ones_like(valid[glob]), valid[glob]], axis=3)
    return c, sources, validity


def pair_features(frames: np.ndarray):
    """Every consecutive pair and every screen cell, including stationary overlays and the player's location."""
    c, source, valid = change_correspondence(frames)
    pi, ri, qi, mode = np.nonzero(valid)
    return _a9b().feats(c[pi + 1, ri, qi], source[pi, ri, qi, mode]), (pi, ri, qi, mode)


def change_probabilities(frames: np.ndarray, model):
    c, source, valid = change_correspondence(frames)
    if not len(source):
        return np.empty((0, GRID, GRID, 2), np.float32)
    exact = (np.abs(c[1:, :, :, None] - source).max((-1, -2, -3)) == 0) & valid
    p = np.full(valid.shape, np.nan, np.float32)
    p[exact] = 0                                                        # the old detector had content-dependent identity bias
    pi, ri, qi, mode = np.nonzero(valid & ~exact)
    if len(pi):
        p[pi, ri, qi, mode] = model.predict(_a9b().feats(c[pi + 1, ri, qi], source[pi, ri, qi, mode]))
    return p


def change_targets(episodes, out_dir: Path, model=None):
    """Soft probabilities [frames,9,9,2]; first frame/geometrically absent sources are undefined. No semantic exclusions.
    Writes a separate versioned store; old map-only targets must never be silently padded or reused."""
    import lightgbm as lgb
    model = model or lgb.Booster(model_file=str(out_dir / "change_teacher.txt"))
    off = frame_offsets(episodes)
    out_dir.mkdir(parents=True, exist_ok=True)
    if (out_dir / "change_targets.npy").exists():
        raise FileExistsError("Refusing to overwrite a target store; choose a new directory")
    pending = out_dir / "change_targets.pending.npy"
    out = np.lib.format.open_memmap(pending, "w+", np.float16, (int(off[-1]), GRID, GRID, len(CHANGE_MODES)))
    out[:] = np.nan
    for i, ep in enumerate(episodes):
        frames = RP.episode_fields(ep.shard, ep.slot)["observations"].numpy()
        block = np.full((ep.T + 1, GRID, GRID, len(CHANGE_MODES)), np.nan, np.float32)
        block[1:] = change_probabilities(frames, model)
        out[off[i]:off[i + 1]] = block
        if i % 500 == 0:
            print(f"targets {i}/{len(episodes)}", flush=True)
    out.flush()
    pending.replace(out_dir / "change_targets.npy")
    (out_dir / "change_targets.json").write_text(json.dumps({"format": CHANGE_FORMAT, "shape": list(out.shape),
        "episodes_sha256": hashlib.sha256("\n".join(e.key for e in episodes).encode()).hexdigest(),
        "semantic_exclusions": [], "modes": list(CHANGE_MODES)}, indent=2))
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
