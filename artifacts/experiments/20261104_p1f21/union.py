"""P1-F2.1's training corpus (PLAN.md): the original support-v2 store and the two authorised collections (coverage-v1,
risk-v1) as ONE virtual replay store, without copying a shard. Its manifest lists the original store's 420 shards
first, so every existing (shard, slot) label (dev and final included) keeps its meaning, then coverage-v1's 128 and
risk-v1's 512 shards, each file relative to the union directory. `use` points replay at it and keeps the ORIGINAL
store's monitor as the checkpoint-selection monitor: the new stores' monitor-hash episodes are moved to 'held_out'
(never trained on). Change-map targets: the frozen teacher that wrote the original targets, on every new training
frame, appended to the original targets (bitwise unchanged). saccade/{data,train,model}.py, replay.py and m0.yaml are
not modified.
Usage (project venv, CPU): python -m artifacts.experiments.20261104_p1f21.union manifest | part I N (I = 0..N-1,
  in parallel) | assemble N | verify
"""
import hashlib, json, os, sys, time
from pathlib import Path

import numpy as np

from saccade import data as D, train as TR

HERE = Path(__file__).resolve().parent
STORES = ("artifacts/craftax_support_v2", "artifacts/craftax_coverage_v1", "artifacts/craftax_risk_v1")
CORPUS = "artifacts/craftax_union_v1"
OUT_DIR = "artifacts/saccade_m0_union"                                  # p1.out_dir: targets/change_targets.npy
UNION, OLD = TR.ROOT / CORPUS, TR.ROOT / "artifacts/saccade_m0/targets"
TARGETS = TR.ROOT / OUT_DIR / "targets"


def build_manifest():
    """The union manifest from the three complete source manifests (shard order and hashes copied)."""
    shards, sources = [], []
    for s in STORES:
        m = json.loads((TR.ROOT / s / "manifest.json").read_text())
        assert m["format"] == "d4mj_episode_store_v1" and m["complete"]
        sources.append({"store": s, "kind": m["kind"], "manifest_sha256": TR.file_sha256(TR.ROOT / s / "manifest.json"),
                        "shards": len(m["shards"]), "episodes": m["episodes"], "transitions": m["transitions"]})
        shards += [{"file": os.path.relpath(TR.ROOT / s / x["file"], UNION), "sha256": x["sha256"], "source": s,
                    "episodes": x["episodes"], "transitions": x["transitions"]} for x in m["shards"]]
    UNION.mkdir(exist_ok=True)
    (UNION / "manifest.json").write_text(json.dumps(
        {"format": "d4mj_episode_store_v1", "kind": "saccade_union_v1", "complete": True, "sources": sources,
         "episodes": sum(s["episodes"] for s in sources), "transitions": sum(s["transitions"] for s in sources),
         "shards": shards}, indent=1) + "\n")


def use(cfg):
    """Replay reads the union; load_split keeps its own rule, then moves the new stores' monitor-hash episodes from
    'monitor' to 'held_out'. Idempotent; refuses any other corpus."""
    if (TR.ROOT / cfg["data"]["corpus"]).resolve() != UNION.resolve():
        raise ValueError("Not the union corpus")
    if getattr(D.load_split, "union", False):
        return
    D.RP.STORE = UNION
    D.RP.manifest.cache_clear(), D.RP._shard.cache_clear()
    n0, base = D.RP.manifest()["sources"][0]["shards"], D.load_split

    def load_split(c):
        s = base(c)
        s["held_out"] = [e for e in s["monitor"] if e.shard >= n0]
        s["monitor"] = [e for e in s["monitor"] if e.shard < n0]
        return s
    load_split.union = True
    D.load_split = load_split


def cfg_union():
    cfg = TR.load_cfg()
    cfg["data"]["corpus"], cfg["p1"]["out_dir"] = CORPUS, OUT_DIR
    return cfg


def _targets_setup():
    import lightgbm as lgb
    cfg = cfg_union(); use(cfg)
    train, n0 = D.load_split(cfg)["train"], D.RP.manifest()["sources"][0]["shards"]
    k = sum(e.shard < n0 for e in train)
    assert all(e.shard < n0 for e in train[:k]) and all(e.shard >= n0 for e in train[k:])
    return train, k, lgb.Booster(model_file=str(OLD / "change_teacher.txt"))


def _parts(train, k, n):
    """The new training episodes (k ..) in n contiguous parts of about equal frames."""
    frames = np.cumsum([e.T + 1 for e in train[k:]])
    return np.split(np.arange(k, len(train)), np.searchsorted(frames, np.linspace(0, frames[-1], n + 1)[1:-1]))


def build_part(i, n):
    """Part i of n of the new training episodes' targets (data.change_targets, unchanged) -> TARGETS/new/part_i."""
    train, k, model = _targets_setup()
    parts = _parts(train, k, n)
    out = TARGETS / "new" / f"part_{i}"
    out.mkdir(parents=True, exist_ok=True)
    D.change_targets([train[j] for j in parts[i]], out, model)
    (out / "done.json").write_text(json.dumps({"part": i, "of": n, "episodes": [int(parts[i][0]), int(parts[i][-1])]}) + "\n")


def assemble(n):
    """A bitwise reproduction of the original targets on a spread of original episodes by the frozen teacher, then the
    original targets followed by the n parts in order -> TARGETS/change_targets.npy; targets.json."""
    train, k, model = _targets_setup()
    off, old = D.frame_offsets(train), np.load(OLD / "change_targets.npy", mmap_mode="r")
    assert old.shape == (int(off[k]), 7, D.GRID), "original targets do not match the original train block"
    report = {"teacher_sha256": TR.file_sha256(OLD / "change_teacher.txt"), "original_targets_sha256": TR.file_sha256(OLD / "change_targets.npy"),
              "original_train_episodes": k, "new_train_episodes": len(train) - k, "frames": int(off[-1]), "parts": n, "reproduction": []}
    rep = TARGETS / "reproduce"
    rep.mkdir(exist_ok=True)
    for i in np.linspace(0, k - 1, 12).astype(int).tolist():            # the teacher reproduces the original targets
        r = np.asarray(D.change_targets([train[i]], rep, model))
        report["reproduction"].append({"episode": i, "bitwise": bool(np.array_equal(r.view(np.uint16), old[off[i]:off[i + 1]].view(np.uint16)))})
    assert all(x["bitwise"] for x in report["reproduction"]), report["reproduction"]
    (rep / "change_targets.npy").unlink()
    rep.rmdir()
    parts = _parts(train, k, n)
    out = np.lib.format.open_memmap(TARGETS / "change_targets.npy", "w+", np.float16, (int(off[-1]), 7, D.GRID))
    out[:off[k]] = old
    for i, p in enumerate(parts):
        d = TARGETS / "new" / f"part_{i}"
        assert json.loads((d / "done.json").read_text())["episodes"] == [int(p[0]), int(p[-1])]
        block = np.load(d / "change_targets.npy", mmap_mode="r")
        assert block.shape == (int(off[p[-1] + 1] - off[p[0]]), 7, D.GRID)
        out[off[p[0]]:off[p[-1] + 1]] = block
    out.flush()
    del out
    report["union_targets_sha256"] = TR.file_sha256(TARGETS / "change_targets.npy")
    for i in range(n):
        d = TARGETS / "new" / f"part_{i}"
        (d / "change_targets.npy").unlink(); (d / "done.json").unlink(); d.rmdir()
    (TARGETS / "new").rmdir()
    (HERE / "targets.json").write_text(json.dumps(report, indent=1) + "\n")
    print("targets", json.dumps(report), flush=True)


def verify():
    """Union identity and split checks against the original store and the dataset census (20261104_dataset_eda)."""
    cfg = cfg_union()
    orig = D.load_split(TR.load_cfg())                                   # the original store, before replay moves
    use(cfg)
    m, s = D.RP.manifest(), D.load_split(cfg)
    n0 = m["sources"][0]["shards"]
    res, key = {}, lambda e: (e.shard, e.slot, e.T, e.key)
    for name in ("monitor", "dev", "final"):
        res[f"{name}_identical_to_original"] = [key(e) for e in s[name]] == [key(e) for e in orig[name]]
    k = sum(e.shard < n0 for e in s["train"])
    res["original_train_is_prefix"] = [key(e) for e in s["train"][:k]] == [key(e) for e in orig["train"]]
    ids = [e.key for v in s.values() for e in v]
    res["episode_ids_unique"] = len(ids) == len(set(ids))
    res["train_episodes"], res["train_transitions"] = len(s["train"]), int(sum(e.T for e in s["train"]))
    res["census_train_matches"] = (res["train_episodes"], res["train_transitions"]) == (22267, 8914902)
    by = lambda v: {src: sum(m["shards"][e.shard]["source"] == src for e in v) for src in STORES}
    res["train_by_source"], res["held_out_by_source"] = by(s["train"]), by(s["held_out"])
    res["census_held_out_matches"] = [res["held_out_by_source"][x] for x in STORES[1:]] == [140, 625]
    res["dev_final_from_original_only"] = all(m["shards"][e.shard]["source"] == STORES[0] for v in ("dev", "final") for e in s[v])
    bad = []                                                              # every shard file's bytes against its source hash
    for i, sh in enumerate(m["shards"]):
        with open(UNION / sh["file"], "rb") as f:
            if hashlib.file_digest(f, "sha256").hexdigest() != sh["sha256"]:
                bad.append(i)
    res["shard_sha256_mismatches"] = bad
    for i in range(len(m["shards"])):                                    # every shard mapped at once in one process
        D.RP._shard(i)
    res["all_shards_mapped_open_fds"] = len(os.listdir("/proc/self/fd"))
    rng = np.random.default_rng(0)                                       # union reads equal direct source reads
    import torch
    for e in [s["train"][j] for j in rng.choice(len(s["train"]), 24, replace=False)]:
        sh = m["shards"][e.shard]
        src = json.loads((TR.ROOT / sh["source"] / "manifest.json").read_text())["shards"]
        direct = torch.load(TR.ROOT / sh["source"] / Path(sh["file"]).name, weights_only=False, mmap=True)["episodes"][e.slot]
        assert Path(sh["file"]).name in [x["file"] for x in src]
        assert torch.equal(direct["observations"], D.RP.episode_fields(e.shard, e.slot)["observations"]) and direct["episode_id"] == e.key
    res["union_reads_equal_source_reads"] = 24
    t = json.loads((HERE / "targets.json").read_text()) if (HERE / "targets.json").exists() else None
    if t:
        tg = np.load(TARGETS / "change_targets.npy", mmap_mode="r")
        res["targets_shape_matches"] = tg.shape == (int(D.frame_offsets(s["train"])[-1]), 7, D.GRID)
        res["targets_sha256_matches_build"] = TR.file_sha256(TARGETS / "change_targets.npy") == t["union_targets_sha256"]
    res["passed"] = all(v for k_, v in res.items() if isinstance(v, bool)) and not bad and t is not None
    (HERE / "union_checks.json").write_text(json.dumps(res, indent=1) + "\n")
    print("verify", json.dumps(res), flush=True)


if __name__ == "__main__":
    {"manifest": build_manifest, "part": lambda: build_part(int(sys.argv[2]), int(sys.argv[3])),
     "assemble": lambda: assemble(int(sys.argv[2])), "verify": verify}[sys.argv[1]]()
