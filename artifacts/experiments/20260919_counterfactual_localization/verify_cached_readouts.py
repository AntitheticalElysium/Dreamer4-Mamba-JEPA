"""Audit published predictions without refitting probes or mutating their cache."""
import hashlib
import io
import json
import sqlite3
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).parent


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def thash(x):
    return hashlib.sha256(x.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()


def node(db, key):
    checksum, data, external, selector = db.execute(
        "SELECT sha256,payload,external,selector FROM nodes WHERE key=?", (key,)
    ).fetchone()
    if external:
        assert sha(external) == checksum
        value = torch.load(external, map_location="cpu", mmap=True, weights_only=False)
    else:
        assert hashlib.sha256(data).hexdigest() == checksum
        value = torch.load(io.BytesIO(data), map_location="cpu", weights_only=False)
    return value[selector] if selector else value


def features(run, arm, split):
    path = run / "features" / f"{arm}.{split}.pt"
    pointer = torch.load(path, map_location="cpu", weights_only=False)
    manifest = json.loads(path.with_suffix(".manifest.json").read_text())
    assert sha(path) == manifest["sha256"]
    if "features" in pointer:
        return pointer["features"], pointer["metadata"]["identity"]
    ref = pointer["cache_ref"]
    with sqlite3.connect(f"file:{ref['database']}?mode=ro", uri=True) as db:
        return node(db, ref["key"]), pointer["metadata"]["identity"]


def main():
    runs = {
        "mamba": ROOT / "artifacts/lewm_gates_20260906/m03_bootstrap/evaluation_v2",
        "transformer": ROOT / "artifacts/lewm_transformer_comparison/m03",
    }
    cache_path = ROOT / "artifacts/lewm_gates_20260906/cache.sqlite3"
    with sqlite3.connect(f"file:{cache_path}?mode=ro", uri=True) as db:
        probes = [(key, json.loads(dep)["dependencies"]) for key, dep in db.execute(
            "SELECT key,dependencies FROM nodes WHERE kind='_fit_probe_many'"
        )]
        result = {"method": "read-only cache; tensor input hashes and payload checksums; no probe refit",
                  "script_sha256": sha(__file__), "runs": {}}
        for label, run in runs.items():
            sidepath = run / "sidecar/sidecar.probe_only.pt"
            side = torch.load(sidepath, map_location="cpu", weights_only=False, mmap=True)
            rows = side["splits"]
            truth = rows["dev"]["outcomes"][..., 0].bool()
            eligible = truth.any(1) & (~truth).any(1)
            episodes = rows["dev"]["episode"]
            ty = rows["train"]["outcomes"].flatten(0, 1).float()
            r = {"sidecar_sha256": sha(sidepath), "dev_roots": len(truth),
                 "opportunity_roots": int(eligible.sum()),
                 "opportunity_episodes": int(episodes[eligible].unique().numel()),
                 "uniform_safe_rate": float((~truth)[eligible].float().mean()), "arms": {}}
            r["opportunity_panel"] = {}
            for split, rr in rows.items():
                yy = rr["outcomes"][..., 0].bool()
                ee = yy.any(1) & (~yy).any(1)
                health, counts = torch.unique(rr["root_continuous"][ee, 0], return_counts=True)
                indices = ee.nonzero().flatten().tolist()
                r["opportunity_panel"][split] = {
                    "roots": int(ee.sum()), "episodes": int(rr["episode"][ee].unique().numel()),
                    "terminal_tail_roots": sum(rr["stratum"][i] == "terminal_tail" for i in indices),
                    "root_health_counts": dict(zip(map(str, health.tolist()), counts.tolist())),
                    "front_lava_roots": int(rr["root_binary"][ee, 0].sum()),
                    "sleeping_roots": int(rr["root_binary"][ee, 8].sum()),
                    "safe_per_action": (~yy[ee]).sum(0).tolist(),
                    "safe_count_histogram": (~yy[ee]).sum(1).bincount(minlength=18).tolist(),
                }
            for arm in ("raw", "tc", "direct_mamba"):
                fs, ids = {}, {}
                for split in ("train", "dev"):
                    fs[split], ids[split] = features(run, arm, split)
                at = torch.eye(17).repeat(len(rows["train"]["outcomes"]), 1)
                ad = torch.eye(17).repeat(len(truth), 1)
                tx = torch.cat((fs["train"]["observed_successor"].flatten(0, 1).float(), at), 1)
                dx = torch.cat((fs["dev"]["observed_successor"].flatten(0, 1).float(), ad), 1)
                tx_hash, ty_hash, dx_hash = thash(tx), thash(ty), thash(dx)
                a = {"identity": ids["dev"], "feature_dimension": fs["dev"]["observed_successor"].shape[-1], "observed_fit_on_observed": {}}
                for hidden, family in ((False, "linear"), (True, "mlp")):
                    matches = []
                    for key, dep in probes:
                        inp = dep["inputs"]
                        if not inp.get("binary") or inp.get("hidden") != hidden:
                            continue
                        if inp["train_x"]["tensor"] != tx_hash or inp["train_y"]["tensor"] != ty_hash:
                            continue
                        for name, identity in inp["evaluation_x"].items():
                            if identity["tensor"] != dx_hash:
                                continue
                            logits = node(db, key)[name].reshape(len(truth), 17, -1)[..., 0]
                            selected = logits.argmin(1)
                            safe = (~truth)[torch.arange(len(truth)), selected]
                            ties = (logits == logits.min(1, keepdim=True).values).sum(1)
                            pair = logits[:, :, None] < logits[:, None, :]
                            tied = logits[:, :, None] == logits[:, None, :]
                            valid_pairs = (~truth)[:, :, None] & truth[:, None, :]
                            auc = ((pair.float() + .5 * tied.float()) * valid_pairs).sum((1, 2)) / valid_pairs.sum((1, 2)).clamp_min(1)
                            movement_score, movement_truth = logits[:, 1:5], truth[:, 1:5]
                            movement_eligible = movement_truth.any(1) & (~movement_truth).any(1)
                            movement_chosen = movement_score.argmin(1) + 1
                            movement_safe = (~truth)[torch.arange(len(truth)), movement_chosen]
                            movement_pairs = (~movement_truth)[:, :, None] & movement_truth[:, None, :]
                            movement_comparisons = (movement_score[:, :, None] < movement_score[:, None, :]).float()
                            movement_comparisons += .5 * (movement_score[:, :, None] == movement_score[:, None, :]).float()
                            movement_auc = (movement_comparisons * movement_pairs).sum((1, 2)) / movement_pairs.sum((1, 2)).clamp_min(1)
                            matches.append({"cache_key": key, "evaluation_key": name,
                                "implementation": dep["implementation"], "settings": inp["settings"],
                                "correct": int(safe[eligible].sum()), "rate": float(safe[eligible].float().mean()),
                                "tied_minimum_roots": int((ties[eligible] > 1).sum()),
                                "macro_within_root_auc": float(auc[eligible].mean()),
                                "movement_diagnostic": {
                                    "warning": "restricted action set; diagnostic only; does not repair the 17-action gate",
                                    "correct_on_original_opportunities": int(movement_safe[eligible].sum()),
                                    "rate_on_original_opportunities": float(movement_safe[eligible].float().mean()),
                                    "movement_opportunity_roots": int(movement_eligible.sum()),
                                    "within_movement_auc": float(movement_auc[movement_eligible].mean()),
                                    "uniform_on_original_opportunities": float((~movement_truth)[eligible].float().mean()),
                                },
                                "selected_action_histogram": torch.bincount(selected[eligible], minlength=17).tolist()})
                    assert matches, (label, arm, family, "no published observed prediction matched exact input bytes")
                    assert len({(m["correct"], m["rate"]) for m in matches}) == 1, (label, arm, family, "ambiguous predictions")
                    a["observed_fit_on_observed"][family] = matches
                r["arms"][arm] = a
                print(label, arm, {k: v[0]["correct"] for k, v in a["observed_fit_on_observed"].items()}, flush=True)
            result["runs"][label] = r
    (OUT / "cached_readout_audit.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
