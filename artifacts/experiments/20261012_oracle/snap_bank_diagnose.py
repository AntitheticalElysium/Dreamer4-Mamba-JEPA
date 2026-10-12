"""Exact CPU dictionary-assignment diagnosis of TRUE nearest-token projection.

No nearest-neighbor search or encoder evaluation is performed. Projected token
bytes are matched to the saved real-token dictionary, then its associated real
source facts are compared with independently replayed target facts. Duplicate
token bytes can have different source metadata; semantic bounds retain all such
assignments rather than pretending the selected physical source is identified.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["JAX_PLATFORMS"] = "cpu"

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "artifacts/eda/oracle_v1"
META = ROOT / "artifacts/eda/diagnosis_rollouts_v1/meta.pt"
CACHE = ROOT / "artifacts/eda/levers_futures_tokens_raw.pt"
NAMES = ("zombie", "skeleton", "cow", "arrow")
GROUPS = {"1": [0], "2-4": [1, 2, 3], "5-8": [4, 5, 6, 7], "9-16": list(range(8, 16))}


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def digest(x):
    return hashlib.sha256(json.dumps(x, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def atomic_torch(x, path):
    temp = path.with_name(path.name + f".tmp-{os.getpid()}")
    with temp.open("wb") as f:
        torch.save(x, f)
        f.flush(); os.fsync(f.fileno())
    os.replace(temp, path)


def atomic_json(x, path):
    temp = path.with_name(path.name + f".tmp-{os.getpid()}")
    with temp.open("w") as f:
        json.dump(x, f, indent=2)
        f.write("\n"); f.flush(); os.fsync(f.fileno())
    os.replace(temp, path)


def ridge(x, head):
    z = ((x.float() - head["mu"]) / head["sd"]).double()
    return (z @ head["w"][:-1] + head["w"][-1]).float()


def ratio(a, b):
    return float(a / b) if b else None


def score(pred, true):
    tp = int((pred & true).sum()); fp = int((pred & ~true).sum()); fn = int((~pred & true).sum())
    return {"true": int(true.sum()), "predicted": int(pred.sum()), "tp": tp, "fp": fp, "fn": fn,
            "recall": ratio(tp, tp + fn), "precision": ratio(tp, tp + fp),
            "census_ratio": ratio(int(pred.sum()), int(true.sum()))}


def main():
    torch.set_num_threads(8)
    reader_path = OUT / "snap_bank_readers.pt"
    target_path = OUT / "snap_visible.pt"
    panel_path = OUT / "panel.pt"
    inputs = [reader_path, target_path, panel_path, META, CACHE]
    batches = sorted((OUT / "snap_resume/TRUE").glob("batch_*.pt"))
    if not batches:
        raise ValueError("No completed TRUE projection batches")
    contract = {"format": "snap_bank_dictionary_assignment_v1", "helper_sha256": sha256(Path(__file__)),
                "inputs": {str(p.relative_to(ROOT)): sha256(p) for p in inputs},
                "batches": {str(p.relative_to(ROOT)): sha256(p) for p in batches},
                "interpretation": "dictionary assignment, not a physical successor prediction",
                "light_match_absolute_tolerance": 0.001}
    contract_id = digest(contract)
    readers = torch.load(reader_path, map_location="cpu", weights_only=False)
    facts = torch.load(target_path, map_location="cpu", weights_only=False)
    panel = torch.load(panel_path, map_location="cpu", weights_only=False, mmap=True)
    assert facts["roots"] == panel["roots"]
    for p in inputs[1:]:
        rel = str(p.relative_to(ROOT))
        expected = readers["source_contract"]["inputs"].get(rel)
        if expected is not None:
            assert contract["inputs"][rel] == expected, f"Source input drift: {rel}"
    bank = readers["bank"].float().contiguous()
    assert bank.shape == (60000, 192)
    selected = readers["selected"].long()
    train_ids = readers["train_ids"].long()
    ntime = readers["fut_shape"][1]
    local_root = selected // (ntime * 63)
    time = (selected // 63) % ntime
    cell = selected % 63
    root = train_ids[local_root]
    cache = torch.load(CACHE, map_location="cpu", weights_only=False, mmap=True)
    cache_bank = cache["fut"][root, time, cell].float()
    assert torch.equal(cache_bank, bank), "Dictionary no longer equals recorded TRAIN tokens"
    del cache_bank, cache
    meta = torch.load(META, map_location="cpu", weights_only=False, mmap=True)
    source_vis = meta["future_visible"][root, 0, time].float()
    source_tile = source_vis[:, :1071].reshape(-1, 63, 17)[torch.arange(len(bank)), cell].argmax(-1)
    mm = source_vis[:, 1071:1512].reshape(-1, 63, 7)[torch.arange(len(bank)), cell] > 0
    source_mob = torch.stack([mm[:, 0], mm[:, 2], mm[:, 1], mm[:, 3:].any(-1)], -1)
    source_light, source_sleep = source_vis[:, 1521], source_vis[:, 1520] > 0.5
    bank_tile_pred = ridge(bank, readers["heads"]["tile"]).argmax(-1)
    bank_mob_score = ridge(bank, readers["heads"]["mob"])
    bank_mob_pred = bank_mob_score > torch.tensor(readers["thresholds"])

    # A group contains all source rows whose FP32 token bytes are identical.
    lookup = {}; members = []; bank_group = np.empty(len(bank), dtype=np.int32)
    for i, token in enumerate(bank.numpy()):
        key = token.tobytes()
        group = lookup.get(key)
        if group is None:
            group = len(members); lookup[key] = group; members.append([])
        members[group].append(i); bank_group[i] = group
    source_tile_np = source_tile.numpy(); source_mob_np = source_mob.numpy()
    source_light_np = source_light.numpy(); source_sleep_np = source_sleep.numpy(); cell_np = cell.numpy()
    group_mob_min = np.stack([source_mob_np[g].all(0) for g in members])
    group_mob_max = np.stack([source_mob_np[g].any(0) for g in members])
    group_tile_possible = np.zeros((len(members), 17), dtype=bool)
    group_tile_ambiguous = np.zeros(len(members), dtype=bool)
    for i, ids in enumerate(members):
        labels = np.unique(source_tile_np[ids]); group_tile_possible[i, labels] = True
        group_tile_ambiguous[i] = len(labels) > 1
    representative = np.array([g[0] for g in members], dtype=np.int64)
    duplicate_groups = [g for g in members if len(g) > 1]
    bank_proof = {"tokens": len(bank), "unique_token_bytes": len(members),
                  "duplicate_groups": len(duplicate_groups),
                  "duplicate_source_entries": sum(len(g) for g in duplicate_groups),
                  "largest_group": max(map(len, members)),
                  "groups_with_tile_ambiguity": int(group_tile_ambiguous.sum()),
                  "groups_with_mob_ambiguity": int((group_mob_min != group_mob_max).any(-1).sum()),
                  "cache_bank_bitwise_equal": True,
                  "reader_error_on_actual_source": {
                      "tile_error_count": int((bank_tile_pred != source_tile).sum()),
                      "tile_accuracy": float((bank_tile_pred == source_tile).float().mean()),
                      "mobs": {n: score(bank_mob_pred[:, j], source_mob[:, j]) for j, n in enumerate(NAMES)}}}
    print(json.dumps({"stage": "bank", "proof": bank_proof}), flush=True)
    parts = []
    resume = OUT / "snap_bank_diagnose_batches"
    resume.mkdir(exist_ok=True)
    occupied = torch.zeros(len(facts["visible"]), dtype=torch.bool)
    upstream_contract_ids = set()
    for batch_path in batches:
        b = torch.load(batch_path, map_location="cpu", weights_only=False)
        start, end = b["start"], b["end"]
        assert not occupied[start:end].any(), "Overlapping TRUE batches"
        occupied[start:end] = True
        upstream_contract_ids.add(b["contract_id"])
        out_part = resume / batch_path.name
        if out_part.exists():
            part = torch.load(out_part, map_location="cpu", weights_only=False)
            assert part["contract_id"] == contract_id, "Diagnostic resume contract drift"
        else:
            raw = b["modes"]["raw"].float()
            projected = b["modes"]["readout"].float()
            assert torch.equal(raw, panel["tokens"][start:end, 16:32].float()), "TRUE batch differs from panel tokens"
            assert torch.equal(raw[:, :, 63:], projected[:, :, 63:]), "Projection unexpectedly changed HUD tokens"
            q = projected[:, :, :63].reshape(-1, 192).contiguous().numpy()
            groups = np.array([lookup.get(x.tobytes(), -1) for x in q], dtype=np.int32)
            assert (groups >= 0).all(), "Projected token not bitwise present in dictionary"
            target = facts["visible"][start:end, 16:32]
            target_tile = target[..., :1071].reshape(-1, 63, 17).argmax(-1).reshape(-1)
            tm = target[..., 1071:1512].reshape(-1, 63, 7) > 0
            target_mob = torch.stack([tm[..., 0], tm[..., 2], tm[..., 1], tm[..., 3:].any(-1)], -1).reshape(-1, 4)
            cells = np.tile(np.arange(63), len(raw) * 16)
            lights = target[..., 1521].flatten().repeat_interleave(63).numpy()
            sleeps = (target[..., 1520].flatten().repeat_interleave(63) > 0.5).numpy()
            same_position_any = np.zeros(len(q), bool); same_position_all = np.zeros(len(q), bool)
            same_light_any = np.zeros(len(q), bool); same_light_all = np.zeros(len(q), bool)
            same_sleep_any = np.zeros(len(q), bool); same_sleep_all = np.zeros(len(q), bool)
            for i, group in enumerate(groups):
                ids = members[group]
                p = cell_np[ids] == cells[i]
                l = np.abs(source_light_np[ids] - lights[i]) <= 0.001
                s = source_sleep_np[ids] == sleeps[i]
                same_position_any[i], same_position_all[i] = p.any(), p.all()
                same_light_any[i], same_light_all[i] = l.any(), l.all()
                same_sleep_any[i], same_sleep_all[i] = s.any(), s.all()
            raw_flat = raw[:, :, :63].reshape(-1, 192)
            part = {"contract_id": contract_id, "start": start, "end": end,
                    "groups": torch.from_numpy(groups), "target_tile": target_tile,
                    "target_mob": target_mob, "raw_tile_pred": ridge(raw_flat, readers["heads"]["tile"]).argmax(-1),
                    "raw_mob_score": ridge(raw_flat, readers["heads"]["mob"]),
                    "same_position_any": torch.from_numpy(same_position_any),
                    "same_position_all": torch.from_numpy(same_position_all),
                    "same_light_any": torch.from_numpy(same_light_any), "same_light_all": torch.from_numpy(same_light_all),
                    "same_sleep_any": torch.from_numpy(same_sleep_any), "same_sleep_all": torch.from_numpy(same_sleep_all)}
            atomic_torch(part, out_part)
        parts.append(part)
        print(json.dumps({"stage": "batch", "complete_roots": end, "total_roots": len(occupied)}), flush=True)
    assert occupied.all(), "Incomplete TRUE batches"
    assert len(upstream_contract_ids) == 1, "Mixed upstream projection contracts"
    p = {k: torch.cat([x[k] for x in parts]) for k in parts[0] if isinstance(parts[0][k], torch.Tensor)}
    groups = p["groups"].numpy()
    rep = representative[groups]
    target_tile = p["target_tile"].numpy(); target_mob = p["target_mob"].numpy()
    projected_tile = bank_tile_pred.numpy()[rep]
    projected_mob = bank_mob_pred.numpy()[rep]
    source_min, source_max = group_mob_min[groups], group_mob_max[groups]
    source_tile_possible = group_tile_possible[groups, target_tile]
    source_tile_certain = source_tile_possible & ~group_tile_ambiguous[groups]
    raw_mob = (p["raw_mob_score"] > torch.tensor(readers["thresholds"])).numpy()
    depth = np.tile(np.repeat(np.arange(16), 63), len(occupied))
    cells = np.tile(np.arange(63), len(occupied) * 16)
    report = {"contract": contract, "contract_id": contract_id, "bank": bank_proof,
              "upstream_contract_id": next(iter(upstream_contract_ids)),
              "matched_projected_tokens": len(groups), "unmatched_tokens": 0,
              "duplicate_token_assignments": int(np.array([len(members[g]) > 1 for g in groups]).sum()),
              "readings": {}}
    for name, ks in {"all": list(range(16)), **GROUPS}.items():
        use = np.isin(depth, ks); mob_use = use & (cells != 31)
        tile_correct = projected_tile == target_tile
        base_tile = p["raw_tile_pred"].numpy() == target_tile
        conf = np.zeros((17, 17), np.int64)
        np.add.at(conf, (target_tile[use], source_tile_np[rep][use]), 1)
        row = {"tokens_including_player": int(use.sum()), "tokens_excluding_player": int(mob_use.sum()),
               "actual_tile_preservation_bounds": [float(source_tile_certain[use].mean()), float(source_tile_possible[use].mean())],
               "raw_tile_reader_accuracy": float(base_tile[use].mean()), "projected_tile_reader_accuracy": float(tile_correct[use].mean()),
               "source_tile_confusion_representative": conf.tolist(),
               "tile_reader_error_given_certain_correct_source": ratio(int((use & source_tile_certain & ~tile_correct).sum()), int((use & source_tile_certain).sum())),
               "mobs": {}}
        for prop in ("position", "light", "sleep"):
            row[f"source_{prop}_match_bounds"] = [float(p[f"same_{prop}_all"][use].float().mean()),
                                                  float(p[f"same_{prop}_any"][use].float().mean())]
        for j, mob in enumerate(NAMES):
            t, d = target_mob[:, j], projected_mob[:, j]
            smin, smax = source_min[:, j], source_max[:, j]
            lost = mob_use & t & ~d
            added = mob_use & ~t & d
            ntrue = int((mob_use & t).sum())
            nt = int(mob_use.sum())
            actual_preservation = [ratio(int((mob_use & t & smin).sum()), ntrue),
                                   ratio(int((mob_use & t & smax).sum()), ntrue)]
            certain_no_mob = ~source_max.any(-1)
            certain_other = ~smax & np.delete(source_min, j, axis=1).any(-1)
            semantic_lost = mob_use & t & ~smax
            readers_correct_source = mob_use & smin
            row["mobs"][mob] = {"raw_reader": score(raw_mob[mob_use, j], t[mob_use]),
                                  "projected_reader": score(d[mob_use], t[mob_use]),
                                  "actual_source_preservation_recall_bounds": actual_preservation,
                                  "actual_source_false_added_bounds": [int((mob_use & ~t & smin).sum()), int((mob_use & ~t & smax).sum())],
                                  "ambiguous_source_fact_count": int((mob_use & (smin != smax)).sum()),
                                  "readout_lost_breakdown": {
                                      "lost_total": int(lost.sum()), "wrong_source_definite": int((lost & ~smax).sum()),
                                      "correct_source_reader_false_negative": int((lost & smin).sum()),
                                      "ambiguous_source": int((lost & (smin != smax)).sum())},
                                  "readout_false_added_breakdown": {
                                      "added_total": int(added.sum()), "wrong_source_definite": int((added & smin).sum()),
                                      "no_source_mob_reader_false_positive": int((added & ~smax).sum()),
                                      "ambiguous_source": int((added & (smin != smax)).sum())},
                                  "definite_semantic_loss_destination": {
                                      "lost_total": int(semantic_lost.sum()),
                                      "no_mob": int((semantic_lost & certain_no_mob).sum()),
                                      "other_mob": int((semantic_lost & certain_other).sum()),
                                      "ambiguous_other_mob": int((semantic_lost & ~certain_no_mob & ~certain_other).sum())},
                                  "projected_reader_error_against_source_fact_bounds": [
                                      ratio(int((mob_use & ((d & ~smax) | (~d & smin))).sum()), nt),
                                      ratio(int((mob_use & ((d & ~smin) | (~d & smax))).sum()), nt)],
                                  "reader_false_negative_on_certain_source_mob": ratio(int((readers_correct_source & ~d).sum()), int(readers_correct_source.sum()))}
        report["readings"][name] = row
    atomic_torch({"contract_id": contract_id, "per_token": p, "bank_group": torch.from_numpy(bank_group),
                  "group_members": members, "source_root": root, "source_time": time, "source_cell": cell,
                  "source_tile": source_tile, "source_mob": source_mob,
                  "source_light": source_light, "source_sleep": source_sleep}, OUT / "snap_bank_diagnose.pt")
    atomic_json(report, OUT / "snap_bank_diagnose.json")
    print(json.dumps({"stage": "complete", "contract_id": contract_id,
                      "bank": bank_proof, "depth9_16": report["readings"]["9-16"]}), flush=True)


if __name__ == "__main__":
    main()
