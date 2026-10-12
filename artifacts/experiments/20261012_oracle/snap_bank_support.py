"""CPU-only support and metric checks for exact TRUE projection assignments.

Support is counted on all lost targets. Distances are descriptive on a fixed
sample of up to 64 actual semantic losses per cow/zombie type. It contrasts the
chosen token with the nearest bank token preserving terrain and all four mob
presence flags; this is not a contextual-coherence or physical-validity test.
"""
from __future__ import annotations

import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["JAX_PLATFORMS"] = "cpu"

import json
from pathlib import Path
import numpy as np
import torch
import snap_bank_diagnose as D


def summary(x):
    a = np.asarray(x, dtype=np.float64)
    return {"n": len(a), "min": float(a.min()), "median": float(np.median(a)),
            "mean": float(a.mean()), "max": float(a.max())} if len(a) else {"n": 0}


def main():
    torch.set_num_threads(8)
    out = D.OUT
    proof_path, readers_path = out / "snap_bank_diagnose.pt", out / "snap_bank_readers.pt"
    facts_path = out / "snap_visible.pt"
    batches = sorted((out / "snap_resume/TRUE").glob("batch_*.pt"))
    contract = {"format": "snap_bank_support_v1", "helper_sha256": D.sha256(Path(__file__)),
                "inputs": {str(p.relative_to(D.ROOT)): D.sha256(p) for p in (proof_path, readers_path, facts_path, D.META)},
                "batches": {str(p.relative_to(D.ROOT)): D.sha256(p) for p in batches},
                "support": "same terrain class and all four mob-presence flags, excluding source player31",
                "sample_seed": 20261012, "max_losses_per_type": 64,
                "interpretation": "dictionary support/metric diagnostic; not a causal context intervention"}
    proof = torch.load(proof_path, map_location="cpu", weights_only=False)
    readers = torch.load(readers_path, map_location="cpu", weights_only=False)
    facts = torch.load(facts_path, map_location="cpu", weights_only=False)
    pp = proof["per_token"]
    rep = torch.tensor([m[0] for m in proof["group_members"]])[pp["groups"].long()]
    source_cell = proof["source_cell"]
    source_tile = proof["source_tile"]
    source_mob = proof["source_mob"]
    bank = readers["bank"].float()
    bank_mob_pred = D.ridge(bank, readers["heads"]["mob"]) > torch.tensor(readers["thresholds"])
    tile = pp["target_tile"]
    target = pp["target_mob"]
    selected_source = source_mob[rep]
    projected = bank_mob_pred[rep]
    target_cell = torch.arange(len(tile)) % 63
    target_depth = (torch.arange(len(tile)) // 63) % 16
    eligible_query = target_cell != 31
    eligible_source = source_cell != 31
    bitweights = torch.tensor([1, 2, 4, 8], dtype=torch.long)
    bank_key = source_tile * 16 + (source_mob.long() * bitweights).sum(-1)
    target_key = tile * 16 + (target.long() * bitweights).sum(-1)
    count = torch.bincount(bank_key[eligible_source], minlength=272)
    count_all = torch.bincount(bank_key, minlength=272)
    slot_count = torch.bincount((bank_key * 63 + source_cell)[eligible_source], minlength=272 * 63).reshape(272, 63)
    selected_player = source_cell[rep] == 31
    report = {"contract": contract, "contract_id": D.digest(contract), "diagnose_contract_id": proof["contract_id"],
              "selected_source_player": {}, "support": {}, "fixed_sample_distance": {}}
    selected_indices = {}
    for j, name in enumerate(D.NAMES):
        sem_lost = eligible_query & target[:, j] & ~selected_source[:, j]
        read_lost = eligible_query & target[:, j] & ~projected[:, j]
        read_added = eligible_query & ~target[:, j] & projected[:, j]
        r = {}
        for depth_name, use in (("all", eligible_query), ("9-16", eligible_query & (target_depth >= 8))):
            r[depth_name] = {"queries": int(use.sum()), "queries_selecting_source_player": int((use & selected_player).sum()),
                "actual_semantic_losses": int((use & sem_lost).sum()),
                "semantic_losses_selecting_source_player": int((use & sem_lost & selected_player).sum()),
                "readout_false_negatives": int((use & read_lost).sum()),
                "false_negatives_selecting_source_player": int((use & read_lost & selected_player).sum()),
                "readout_false_positives": int((use & read_added).sum()),
                "false_positives_selecting_source_player": int((use & read_added & selected_player).sum())}
        report["selected_source_player"][name] = r
        if name not in ("cow", "zombie"):
            continue
        report["support"][name] = {}
        for loss_name, loss in (("actual_semantic_loss", sem_lost), ("readout_false_negative", read_lost)):
            report["support"][name][loss_name] = {}
            for depth_name, filt in (("all", torch.ones(len(tile), dtype=torch.bool)), ("9-16", target_depth >= 8)):
                ids = torch.where(loss & filt)[0]
                c = count[target_key[ids]]
                ca = count_all[target_key[ids]]
                cs = slot_count[target_key[ids], target_cell[ids]]
                report["support"][name][loss_name][depth_name] = {"lost_targets": len(ids),
                    "any_same_semantics_bank_available": int((ca > 0).sum()),
                    "nonplayer_same_semantics_available": int((c > 0).sum()),
                    "same_screen_slot_and_semantics_available": int((cs > 0).sum()),
                    "candidate_nonplayer_count": summary(c.numpy()), "candidate_same_slot_count": summary(cs.numpy())}
        ids = torch.where(sem_lost)[0].numpy()
        rng = np.random.default_rng(20261012 + j)
        selected_indices[name] = np.sort(rng.choice(ids, min(64, len(ids)), replace=False))

    # Fetch only the fixed loss sample and all player tokens from the saved TRUE batches.
    query = {}; players = []
    for path in batches:
        b = torch.load(path, map_location="cpu", weights_only=False)
        raw = b["modes"]["raw"].float()
        players.append(raw[:, :, 31].reshape(-1, 192))
        first, last = b["start"] * 16 * 63, b["end"] * 16 * 63
        flat = raw[:, :, :63].reshape(-1, 192)
        for name, ids in selected_indices.items():
            here = ids[(ids >= first) & (ids < last)]
            for ix in here.tolist():
                query[ix] = flat[ix - first].clone()
    bank_double = bank.double()
    bank_sq = bank_double.square().sum(-1)
    sample_rows = {}
    for name, ids in selected_indices.items():
        q = torch.stack([query[int(i)] for i in ids]).double()
        distance_sq = q.square().sum(-1, keepdim=True) - 2 * q @ bank_double.T + bank_sq[None]
        rows = []
        for ii, ix in enumerate(ids.tolist()):
            candidates = eligible_source & (bank_key == target_key[ix])
            same_slot = candidates & (source_cell == target_cell[ix])
            chosen = int(rep[ix])
            chosen_sq = float((q[ii] - bank_double[chosen]).square().sum())
            # Direct FP64 distances certify the selected minimizer's reported distance.
            correct = int(distance_sq[ii].masked_fill(~candidates, torch.inf).argmin()) if candidates.any() else None
            correct_sq = float((q[ii] - bank_double[correct]).square().sum()) if correct is not None else None
            slot = int(distance_sq[ii].masked_fill(~same_slot, torch.inf).argmin()) if same_slot.any() else None
            slot_sq = float((q[ii] - bank_double[slot]).square().sum()) if slot is not None else None
            rows.append({"flat_target_index": ix, "root": ix // (16 * 63), "depth": int(target_depth[ix]) + 1,
                         "cell": int(target_cell[ix]), "chosen_bank_index": chosen,
                         "correct_bank_index": correct, "correct_same_slot_bank_index": slot,
                         "chosen_squared_distance": chosen_sq, "nearest_correct_squared_distance": correct_sq,
                         "nearest_correct_same_slot_squared_distance": slot_sq,
                         "correct_to_chosen_euclidean_ratio": np.sqrt(correct_sq / chosen_sq) if correct_sq is not None and chosen_sq else None,
                         "chosen_has_same_terrain": bool(source_tile[chosen] == tile[ix]),
                         "chosen_source_player": bool(source_cell[chosen] == 31)})
        sample_rows[name] = rows
        available = [x for x in rows if x["nearest_correct_squared_distance"] is not None]
        report["fixed_sample_distance"][name] = {"sampled_losses": len(rows),
            "correct_semantic_candidate_available": len(available),
            "chosen_closer_than_correct_candidate": sum(x["chosen_squared_distance"] < x["nearest_correct_squared_distance"] for x in available),
            "correct_to_chosen_euclidean_ratio": summary([x["correct_to_chosen_euclidean_ratio"] for x in available]),
            "chosen_same_terrain_count": sum(x["chosen_has_same_terrain"] for x in rows),
            "chosen_source_player_count": sum(x["chosen_source_player"] for x in rows)}

    # Player-token targets are not covered by the map mob reader.
    player_raw = torch.cat(players)
    player_ids = torch.where(target_cell == 31)[0]
    player_projected = bank[rep[player_ids]]
    fv = facts["visible"][:, 16:32].flatten(0, 1)
    real_facing, real_sleep = fv[:, 1516:1520].argmax(-1), fv[:, 1520] > 0.5
    face_raw = D.ridge(player_raw, readers["heads"]["facing"]).argmax(-1)
    face_proj = D.ridge(player_projected, readers["heads"]["facing"]).argmax(-1)
    sleep_raw = D.ridge(player_raw, readers["heads"]["sleep"])[:, 0] > 0.5
    sleep_proj = D.ridge(player_projected, readers["heads"]["sleep"])[:, 0] > 0.5
    meta = torch.load(D.META, map_location="cpu", weights_only=False, mmap=True)
    source_fv = meta["future_visible"][proof["source_root"], 0, proof["source_time"]].float()
    source_face = source_fv[:, 1516:1520].argmax(-1)
    source_player_selection = source_cell[rep[player_ids]] == 31
    report["player_token"] = {"targets": len(player_ids),
        "select_source_player": int(source_player_selection.sum()),
        "select_source_nonplayer": int((~source_player_selection).sum()),
        "raw_facing_accuracy": float((face_raw == real_facing).float().mean()),
        "projected_facing_accuracy": float((face_proj == real_facing).float().mean()),
        "raw_sleep_accuracy": float((sleep_raw == real_sleep).float().mean()),
        "projected_sleep_accuracy": float((sleep_proj == real_sleep).float().mean()),
        "selected_player_source_facing_match_target": float((source_face[rep[player_ids]][source_player_selection] == real_facing[source_player_selection]).float().mean()),
        "facing_reader_error_on_selected_source_player": float((face_proj[source_player_selection] != source_face[rep[player_ids]][source_player_selection]).float().mean())}
    report["bank_source_slices"] = json.loads((out / "snap_bank_source_slices.json").read_text())
    D.atomic_json(report, out / "snap_bank_support.json")
    D.atomic_torch({"contract": contract, "contract_id": report["contract_id"], "samples": sample_rows}, out / "snap_bank_support.pt")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
