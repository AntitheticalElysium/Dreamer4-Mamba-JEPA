"""Separating real-token projection diagnostic; see NOTEBOOK 10-12 follow-up.

Same 60k FIT-diagnosis bank and D checkpoints as the failed run. Measure raw,
readout-only projection, pre-projection output under projected feedback, and
projected feedback output. TRUE measures projection damage. Bank membership
is not a guarantee of a valid joint frame. Atomic root batches are resumable.
"""
import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[3]
for d in ("20260927_levers", "20260926_diagnosis", "20260921_readout_ladder", "20261012_oracle"):
    sys.path.insert(0, str(ROOT / "artifacts/experiments" / d))
import oracle_eval as OE
OUT, H, W, SEED = OE.OUT, 16, 15, 20261012
GROUPS = {"1": [0], "2-4": list(range(1, 4)), "5-8": list(range(4, 8)), "9-16": list(range(8, 16))}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def atomic(value, path, json_format=False):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w" if json_format else "wb") as f:
        json.dump(value, f) if json_format else torch.save(value, f)
        f.flush(); os.fsync(f.fileno())
    os.replace(tmp, path)


def log(**row):
    print(json.dumps(row), flush=True)


class Projector:
    def __init__(self, bank, device="cuda", query_chunk=256, bank_chunk=8192):
        self.values = bank.float().to(device)
        self.bank = self.values.double()
        self.norms = self.bank.square().sum(-1)
        self.query_chunk, self.bank_chunk = query_chunk, bank_chunk

    @torch.no_grad()
    def indices(self, queries):
        ids = []
        for start in range(0, len(queries), self.query_chunk):
            q = queries[start:start + self.query_chunk].to(self.bank.device).double()
            norm = q.square().sum(-1, keepdim=True)
            best = torch.full((len(q),), float("inf"), device=q.device, dtype=q.dtype)
            index = torch.zeros(len(q), device=q.device, dtype=torch.long)
            for b in range(0, len(self.bank), self.bank_chunk):
                sub = self.bank[b:b + self.bank_chunk]
                distance = (norm - 2 * q @ sub.T + self.norms[b:b + len(sub)][None]).clamp_min_(0)
                value, local = distance.min(-1)
                # Recompute candidate distances directly: cancellation in the
                # norm expansion otherwise breaks duplicate/self-match ties.
                value = (q - sub[local]).square().sum(-1)
                better = value < best  # first bank index wins ties
                index = torch.where(better, local + b, index)
                best = torch.minimum(best, value)
            ids.append(index.cpu())
        return torch.cat(ids)

    @torch.no_grad()
    def __call__(self, frames):
        ids = self.indices(frames[:, :63].reshape(-1, 192))
        out = frames.clone()
        out[:, :63] = self.values[ids.to(self.values.device)].cpu().view(len(frames), 63, 192)
        assert torch.equal(out[:, 63:], frames[:, 63:]), "projection changed HUD"
        return out


def projector_checks(device):
    g = torch.Generator().manual_seed(SEED)
    bank = torch.randn(127, 192, generator=g); bank[126] = bank[0]
    q = torch.cat([bank[:13], torch.randn(21, 192, generator=g)])
    expected = (q.double()[:, None] - bank.double()[None]).square().sum(-1).argmin(-1)
    # Duplicate bank rows can have different IDs under FP64 cancellation;
    # compare the selected token values and their actual squared distances.
    for qc, bc in ((7, 19), (34, 127)):
        found = Projector(bank, device, qc, bc).indices(q)
        assert torch.equal(bank[found], bank[expected]), "nearest token mismatch"
        assert torch.equal((q.double() - bank[found].double()).square().sum(-1),
                           (q.double() - bank[expected].double()).square().sum(-1))
    p = Projector(bank, device, 7, 19)
    f = torch.randn(2, 81, 192, generator=g); projected = p(f)
    assert torch.equal(projected, p(projected)), "projection not idempotent"
    log(stage="projection_checks", exhaustive_equal=True, chunk_equal=True, idempotent=True, hud_equal=True)


def export_reader(fn):
    outer = dict(zip(fn.__code__.co_freevars, [c.cell_contents for c in fn.__closure__]))
    xs = outer["xs"]
    inner = dict(zip(xs.__code__.co_freevars, [c.cell_contents for c in xs.__closure__]))
    return {"mu": inner["mu"], "sd": inner["sd"], "w": outer["w"]}


class Readers:
    def __init__(self, state):
        self.state, self.thr = state["heads"], torch.tensor(state["thresholds"])

    def __getattr__(self, name):
        if name not in self.state:
            raise AttributeError(name)
        s = self.state[name]
        return lambda q: (torch.cat([((q.float() - s["mu"]) / s["sd"]).double(),
                                     torch.ones(len(q), 1, dtype=torch.float64)], 1) @ s["w"]).float()


def bank_and_readers(source_contract):
    import teval as T
    import scoreboard as SB
    path = OUT / "snap_bank_readers.pt"
    if path.exists():
        st = torch.load(path, weights_only=False, map_location="cpu")
        assert st["source_contract"] == source_contract, "bank/reader source drift"
        log(stage="bank_resume", bank_rows=len(st["bank"]))
        return st
    meta, train_roots, train_seeds = T.split()
    cache = T.build_cache("raw", torch.device("cpu"))
    train_ids = torch.where(train_roots)[0]
    fut = cache["fut"][train_ids][:, :, :63].flatten(0, 2)
    selected = torch.randperm(len(fut), generator=torch.Generator().manual_seed(SEED))[:60000]
    probes = SB.AllProbes(T, cache, meta, train_roots, train_seeds)
    state = {"source_contract": source_contract, "bank": fut[selected].float(), "selected": selected,
             "train_ids": train_ids, "train_seeds": train_seeds, "fut_shape": list(cache["fut"].shape),
             "heads": {k: export_reader(getattr(probes, k)) for k in ("tile", "mob", "hud", "facing", "sleep", "light")},
             "thresholds": probes.thr}
    frozen = Readers(state)
    for k in ("tile", "mob", "facing", "sleep"):
        q = state["bank"][:64]
        assert torch.equal(getattr(frozen, k)(q), getattr(probes, k)(q)), k
    atomic(state, path)
    log(stage="bank_created", bank_rows=len(selected), heads_equal=True)
    return state


@torch.no_grad()
def trajectory(world, oc, tok, act, lay, config, projector=None):
    frames = [tok[:, j].float() for j in range(1, 16)]
    before, after = [], []
    for k in range(1, H + 1):
        emitted = OE.step(world, oc, torch.stack(frames[-W:], 1), act[:, k:k + W],
                          lay[:, k + 1:k + 1 + W], torch.device("cuda"), config)
        fed = projector(emitted) if projector else emitted
        before.append(emitted); after.append(fed); frames.append(fed)
    return torch.stack(before, 1), torch.stack(after, 1)


def alignment(pred, root, true_offsets):
    from scroll import estimate
    from driftanat import offsets
    previous = torch.cat([root[:, None], pred[:, :-1]], 1)
    shifts = torch.stack([estimate(previous[:, k], pred[:, k]) for k in range(H)], 1)
    return (offsets(shifts) == true_offsets).all(-1)


def counts(pred, truth, previous_truth, readers):
    R = len(pred)
    scores = readers.mob(pred[:, :, OE.MAPC].reshape(-1, 192).float()).view(R, H, 62, 4)
    decoded = torch.zeros(R, H, 63, 3, dtype=torch.bool)
    for j, name in enumerate(OE.TYPES):
        decoded[:, :, OE.MAPC, j] = scores[..., OE.PROBE_CH[name]] > readers.thr[OE.PROBE_CH[name]]
    arrived, left = truth & ~previous_truth, previous_truth & ~truth
    return {k: v.sum(2).tolist() for k, v in {"true": truth, "predicted": decoded,
             "true_positive": decoded & truth, "false_positive": decoded & ~truth,
             "arrived": arrived, "arrive_true_positive": decoded & arrived,
             "left": left, "residue": decoded & left}.items()}


def aggregate(count, mask):
    result = {}
    for name, ks in GROUPS.items():
        m = mask[:, ks, None]
        total = {k: (torch.tensor(v)[:, ks] * m).sum((0, 1)) for k, v in count.items()}
        row = {}
        for j, t in enumerate(OE.TYPES):
            ratio = lambda n, d: float(total[n][j] / total[d][j]) if total[d][j] else None
            row[t] = {"census": ratio("predicted", "true"), "recall": ratio("true_positive", "true"),
                      "precision": ratio("true_positive", "predicted"), "fp_per_true": ratio("false_positive", "true"),
                      "arrive_recall": ratio("arrive_true_positive", "arrived"), "residue": ratio("residue", "left"),
                      "n_true": int(total["true"][j]), "n_arrive": int(total["arrived"][j]), "n_left": int(total["left"][j])}
        result[name] = row
    return result


def preservation(raw, projected, readers, layouts):
    import scoreboard as SB
    a, b = raw.flatten(0, 1), projected.flatten(0, 1)
    ta = readers.tile(a[:, OE.MAPC].reshape(-1, 192)).view(len(a), 62, 17).argmax(-1)
    tb = readers.tile(b[:, OE.MAPC].reshape(-1, 192)).view(len(a), 62, 17).argmax(-1)
    idx = {c: j for j, c in enumerate(OE.MAPC)}
    regions = {"all": OE.MAPC, "near": SB.NEAR4, "interior": SB.INTER, "edge": SB.EDGE}
    out = {"hud_bit_equal": torch.equal(a[:, 63:], b[:, 63:]),
           "tile_reader_agreement": {k: float((ta[:, [idx[c] for c in v]] == tb[:, [idx[c] for c in v]]).float().mean()) for k, v in regions.items()},
           "tile_reader_agreement_by_original_class": {str(j): float((tb[ta == j] == j).float().mean()) if (ta == j).any() else None for j in range(17)},
           "facing_reader_agreement": float((readers.facing(a[:, 31]).argmax(-1) == readers.facing(b[:, 31]).argmax(-1)).float().mean()),
           "sleep_reader_agreement": float(((readers.sleep(a[:, 31])[:, 0] > .5) == (readers.sleep(b[:, 31])[:, 0] > .5)).float().mean()),
           "map_token_mse": float((a[:, :63] - b[:, :63]).square().mean()),
           "player_token_mse": float((a[:, 31] - b[:, 31]).square().mean())}
    target = layouts[:, 16:32, :, 3].flatten(0, 1)[:, OE.MAPC] > 0
    out["arrow"] = {}
    for key, frames in (("raw", a), ("readout", b)):
        d = readers.mob(frames[:, OE.MAPC].reshape(-1, 192)).view(len(a), 62, 4)[..., 3] > readers.thr[3]
        out["arrow"][key] = {"n_true": int(target.sum()), "n_predicted": int(d.sum()),
                              "n_true_positive": int((d & target).sum()), "n_false_positive": int((d & ~target).sum())}
    return out


def semantic_metrics(pred, readers, visible):
    import scoreboard as SB
    frames = pred.flatten(0, 1).float()
    facts = SB.AllProbes.facts(visible[:, 16:32].flatten(0, 1))
    labels = readers.tile(frames[:, OE.MAPC].reshape(-1, 192)).view(len(frames), 62, 17).argmax(-1)
    correct = labels == facts["tile"]
    idx = {c: j for j, c in enumerate(OE.MAPC)}
    regions = {"all": OE.MAPC, "near": SB.NEAR4, "interior": SB.INTER, "edge": SB.EDGE}
    out = {"tile_accuracy": {k: float(correct[:, [idx[c] for c in cells]].float().mean()) for k, cells in regions.items()},
           "tile_accuracy_by_true_class": {str(j): float(correct[facts["tile"] == j].float().mean()) if (facts["tile"] == j).any() else None for j in range(17)},
           "tile_confusion": torch.bincount((facts["tile"] * 17 + labels).flatten(), minlength=17 * 17).view(17, 17).tolist(),
           "facing_accuracy": float((readers.facing(frames[:, 31]).argmax(-1) == facts["facing"]).float().mean()),
           "sleep_accuracy": float(((readers.sleep(frames[:, 31])[:, 0] > .5) == (facts["sleep"] > .5)).float().mean())}
    hp = readers.hud(frames[:, 63:81].flatten(1))
    out["hud"] = {}
    for j, name in enumerate(("health", "food", "drink", "energy")):
        p, y = hp[:, j], facts["hud"][:, j]
        out["hud"][name] = {"r2": float(1 - (p - y).square().sum() / (y - y.mean()).square().sum().clamp_min(1e-9)),
                              "bias": float((p - y).mean()), "mae": float((p - y).abs().mean())}
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("worlds", nargs="+")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--query-chunk", type=int, default=256)
    parser.add_argument("--bank-chunk", type=int, default=8192)
    parser.add_argument("--stop-after-batches", type=int)
    args = parser.parse_args()
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    started = time.time(); projector_checks("cuda")
    relevant = [Path(__file__), Path(OE.__file__), ROOT / "artifacts/experiments/20261012_oracle/oracle_world.py"]
    relevant += [ROOT / "artifacts/experiments/20260927_levers" / f"{n}.py" for n in ("tworld", "teval", "scoreboard", "scroll", "driftanat")]
    source = {str(p.relative_to(ROOT)): sha(p) for p in relevant}
    inputs = {str(p.relative_to(ROOT)): sha(p) for p in (OUT / "panel.pt", OUT / "snap_visible.pt", ROOT / "artifacts/eda/levers_futures_tokens_raw.pt", ROOT / "artifacts/eda/diagnosis_rollouts_v1/meta.pt")}
    bank_state = bank_and_readers({"sources": source, "inputs": inputs})
    readers = Readers(bank_state)
    projector = Projector(bank_state["bank"], "cuda", args.query_chunk, args.bank_chunk)
    check = torch.cat([bank_state["bank"][:63].view(1, 63, 192), torch.zeros(1, 18, 192)], 1)
    assert torch.equal(projector(check), projector(projector(check))), "real-bank idempotence"
    panel = torch.load(OUT / "panel.pt", map_location="cpu", weights_only=False)
    visible_state = torch.load(OUT / "snap_visible.pt", map_location="cpu", weights_only=False)
    assert visible_state["roots"] == panel["roots"], "semantic replay root mismatch"
    visible = visible_state["visible"]
    tok, lay, act = panel["tokens"], panel["layouts"], panel["actions"]
    from scroll import estimate
    from driftanat import offsets
    from d4mj.config import config_from_dict
    import spatial as Sp
    config = config_from_dict(torch.load(Sp.CHECKPOINT, map_location="cpu", weights_only=False)["config"])
    shifts = torch.stack([estimate(tok[:, k].float(), tok[:, k + 1].float()) for k in range(31)], 1)
    true_offsets = offsets(shifts[:, 15:31])
    truth = torch.stack([lay[..., OE.LAYOUT_CH[t]] > 0 for t in OE.TYPES], -1); truth[:, :, 31] = False
    new_batches = 0
    for argument in args.worlds:
        name, path = argument.split("=", 1)
        contract = {"format": "oracle_snap_contract_v1", "sources": source, "inputs": inputs,
                    "bank_sha256": sha(OUT / "snap_bank_readers.pt"), "world": name,
                    "checkpoint_sha256": sha(path) if path != "TRUE" else None,
                    "roots": panel["roots"], "batch_size": args.batch_size, "seed": SEED,
                    "projection": "exhaustive L2 FP64 distances; 60k unchanged FP32 real tokens; all63map; HUD unchanged",
                    "query_chunk": args.query_chunk, "bank_chunk": args.bank_chunk}
        contract_id = hashlib.sha256(json.dumps(contract, sort_keys=True).encode()).hexdigest()
        directory = OUT / "snap_resume" / name; directory.mkdir(parents=True, exist_ok=True)
        manifest = directory / "contract.json"
        if manifest.exists():
            assert json.loads(manifest.read_text()) == contract, "resume contract drift"
        else:
            atomic(contract, manifest, True)
        world = oc = None; pieces = []
        for start in range(0, len(tok), args.batch_size):
            end = min(start + args.batch_size, len(tok)); dest = directory / f"batch_{start:04d}_{end:04d}.pt"
            if dest.exists():
                piece = torch.load(dest, map_location="cpu", weights_only=False)
                assert piece["contract_id"] == contract_id and piece["start"] == start and piece["end"] == end
                log(stage="batch_resume", world=name, roots_done=end, roots_total=len(tok))
            else:
                sl = slice(start, end)
                if path == "TRUE":
                    raw = tok[sl, 16:32].float(); projected = projector(raw.flatten(0, 1)).view_as(raw)
                    modes = {"raw": raw, "readout": projected}
                else:
                    if world is None: world, oc = OE.load(path, torch.device("cuda"))
                    raw, _ = trajectory(world, oc, tok[sl], act[sl], lay[sl], config)
                    readout = projector(raw.flatten(0, 1)).view_as(raw)
                    pre, post = trajectory(world, oc, tok[sl], act[sl], lay[sl], config, projector)
                    assert torch.equal(raw[:, 0], pre[:, 0]), "depth1 changed before feedback"
                    modes = {"raw": raw, "readout": readout, "feedback_pre": pre, "feedback_projected": post}
                raw_alignment = alignment(modes["raw"], tok[sl, 15].float(), true_offsets[sl])
                aligns = {"raw": raw_alignment, "readout": raw_alignment}
                if path != "TRUE":
                    fa = alignment(modes["feedback_projected"], tok[sl, 15].float(), true_offsets[sl])
                    aligns.update(feedback_pre=fa, feedback_projected=fa)
                piece = {"contract_id": contract_id, "start": start, "end": end, "modes": modes, "aligned": aligns}
                atomic(piece, dest)
                restored = torch.load(dest, map_location="cpu", weights_only=False)
                assert all(torch.equal(restored["modes"][m], v) for m, v in modes.items()), "resume changed predictions"
                piece = restored; new_batches += 1
                log(stage="batch_done", world=name, roots_done=end, roots_total=len(tok), seconds=round(time.time() - started, 1), gpu_peak_gb=round(torch.cuda.max_memory_allocated() / 2**30, 3))
            pieces.append(piece)
            if args.stop_after_batches and new_batches >= args.stop_after_batches:
                log(stage="interruption_check", world=name, roots_done=end, resumable=True)
                return
        if world is not None: del world, oc; torch.cuda.empty_cache()
        predictions = {m: torch.cat([p["modes"][m] for p in pieces]) for m in pieces[0]["modes"]}
        masks = {m: torch.cat([p["aligned"][m] for p in pieces]) for m in pieces[0]["aligned"]}
        stats = {"format": "oracle_snap_stats_v1", "world": name, "types": list(OE.TYPES), "depths": list(range(1, H + 1)), "roots": panel["roots"], "contract": contract, "modes": {}}
        result = {"world": name, "contract_id": contract_id, "modes": {}}
        common = torch.stack(list(masks.values())).all(0)
        for m, pred in predictions.items():
            row = counts(pred, truth[:, 16:32], truth[:, 15:31], readers)
            stats["modes"][m] = {"counts": row, "aligned": masks[m].tolist()}
            result["modes"][m] = {"aligned_share": masks[m].float().mean(0).tolist(), "all": aggregate(row, torch.ones_like(common)), "common_aligned": aggregate(row, common)}
            result["modes"][m]["semantic"] = semantic_metrics(pred, readers, visible)
        result["projection_preservation"] = preservation(predictions["raw"], predictions["readout"], readers, lay)
        atomic(stats, OUT / f"stats_{name}.json", True); atomic(result, OUT / f"snap_{name}.json", True)
        log(stage="world_done", world=name, roots=len(tok), seconds=round(time.time() - started, 1), late_cow={m: result["modes"][m]["all"]["9-16"]["cow"] for m in result["modes"]})
        del predictions, pieces
    log(stage="done", seconds=round(time.time() - started, 1))


if __name__ == "__main__":
    main()
