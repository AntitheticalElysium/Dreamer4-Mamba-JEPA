"""Effective rank and scale across the joint budget, on the SEALED G1 DEV window ledger.

An earlier version of this measurement used an ad hoc 256-frame sample whose ledger and code were
never committed, and it reported the WRONG ORDERING at G1 (raw 7.44 vs TC 9.66). On the
predeclared windows the sealed screen records raw 23.45 against TC 9.87. A representation's
measured rank depends on how many and which frames it is measured over -- a richer representation
needs more diverse samples to reveal its dimensions -- so an unsealed sample is not evidence about
an arm's geometry. This reads the ledger G1 itself used.

Writes evidence/rank_trajectory.json with every checkpoint hash and the ledger hash.
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch

ROOT = Path("/home/antithetical/EPITA/PERSO/DynamicHorizons-Mamba-JEPA")
sys.path.insert(0, str(ROOT))
from d4mj.data import _sha256
from d4mj.world_api import load_bundle

STORES = (ROOT / "artifacts/craftax_expert_store_v1", ROOT / "artifacts/craftax_support_v2")


def ledger_frames(ledger, split="dev", frames_per=4):
    """The exact windows G1 screened, resolved to frames from the episode store."""
    # The ledger spans BOTH corpus sources, so both are indexed.
    index, manifests = {}, {}
    for store in STORES:
        manifests[store] = json.loads((store / "manifest.json").read_text())
        for position, record in enumerate(manifests[store]["shards"]):
            payload = torch.load(store / record["file"], weights_only=False, mmap=True)
            for slot, fields in enumerate(payload["episodes"]):
                index[fields["episode_id"]] = (store, position, slot)
            del payload
    cache, out = {}, []
    for episode_id, start in zip(ledger[split]["episode_ids"], ledger[split]["starts"]):
        store, shard, slot = index[episode_id]
        key = (store, shard)
        if key not in cache:
            cache[key] = torch.load(store / manifests[store]["shards"][shard]["file"],
                                    weights_only=False, mmap=True)
        observations = cache[key]["episodes"][slot]["observations"]
        out.append(observations[start:start + frames_per])
    return torch.stack(out)


@torch.no_grad()
def geometry(bundle, windows, batch=16):
    z = []
    for start in range(0, len(windows), batch):
        chunk = windows[start:start + batch].to(bundle.device)
        z.append(bundle.encoder(chunk)[:, :, 0].float().reshape(-1, bundle.config.encoder.latent_dim).cpu())
    z = torch.cat(z)
    centred = z - z.mean(0, keepdim=True)
    spectrum = torch.linalg.svdvals(centred) ** 2
    share = spectrum / spectrum.sum()
    rank = float(torch.exp(-(share * share.clamp_min(1e-12).log()).sum()))
    return {"frames": int(len(z)), "effective_rank": rank, "mean_abs_z": float(z.abs().mean()),
            "std_z": float(z.std())}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=ROOT / "artifacts/lewm_m4_canonical")
    parser.add_argument("--out", type=Path, default=ROOT / "artifacts/experiments/20260921_m4_baseline/evidence")
    parser.add_argument("--steps", type=int, nargs="+", default=[2000, 4000, 6000, 8000, 10000])
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)

    ledger_path = args.run / "G1/windows.json"
    ledger = json.loads(ledger_path.read_text())
    windows = ledger_frames(ledger)
    report = {"schema": "d4mj_rank_trajectory_v1",
              "ledger": {"path": str(ledger_path), "sha256": _sha256(ledger_path),
                         "split": "dev", "windows": len(ledger["dev"]["episode_ids"])},
              "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "arms": {}}
    for arm in ("raw", "tc"):
        rows = []
        for step in args.steps:
            path = args.run / arm / "joint" / f"step-{step:06d}.pt"
            if not path.is_file():
                continue
            bundle, _ = load_bundle(path)
            bundle.encoder.freeze()
            row = {"step": step, "checkpoint_sha256": _sha256(path), **geometry(bundle, windows)}
            rows.append(row)
            print(json.dumps({"arm": arm, **{k: v for k, v in row.items()
                                             if k != "checkpoint_sha256"}}), flush=True)
            del bundle
            torch.cuda.empty_cache()
        report["arms"][arm] = rows
    (args.out / "rank_trajectory.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"status": "complete", "ledger_sha256": report["ledger"]["sha256"][:16]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
