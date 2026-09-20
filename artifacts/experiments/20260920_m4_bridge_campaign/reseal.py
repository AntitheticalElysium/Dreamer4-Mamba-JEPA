"""Checklist 0.2b: re-measure frozen-evaluation parity across the pre-M4 and M4 trees.

The M4 bridge edits four files inside the `sources.py` runtime closure, so every sealed M0-M3
checkpoint stops matching the tree exactly. That is correct for *training resume*, which must never
continue a run under a different sampler, and `checkpoint.py` keeps refusing it. It is too strong
for *frozen evaluation*, which never calls the sampler -- so `load_m03_bundle` consults a measured
parity proof instead, and this produces the third one.

Procedure, exactly as `d4mj/m03/README.md` documents it:

  1. a worktree at the pre-M4 tag is the reference tree
  2. `d4mj/m03/` is copied into it -- that directory is deliberately OUTSIDE the runtime closure,
     so carrying this measurement code in does not perturb the manifest being measured
  3. each tree is dumped TWICE, because `advance` is not reproducible on a cold Triton autotune
     state and a cross-tree gap means nothing until the within-tree spread bounds it
  4. `frozen_eval_proof` turns the four dumps into a hash-pinned record

The proof passes only when `cross_tree_max_abs <= max(tolerance, within_tree_max_abs)`. If it
fails, the M4 edits changed frozen-evaluation numerics and the design is wrong -- not the guard.
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import torch

ROOT = Path("/home/antithetical/EPITA/PERSO/DynamicHorizons-Mamba-JEPA")
HERE = Path(__file__).parent
TAG = "lewm-closure-m3"
ARMS = {"raw": "artifacts/lewm_gates_20260906/paired/raw/joint/step-010000.pt",
        "tc": "artifacts/lewm_gates_20260906/paired/tc/joint/step-010000.pt"}


def dump(tree: Path, out: Path, device: str) -> None:
    """One tree's frozen-evaluation surface, written from inside that tree."""
    script = (
        "import sys, torch; sys.path.insert(0, %r)\n"
        "from d4mj.m03.gate import frozen_eval_parity\n"
        "from pathlib import Path\n"
        "arms = {k: Path(%r) / v for k, v in %r.items()}\n"
        "torch.save(frozen_eval_parity(arms, %r, True), %r)\n"
    ) % (str(tree), str(ROOT), ARMS, device, str(out))
    environment = dict(os.environ, TRITON_F32_DEFAULT="ieee", JAX_PLATFORMS="cpu",
                       PYTHONPATH=str(tree))
    subprocess.run([str(ROOT / ".venv/bin/python"), "-c", script], check=True, cwd=str(tree),
                   env=environment)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worktree", type=Path, default=Path("/tmp/d4mj-closure-m3"))
    parser.add_argument("--dumps", type=Path, default=HERE / "cache/parity")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--tolerance", type=float, default=1e-5)
    parser.add_argument("--runs", type=int, default=2)
    args = parser.parse_args(argv)
    args.dumps.mkdir(parents=True, exist_ok=True)

    if not args.worktree.exists():
        subprocess.run(["git", "-C", str(ROOT), "worktree", "add", "--detach",
                        str(args.worktree), TAG], check=True)
    # m03/ is outside the runtime closure, so carrying the measurement code into the reference
    # tree leaves that tree's manifest untouched -- the property this whole approach rests on.
    subprocess.run(["rsync", "-a", "--delete", str(ROOT / "d4mj/m03") + "/",
                    str(args.worktree / "d4mj/m03") + "/"], check=True)
    # `third_party/sources/` and the vendored checkouts are gitignored, so a fresh worktree has
    # none of the bytes `lewm_source_manifest` hashes or the repositories it runs rev-parse on.
    # Link them rather than copy: the manifest must read the SAME bytes in both trees, and the
    # measurement is meaningless if the reference tree pins a different checkout.
    for entry in sorted((ROOT / "third_party").iterdir()):
        link = args.worktree / "third_party" / entry.name
        if not link.exists():
            link.symlink_to(entry)
    lock = ROOT / "requirements-lewm-rtx3060.lock.txt"
    if lock.exists() and not (args.worktree / lock.name).exists():
        (args.worktree / lock.name).symlink_to(lock)

    paths = {"reference": [], "current": []}
    for label, tree in (("reference", args.worktree), ("current", ROOT)):
        for run in range(args.runs):
            out = args.dumps / f"{label}-{run}.pt"
            if not out.exists():
                print(json.dumps({"stage": "dump", "tree": label, "run": run}), flush=True)
                dump(tree, out, args.device)
            paths[label].append(out)

    sys.path.insert(0, str(ROOT))
    from d4mj.m03.gate import frozen_eval_proof
    proof = frozen_eval_proof(paths["reference"], paths["current"], args.tolerance)
    print(json.dumps({"stage": "proof", "status": proof["status"],
                      "parity": proof["parity"],
                      "changed": sorted(proof["changed_runtime_files"])}, indent=1), flush=True)
    if proof["status"] != "pass":
        raise SystemExit("frozen-eval parity FAILED: the M4 edits changed evaluation numerics")

    record = ROOT / "d4mj/m03/frozen_eval_compat.json"
    document = json.loads(record.read_text())
    proofs = list(document["proofs"]) if isinstance(document.get("proofs"), list) else [document]
    digest = proof["current_manifest_digest"]
    # The proof describing the CURRENT tree goes first. `_frozen_eval_delta` and
    # `load_m03_bundle` both take the first covering record, and the test fixture exercises
    # `records[0]`, so a stale tree's proof sitting in front of the live one fails everything.
    proofs = [proof] + [p for p in proofs if p.get("current_manifest_digest") != digest]
    document["proofs"] = proofs
    document["note"] = ("one proof per reference tree -- checkpoints sealed against different "
                        "trees need their own measurement, and each is verified in full and on "
                        "its own")
    document["regenerated_unix"] = time.time()
    record.write_text(json.dumps(document, indent=2) + "\n")
    print(json.dumps({"status": "resealed", "proofs": len(proofs)}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
