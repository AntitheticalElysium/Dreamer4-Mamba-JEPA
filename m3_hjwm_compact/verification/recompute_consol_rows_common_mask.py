"""Re-evaluate the eight committed consolidation 16k checkpoints with the
corrected common-union-mask symmetric_eval and PERSIST the rows (2026-07-15
companion finding: the common-mask metric repair landed in code after the
consolidation run, so consol_rows_*.json still carried the old per-target
fields; the corrected numbers existed only in the companion's rerun).

Overwrites reviews/artifacts/consol_rows_<arm>.json and appends a
"final_corrected" block per arm to consolidation.json. The evaluation bundle
is the archived seeds-63-78 set (fixed, identical for every arm)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

COMPACT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = COMPACT_ROOT.parent
sys.path.insert(0, str(COMPACT_ROOT))
sys.path.insert(0, str(COMPACT_ROOT / "verification"))

from model import ModelConfig  # noqa: E402
from ssl_ijepa import IJEPAPretrainer  # noqa: E402
from fork_oracle_v2 import ENCODER_CKPT, sha256_file  # noqa: E402
from consolidation import (  # noqa: E402
    ARTIFACTS, FINAL_BUNDLE, build_world, seed_level_summary, symmetric_eval)

ARMS = (
    ("C1_gru_s101", "gru", 192), ("C1_gru_s202", "gru", 192),
    ("C1_gru_s303", "gru", 192),
    ("C2_glob64_s101", "global_gru", 64), ("C2_glob64_s202", "global_gru", 64),
    ("C2_glob64_s303", "global_gru", 64),
    ("C3_gru_shuf", "gru", 192), ("C3_glob64_shuf", "global_gru", 64),
)


def main():
    device = torch.device("cuda")
    final_anchors = torch.load(FINAL_BUNDLE, weights_only=False)
    pretrainer = IJEPAPretrainer(
        ModelConfig(temporal_backend="gru", predictor="deterministic", mask_ratio=0.0))
    pretrainer.load_state_dict(
        torch.load(ENCODER_CKPT, weights_only=False)["pretrainer"], strict=True)
    encoder = pretrainer.target_encoder.to(device).eval()

    report = json.loads((ARTIFACTS / "consolidation.json").read_text())
    report.setdefault("hashes", {})["final_bundle_recheck"] = sha256_file(FINAL_BUNDLE)
    for name, backend, hidden in ARMS:
        ckpt = torch.load(ARTIFACTS / f"consol_{name}_16000.pt", weights_only=False)
        world = build_world(backend, hidden, device).eval()
        trainable = {n for n, p in world.named_parameters() if p.requires_grad}
        assert trainable == set(ckpt["trainable"]), \
            f"{name}: checkpoint/trainable name mismatch"
        with torch.no_grad():
            for n, p in world.named_parameters():
                if n in ckpt["trainable"]:
                    p.copy_(ckpt["trainable"][n].to(device))
        rows = symmetric_eval(world, encoder, final_anchors, device)
        (ARTIFACTS / f"consol_rows_{name}.json").write_text(json.dumps(rows))
        summary = {variant: seed_level_summary(rows, f"retrieval_{variant}")
                   for variant in ("all", "patch", "changed")}
        summary["separation_all"] = seed_level_summary(rows, "separation_all")
        summary["separation_patch"] = seed_level_summary(rows, "separation_patch")
        report["arms"][name]["final_corrected"] = summary
        print(f"[{name}] corrected changed retrieval {summary['changed']['mean']:.5f} "
              f"CI {[round(x, 5) for x in summary['changed']['ci95']]}", flush=True)
        del world
        torch.cuda.empty_cache()
    report["correction_note"] = (
        "2026-07-15: rows re-evaluated from committed 16k checkpoints with the "
        "common-union-mask symmetric_eval (per-target masks retained only as "
        "d_changed_targetmask_diagnostic); original 'final' summaries kept for "
        "the audit trail, 'final_corrected' is authoritative.")
    (ARTIFACTS / "consolidation.json").write_text(json.dumps(report, indent=2))
    print("saved", ARTIFACTS / "consolidation.json")


if __name__ == "__main__":
    main()
