"""Collect every phase into one diagnosis record, separating measured location from cause.

This script previously predated phase 3b and did not read `memory_and_u.json`, so re-running
the documented pipeline would have silently overwritten a corrected diagnosis with an older,
wrong conclusion. It now requires every input it should summarise and refuses rather than
publishing a partial record, and it records the sha256 of each input it consumed.
"""

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
E = HERE / "evidence"

REQUIRED = {
    "ladder": "phase1/ladder.json",
    "controls": "phase1/controls.json",
    "health": "phase1/health.json",
    "anchors": "phase1/anchors.json",
    "label_sanity": "phase1/label_sanity.json",
    "rungs": "phase2/ladder.json",
    "memorization": "phase2/memorization.json",
    "dynamics": "phase3/dynamics.json",
    "memory_and_u": "phase3/memory_and_u.json",
}
OPTIONAL = {
    "transformer_dynamics": "phase4/transformer_dynamics.json",
    "derangement": "phase4/derangement_distribution.json",
    "health_controls": "phase4/health_controls.json",
    "u_memory": "phase4/u_world_memory.json",
    "confirmation": "phase4/confirmation_panel.json",
    "paired": "phase4/paired_intervals.json",
    "phase5_raw": "phase5/mamba_raw.json",
    "phase5_tc": "phase5/mamba_tc.json",
    "phase6_masked": "phase6/masked_reencode.json",
    "phase6_u": "phase6/u_world_internals.json",
}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--allow-missing", action="store_true",
                        help="publish a partial record; it is labelled partial and is not a diagnosis")
    args = parser.parse_args(argv)

    missing = [name for name, rel in REQUIRED.items() if not (E / rel).exists()]
    if missing and not args.allow_missing:
        print(json.dumps({"status": "refused", "missing": missing,
                          "why": "refusing to overwrite evidence/diagnosis.json without every required "
                                 "input; pass --allow-missing to publish an explicitly partial record"}))
        return 2

    loaded, inputs = {}, {}
    for name, rel in {**REQUIRED, **OPTIONAL}.items():
        path = E / rel
        if path.exists():
            loaded[name] = json.loads(path.read_text())
            inputs[rel] = sha(path)

    # ---- objective effect on identical rows -------------------------------------------------
    by_obj = defaultdict(list)
    for row in loaded.get("ladder", {}).get("rows", []):
        if not row["action_input"]:
            by_obj[(row["arm"], row["family"], row["objective"])].append(row["dev"]["safe_choice"])
    objective = {f"{a}|{f}": {o: round(float(np.mean(by_obj[(a, f, o)])), 1)
                              for o in ("bce_death", "rank") if (a, f, o) in by_obj}
                 for a, f, _ in by_obj}

    rung_table = defaultdict(dict)
    for row in loaded.get("rungs", {}).get("rows", []):
        rung_table[row["arm"]][row["rung"]] = {
            "dim": row["dim"], "mean_safe": round(float(np.mean(row["safe_choice"])), 1),
            "auc": round(float(np.mean(row["within_root_auc"])), 4),
            "health_r2": round(row["health_r2_linear"], 3)}

    ctrl = defaultdict(list)
    for row in loaded.get("controls", {}).get("fitted", []):
        ctrl[row["control"]].append(row["dev"]["safe_choice"])
    ctrl = {k: round(float(np.mean(v)), 1) for k, v in ctrl.items()}

    dyn = {arm: {r["condition"]: round(r["mean_safe"], 1) for r in payload["rows"]}
           for arm, payload in loaded.get("dynamics", {}).get("arms", {}).items()}
    b = loaded.get("memory_and_u", {})
    mem = {a: {r["condition"]: r["mean_safe"] for r in v["rows"]}
           for a, v in b.get("memory_interface", {}).items()}
    uworld = {k: {r["condition"]: r["mean_safe"] for r in v["rows"]}
              for k, v in b.get("u_world", {}).items() if k != "provenance"}

    record = {
        "schema": "d4mj_localization_diagnosis_v2",
        "status": "PARTIAL -- inputs missing" if missing else "complete",
        "inputs_consumed": inputs,
        "status_of_claims": "three measured SYMPTOMS; only the TC CLS->z degradation is tightly "
                            "component-localized. See `overstatements_withdrawn`.",
        "protocol": {
            "primary_metric": "safe-action choice on the 36 DEV roots offering both a fatal and a safe action",
            "supervision": "within-root pair ranking softplus(score_safe - score_fatal)",
            "seeds": 3, "family_for_rungs": "mlp128",
            "EXPLORATORY": "the objective and head were selected on the same 36-root DEV panel the "
                           "headline is reported on. Grouped inner-TRAIN selection and an untouched "
                           "confirmation panel are in phase 4; until those land these are exploratory.",
            "patch_cache_dtype": "float16 for patches/pooled taps; z and cls are float32"},
        "symptom_1_readout_objective": {"by_arm_family": objective,
            "conclusion": "a large part of the reported deficit was decision-objective mismatch in the "
                          "MEASUREMENT. BCE and ranking test different capabilities; this does not make "
                          "the BCE numbers meaningless."},
        "symptom_2_export_degradation": {"rungs": rung_table,
            "memorization_control": loaded.get("memorization", {}).get("rows"),
            "tightly_localized": "TC loses most of its signal between CLS (33.0) and z (21.0). That is the "
                                 "one component-exact localization in this package.",
            "raw_is_gentler": "raw declines patches 36.0 -> cls 31.3 -> z 29.7, spread across the path",
            "width_claim_scope": "192 post-hoc dimensions suffice ON THIS PANEL. This does NOT show an "
                                 "end-to-end trained 192-d latent has adequate capacity."},
        "symptom_3_generated_state_utility": {"by_arm": dyn, "memory_interface_c4": mem,
            "conclusion": "generated LeWM states show no demonstrated incremental safe-death ordering "
                          "beyond the root/history+action control. Direct's do."},
        "u_world_negative_result": {"by_world": uworld,
            "provenance": b.get("u_world", {}).get("provenance"),
            "role": "DIAGNOSTIC for Raw/TC, not a canonical architecture",
            "conclusion": "observed u is perfectly readable (36/36) while generated u does not beat its "
                          "control (23.0 vs 24.3). Replacing z with this existing pooled/PCA export does "
                          "not by itself solve the gate. It does NOT rule out every patch-derived world: "
                          "frozen TC encoder, one pooled representation, MSE-only, one seed, no joint "
                          "adaptation, no propagated token grid."},
        "controls": {"analytic": loaded.get("controls", {}).get("analytic"), "fitted_mean": ctrl},
        "overstatements_withdrawn": [
            "'spatial layout is the missing representation' -- WITHDRAWN. patch_mean discards token "
            "position and still reaches 34.7 (raw) / 32.7 (tc), so these data do not support a spatial "
            "arrangement requirement. The supported claim is narrower: final patch-token outputs carry a "
            "readable signal the learned CLS/projector export does not preserve equally.",
            "'the sequence mixer is exonerated' -- WITHDRAWN for this package. The Transformer arms were "
            "never run through the rank/dynamics/derangement ladder; the prior exoneration came from a "
            "different protocol.",
            "'health is not the explanation' -- WITHDRAWN. Direct's poor scalar health R^2 does not settle "
            "it: a representation can encode the dead/alive boundary sharply while regressing magnitude "
            "poorly. Binary health, HUD-crop and masked-HUD controls are required.",
            "'the predictor is not action-conditional' -- SOFTENED. It is architecturally action-conditioned. "
            "The evidence says its output adds no demonstrated safe-death ordering; generated u does lose "
            "23.0 -> 15.0 under derangement, so it is not action-blind.",
            "'192 dimensions rule out capacity' -- SCOPED to post-hoc representation width on this panel.",
            "'three separable causes' -- restated as three SYMPTOMS."],
        "phase5_internals": {a: loaded.get(f"phase5_{k}", {}).get("rows")
                             for a, k in (("mamba_raw", "raw"), ("mamba_tc", "tc"))},
        "phase6_masked_reencode": loaded.get("phase6_masked", {}).get("rows"),
        "phase6_u_internals": loaded.get("phase6_u", {}).get("rows"),
        "not_established": [
            "that SIGReg or joint training CAUSED the export degradation",
            "which stage of the predictor fails: pair/action projection, Mamba blocks, prediction "
            "projector, or the loss. The ladder stops at final h and generated output.",
            "anything about multistep rollout, achievements or closed-loop control",
            "generalization beyond this 36-root DEV panel"],
        "m4_authorized": False}
    (E / "diagnosis.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps({"status": record["status"], "inputs": len(inputs),
                      "path": str(E / "diagnosis.json")}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
