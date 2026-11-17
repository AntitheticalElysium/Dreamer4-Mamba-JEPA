"""Seal the F5-specific recipe, evidence and evaluation dependencies."""
from datetime import datetime, timezone
import json
from pathlib import Path

from saccade import train as TR
from . import train_p1f5 as launch

HERE = Path(__file__).resolve().parent


if __name__ == "__main__":
    inputs = {
        "artifacts/experiments/20261106_p1f4/run/ckpt_10.pt",
        "artifacts/experiments/20261109_p1_hud_change/run/ckpt_10.pt",
        "artifacts/experiments/20261109_p1_hud_change/consequences.py",
        "artifacts/experiments/20261109_p1_hud_change/natural_pair.py",
        "artifacts/experiments/20261106_determinant_curves/curves.py",
        "artifacts/experiments/20261106_determinant_routes/routes.py",
        "artifacts/experiments/20261106_decision_levels/crafting.py",
        "artifacts/experiments/20261106_decision_levels/frozen.py",
        "artifacts/experiments/20261030_actor_interface/interface.py",
        "artifacts/experiments/20261029_generation_diagnosis/field_calibration.py",
        "artifacts/experiments/20261101_regression_localization/trace.py",
        "artifacts/experiments/20261105_consequence_trace/trace.py",
        "artifacts/experiments/20261105_consequence_trace/recursion_steps.py",
        "artifacts/experiments/20261107_f4_followups/belief_probe.py",
        "artifacts/experiments/20261107_f4_followups/belief_path_probe.py",
        "artifacts/experiments/20261107_f4_followups/danger_checks.py",
        "artifacts/experiments/20261107_f4_followups/danger_readings.py",
        "artifacts/experiments/20261107_f4_followups/danger_m.py",
        "artifacts/experiments/20261107_f4_followups/routes_f4.py",
        "artifacts/experiments/20261107_f4_followups/readings/counters_dev.npz",
        "artifacts/experiments/20261108_full_frame_change/phase_bank.pt",
        "artifacts/experiments/20261108_full_frame_change/SINGLE_PLANE_QUALIFICATION.json",
        "artifacts/experiments/20261111_change_forecast_gate/gate.py",
        "artifacts/experiments/20261111_change_forecast_gate/CHECKS.json",
        "artifacts/experiments/20261104_p1f21/run.py",
        "artifacts/experiments/20261103_p1f2/chooser.json",
        "artifacts/experiments/20261103_p1f2/chooser.py",
        "artifacts/experiments/20261102_direct_chooser/direct.py",
        "artifacts/craftax_support_v2/manifest.json",
        "artifacts/eda/replay.py",
        "artifacts/eda/s0/s0_dev_labels.npz",
        "artifacts/eda/s0/s0_train400_labels.npz",
        "artifacts/eda/s0/s0_calib.json",
        "artifacts/eda/system_design_v1/acuity/acuity_precision.json",
    }
    provenance = TR.run_provenance(launch.recipe())
    inputs.update(provenance["source_sha256"])
    inputs.update(str(p.relative_to(TR.ROOT)) for p in (TR.ROOT / "saccade/eval").glob("*.py"))
    inputs.update(str(p.relative_to(TR.ROOT)) for p in HERE.glob("*.py"))
    for name in ("PLAN.md", "FORECAST_GATE.json", "checks.json", "eval_checks.json"):
        inputs.add(str((HERE / name).relative_to(TR.ROOT)))
    files = {p: TR.file_sha256(TR.ROOT / p) for p in sorted(inputs)}
    seal = {"prepared_at_utc": datetime.now(timezone.utc).isoformat(), "torch": provenance["torch"], "files": files}
    temp = HERE / "SOURCE_PINS.tmp"
    temp.write_text(json.dumps(seal, indent=2) + "\n")
    temp.replace(HERE / "SOURCE_PINS.json")
    print(f"sealed {len(files)} files")
