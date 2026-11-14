"""Seal F6.1's recipe, evidence and evaluation dependencies: everything F6 sealed except its token-forecast
instruments, plus this folder and the modules its sleep split reads."""
from datetime import datetime, timezone
import json
from pathlib import Path

from saccade import train as TR
from . import train as launch

HERE = Path(__file__).resolve().parent
EXP = "artifacts/experiments"


if __name__ == "__main__":
    inputs = set(json.loads((launch.F5.HERE / "SOURCE_PINS.json").read_text())["files"])
    inputs.update(f"{EXP}/20261113_token_forecast_gate/{n}.py" for n in ("gate", "gate2", "d1"))
    inputs.update(f"{EXP}/{n}.py" for n in ("20261107_danger_adversarial/sleep", "20261107_danger_adversarial/analyse",
                                            "20261106_determinant_curves/curves", "20261113_p1f6/train"))
    inputs.update(TR.run_provenance(launch.recipe())["source_sha256"])
    inputs.update(str(p.relative_to(TR.ROOT)) for p in (TR.ROOT / "saccade/eval").glob("*.py"))
    inputs.update(str(p.relative_to(TR.ROOT)) for p in launch.F5.HERE.glob("*.py"))
    inputs.update(str(p.relative_to(TR.ROOT)) for p in HERE.glob("*.py"))
    inputs.update(str((HERE / n).relative_to(TR.ROOT)) for n in ("PLAN.md", "checks.json", "eval_checks.json"))
    files = {p: TR.file_sha256(TR.ROOT / p) for p in sorted(inputs) if (TR.ROOT / p).exists()}
    missing = sorted(p for p in inputs if not (TR.ROOT / p).exists())
    assert not missing, missing
    seal = {"prepared_at_utc": datetime.now(timezone.utc).isoformat(), "torch": TR.run_provenance(launch.recipe())["torch"],
            "files": files}
    temp = HERE / "SOURCE_PINS.tmp"
    temp.write_text(json.dumps(seal, indent=2) + "\n")
    temp.replace(HERE / "SOURCE_PINS.json")
    print(f"sealed {len(files)} files")
