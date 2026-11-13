"""Seal F6's recipe, evidence and evaluation dependencies: everything F5 sealed, plus this folder and the gate's
instruments its clock readings reuse."""
from datetime import datetime, timezone
import json
from pathlib import Path

from saccade import train as TR
from . import train as launch

HERE = Path(__file__).resolve().parent
GATE = "artifacts/experiments/20261113_token_forecast_gate"


if __name__ == "__main__":
    inputs = set(json.loads((launch.F5.HERE / "SOURCE_PINS.json").read_text())["files"])
    inputs.update(f"{GATE}/{n}.py" for n in ("gate", "gate2", "gate3", "d1"))
    inputs.update(TR.run_provenance(launch.recipe())["source_sha256"])
    inputs.update(str(p.relative_to(TR.ROOT)) for p in (TR.ROOT / "saccade/eval").glob("*.py"))
    inputs.update(str(p.relative_to(TR.ROOT)) for p in launch.F5.HERE.glob("*.py"))
    inputs.update(str(p.relative_to(TR.ROOT)) for p in HERE.glob("*.py"))
    inputs.update(str((HERE / n).relative_to(TR.ROOT)) for n in ("PLAN.md", "checks.json", "eval_checks.json"))
    files = {p: TR.file_sha256(TR.ROOT / p) for p in sorted(inputs)}
    seal = {"prepared_at_utc": datetime.now(timezone.utc).isoformat(), "torch": TR.run_provenance(launch.recipe())["torch"],
            "files": files}
    temp = HERE / "SOURCE_PINS.tmp"
    temp.write_text(json.dumps(seal, indent=2) + "\n")
    temp.replace(HERE / "SOURCE_PINS.json")
    print(f"sealed {len(files)} files")
