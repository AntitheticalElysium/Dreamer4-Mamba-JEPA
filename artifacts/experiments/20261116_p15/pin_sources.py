"""Seal what P1.5's service runs before its evaluation: the queue's whole import closure inside the repository (the
trainer, the recipes it builds on, the worker and its checkpoint rule) and the trainer's provenance files. The
evaluation, written during training, is not sealed here: it must not modify any sealed file."""
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

from saccade import train as TR
from . import run

HERE = Path(__file__).resolve().parent


if __name__ == "__main__":
    mods = [Path(m.__file__).resolve() for m in list(sys.modules.values()) if getattr(m, "__file__", None)]
    files = {str(p.relative_to(TR.ROOT)) for p in mods if p.is_relative_to(TR.ROOT) and ".venv" not in p.parts and p.is_file()}
    files |= set(TR.run_provenance(run.launch.recipe())["source_sha256"]) | {"artifacts/experiments/20261116_p15/checks.json"}
    seal = {"prepared_at_utc": datetime.now(timezone.utc).isoformat(), "torch": TR.run_provenance(run.launch.recipe())["torch"],
            "files": {p: TR.file_sha256(TR.ROOT / p) for p in sorted(files)}}
    (HERE / "SOURCE_PINS.tmp").write_text(json.dumps(seal, indent=2) + "\n")
    (HERE / "SOURCE_PINS.tmp").replace(HERE / "SOURCE_PINS.json")
    print(f"sealed {len(files)} files")
