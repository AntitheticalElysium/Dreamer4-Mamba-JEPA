"""Seal an F2 arm's sources and inputs after its preflight passed (F2.1 here; F2.2 calls `seal`). Never starts training.
Starts from F2's own pin list (every file F2's launch depended on, at its CURRENT hash) and adds the arm's files."""
from datetime import datetime, timezone
import json
from pathlib import Path
from saccade import train as TR
HERE = Path(__file__).resolve().parent


def seal(here, names, inputs=()):
    files = set(json.loads((here.parent / '20261103_p1f2/SOURCE_PINS.json').read_text())['files'])
    files.update(str(p.relative_to(TR.ROOT)) for p in here.glob('*.py'))
    files.update(str((here / n).relative_to(TR.ROOT)) for n in names)
    files.update(str((TR.ROOT / p).relative_to(TR.ROOT)) for p in inputs)
    missing = [p for p in files if not (TR.ROOT / p).is_file()]
    if missing:
        raise FileNotFoundError(missing)
    manifest = dict(prepared_at_utc=datetime.now(timezone.utc).isoformat(), torch=TR.run_provenance(TR.load_cfg())['torch'],
                    files={p: TR.file_sha256(TR.ROOT / p) for p in sorted(files)})
    (here / 'SOURCE_PINS.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print('sealed', len(files), 'source/input files')


def main():
    seal(HERE, ('PLAN.md', 'checks.json', 'union_checks.json', 'targets.json'),
         ('artifacts/craftax_union_v1/manifest.json', 'artifacts/craftax_support_v2/manifest.json',
          'artifacts/craftax_coverage_v1/manifest.json', 'artifacts/craftax_risk_v1/manifest.json',
          'artifacts/saccade_m0/targets/change_teacher.txt', 'artifacts/saccade_m0_union/targets/change_targets.npy'))


if __name__ == '__main__':
    main()
