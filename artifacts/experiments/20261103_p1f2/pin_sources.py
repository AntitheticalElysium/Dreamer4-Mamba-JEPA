"""Seal source and fixed-evaluation inputs after preflight. Never starts training."""
from datetime import datetime, timezone
import importlib
import json
from pathlib import Path
from saccade import train as TR
HERE=Path(__file__).resolve().parent


def main():
    files=set(json.loads((HERE.parent/'20261102_p1f16/SOURCE_PINS.json').read_text())['files'])
    files.update(str(p.relative_to(TR.ROOT)) for p in HERE.glob('*.py'))
    files.update(str((HERE/n).relative_to(TR.ROOT)) for n in ('chooser.json','PLAN.md','queue.sh','checks.json','eval_smoke.json','unit_tests.log'))
    for rel in ('20261102_direct_chooser/direct.py','20261103_chooser_development/common.py',
        '20261103_p1f16_review/local_uptake.py','20261102_p1f15_readiness/uptake_local.py',
        '20261102_p1f15_readiness/analyze_uptake.py'):
        p=HERE.parent/rel
        if p.exists():files.add(str(p.relative_to(TR.ROOT)))
    # Existing immutable metadata/results used for paired comparisons, not new dev fitting.
    inputs=[HERE.parent/'20261102_p1f16/run/ckpt_10.pt',
        HERE.parent/'20261102_p1f16/run/s1/results.reader.pt',
        HERE.parent/'20261102_p1f16/run/s1/results.json',
        HERE.parent/'20261103_p1f16_review/local_identity.json',
        HERE.parent/'20261103_p1f16_review/p1f16/local.npz',
        HERE.parent/'20261103_p1f16_review/p1f16/readers.pt',
        HERE.parent/'20261102_p1f15_readiness/uptake_local/raw.npz']
    inputs.extend(TR.ROOT/'artifacts/eda/s0'/n for n in ('s0_train400_labels.npz','s0_dev_labels.npz'))
    for name in ('saved','confirmation'):inputs.append(HERE.parent/f'20261102_direct_chooser/predictions_611601_{name}.npz')
    inputs.extend((HERE.parent/'20261102_p1f16/readings/codex').glob('*'))
    for p in inputs:
        if not p.is_file():raise FileNotFoundError(p)
    files.update(str(p.relative_to(TR.ROOT)) for p in inputs)
    manifest=dict(prepared_at_utc=datetime.now(timezone.utc).isoformat(),torch=TR.run_provenance(TR.load_cfg())['torch'],
        files={p:TR.file_sha256(TR.ROOT/p) for p in sorted(files)})
    (HERE/'SOURCE_PINS.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print('sealed',len(files),'source/input/qualification files')


if __name__=='__main__':main()
