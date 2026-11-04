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
    # F2.2 imports these F2.1 helpers too. Seal shared code even outside `here`.
    files.update(str((HERE / p).relative_to(TR.ROOT)) for p in ('run.py', 'checks.py', 'pin_sources.py'))
    missing = [p for p in files if not (TR.ROOT / p).is_file()]
    if missing:
        raise FileNotFoundError(missing)
    manifest = dict(prepared_at_utc=datetime.now(timezone.utc).isoformat(), torch=TR.run_provenance(TR.load_cfg())['torch'],
                    files={p: TR.file_sha256(TR.ROOT / p) for p in sorted(files)})
    (here / 'SOURCE_PINS.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print('sealed', len(files), 'source/input files')


def main():
    extra = [str(p.relative_to(TR.ROOT)) for p in
             (HERE.parent/'20261104_f2_own_gaze').glob('*.py')]
    extra += [str(p.relative_to(TR.ROOT)) for p in (HERE.parent/'20261103_p1f2/readings/codex').glob('*.json')]
    extra += ['artifacts/experiments/20261103_p1f2/readings/fovea/identity.json',
              'artifacts/experiments/20261103_p1f2/readings/fovea/p1f2/local.npz',
              'artifacts/experiments/20261103_p1f2/readings/fovea/p1f16/local.npz',
              'artifacts/experiments/20261102_p1f15_readiness/uptake_local/raw.npz']
    extra += ['artifacts/experiments/20261103_p1f2/run/ckpt_10.pt',
              'artifacts/eda/s0/s0_calib.json',
              'artifacts/experiments/20261031_learned_fixation_sweep/locus/w1_10.npz',
              'artifacts/experiments/20261031_learned_fixation_sweep/locus/p1f_10.npz']
    extra += [f'artifacts/experiments/{folder}/readings/locus/{world}.npz' for folder, world in
              [('20261101_p1f15','p1f15'), ('20261102_p1f16','p1f16'), ('20261103_p1f2','p1f2')]]
    extra += [str((HERE/f'heldout_{n}.npz').relative_to(TR.ROOT)) for n in ('coverage', 'risk')]
    seal(HERE, ('PLAN.md', 'checks.json', 'union_checks.json', 'targets.json', 'heldout.json', 'eval_checks.json'),
         ('artifacts/craftax_union_v1/manifest.json', 'artifacts/craftax_support_v2/manifest.json',
          'artifacts/craftax_coverage_v1/manifest.json', 'artifacts/craftax_risk_v1/manifest.json',
          'artifacts/saccade_m0/targets/change_teacher.txt', 'artifacts/saccade_m0_union/targets/change_targets.npy', *extra))


if __name__ == '__main__':
    main()
