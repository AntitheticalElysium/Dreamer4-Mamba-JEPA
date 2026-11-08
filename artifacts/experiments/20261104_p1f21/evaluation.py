"""The usual F2 battery against F2, plus the new held-out distributions. No training.

Run after the unchanged checkpoint rule. Each command has a separate log and a
persistent status; --dry-run prints the exact queue without touching its outputs.
Only the existing CPU uptake stage overlaps GPU work; its child is reaped on failure.
"""
import json
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
BASE = HERE.parent/'20261103_p1f2/run/ckpt_10.pt'
TOOLS = 'artifacts.experiments.20261104_p1f21.eval_tools'


def commands(checkpoint):
    checkpoint = Path(checkpoint).resolve()
    r = HERE/'readings'
    def legacy(module, *args):return ['legacy', module, *map(str, args)]
    steps = [
        ('s1', legacy('saccade.eval.s1', checkpoint, checkpoint.parent/'s1/results.json'), False),
        ('uptake', legacy('artifacts.experiments.20261101_p1f15.uptake', checkpoint, r/'uptake'), True),
        ('interface_random', legacy('artifacts.experiments.20261030_actor_interface.interface',
            r/'interface_random.json', '--policy=random', f'p1f2={BASE}', f'p1f21={checkpoint}'), False),
        ('interface_guided', legacy('artifacts.experiments.20261030_actor_interface.interface',
            r/'interface_guided.json', '--policy=guided', f'p1f2={BASE}', f'p1f21={checkpoint}'), False),
        ('m05', legacy('artifacts.experiments.20261030_m05_guided_fovea.evaluate',
            r/'eval_m05', f'base={BASE}', f'p1f21_1500={checkpoint}'), False),
        ('m05_summary', legacy('artifacts.experiments.20261030_m05_guided_fovea.readings',
            r/'eval_m05/results.json', 'base', 'p1f21_1500'), True),
        ('geometry', legacy('artifacts.experiments.20261030_m05_guided_fovea.dissect.token_geometry',
            r/'eval_m05', r/'token_geometry.json', 'base', 'p1f21_1500'), True),
        ('render', legacy('artifacts.experiments.20261030_decoder_diagnosis.refit',
            r/'render_p1f21.json', f'--ckpt={checkpoint}', '--trained-only'), False),
        ('codex_panel', ['codex_panel', str(checkpoint)], False),
        ('codex_cpu', ['codex_cpu', str(checkpoint)], True),
        ('codex_sampler', ['codex_sampler', str(checkpoint)], False),
        ('consistency', legacy('artifacts.experiments.20261101_p1f15.consistency',
            checkpoint, r/'consistency'), False),
        ('locus', legacy('artifacts.experiments.20261031_learned_fixation_sweep.locus',
            r/'locus', f'--ckpt=p1f21={checkpoint}'), False),
        ('paired_summary', ['summary', str(checkpoint)], True),
        ('fovea', ['fovea', str(checkpoint)], False),
        ('head', ['head', str(checkpoint)], False),
        ('gaze', ['gaze', str(checkpoint)], True),
    ]
    for source in ('dev', 'coverage', 'risk'):
        steps.append((f'own_{source}', ['own', str(checkpoint), source], False))
    return [(name, [sys.executable, '-u', '-m', TOOLS, *args], cpu) for name, args, cpu in steps]


def save(path, value):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2)+'\n'); temp.replace(path)


def main():
    checkpoint = Path(sys.argv[1]).resolve()
    queue = commands(checkpoint)
    if '--dry-run' in sys.argv:
        print(json.dumps(queue, indent=2)); return
    from .run import validate_sources
    validate_sources(HERE)
    r = HERE/'readings'; logs = r/'logs'
    logs.mkdir(parents=True, exist_ok=True)
    (checkpoint.parent/'s1').mkdir(exist_ok=True)
    # Locus summaries reuse immutable reference arrays, exactly as F2's queue did.
    locus = r/'locus'; locus.mkdir(exist_ok=True)
    refs = {
        'w1_10': HERE.parent/'20261031_learned_fixation_sweep/locus/w1_10.npz',
        'p1f_10': HERE.parent/'20261031_learned_fixation_sweep/locus/p1f_10.npz',
        'p1f15': HERE.parent/'20261101_p1f15/readings/locus/p1f15.npz',
        'p1f16': HERE.parent/'20261102_p1f16/readings/locus/p1f16.npz',
        'p1f2': HERE.parent/'20261103_p1f2/readings/locus/p1f2.npz',
    }
    for name, src in refs.items():
        if not src.is_file():raise FileNotFoundError(src)
        dest = locus/f'{name}.npz'
        if not dest.exists():dest.symlink_to(src.resolve())
        assert dest.resolve() == src.resolve()
    status = dict(checkpoint=str(checkpoint), phase='evaluation', steps={})
    path = r/'STATUS.json'
    uptake = None
    try:
        for name, command, cpu in queue:
            validate_sources(HERE)
            status['current'] = name
            status['steps'][name] = dict(started_at=datetime.now(timezone.utc).isoformat(), phase='running', command=command)
            save(path, status)
            env = dict(os.environ, OMP_NUM_THREADS='4', MKL_NUM_THREADS='4')
            if cpu:env['CUDA_VISIBLE_DEVICES'] = ''
            with open(logs/f'{name}.log', 'a') as log:
                if name == 'uptake':
                    uptake = subprocess.Popen(['nice', '-n', '10', *command], cwd=ROOT, env=env,
                                              stdout=log, stderr=subprocess.STDOUT)
                    status['steps'][name]['pid'] = uptake.pid
                    save(path, status)
                    continue
                subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
            status['steps'][name].update(phase='complete', completed_at=datetime.now(timezone.utc).isoformat())
            save(path, status)
        status['current'] = 'uptake'; save(path, status)
        if uptake.wait() != 0:raise RuntimeError('CPU uptake failed; see readings/logs/uptake.log')
        status['steps']['uptake'].update(phase='complete', completed_at=datetime.now(timezone.utc).isoformat())
        status.update(phase='complete', completed_at=datetime.now(timezone.utc).isoformat())
    except BaseException as e:
        if uptake is not None and uptake.poll() is None:
            uptake.terminate(); uptake.wait()
        status.update(phase='failed', error=repr(e), failed_at=datetime.now(timezone.utc).isoformat())
        if 'current' in status:status['steps'][status['current']]['phase'] = 'failed'
        raise
    finally:
        save(path, status)


if __name__ == '__main__':main()
