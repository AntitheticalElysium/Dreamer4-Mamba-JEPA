"""P1-MultiStep-Mamba readout: one subprocess per instrument, in order, on the selected checkpoint.
Includes full evaluation battery + belief timer probes.
"""
import os
os.environ.setdefault('JAX_PLATFORMS', 'cpu')
import json
from pathlib import Path
import subprocess
import sys
from saccade import train as TR

HERE = Path(__file__).resolve().parent
CT = 'artifacts.experiments.20261105_consequence_trace'
F4F = 'artifacts.experiments.20261107_f4_followups'


def steps(ckpt):
    ck = Path(ckpt).resolve()
    rel = str(ck.relative_to(TR.ROOT))
    r = HERE / 'readings'
    return [
        ('s1', ['-m', 'saccade.eval.s1', str(ck), str(ck.parent / 's1/results.json')]),
        ('consequences', ['-m', 'artifacts.experiments.20261109_p1_hud_change.consequences', rel, str(r / 'consequences')]),
        ('interface', ['-m', 'artifacts.experiments.20261030_actor_interface.interface', str(r / 'interface.json'),
                       '--policy=random', f'multistep_4steps={rel}']),
        ('recursion_4', ['-m', f'{CT}.recursion_steps', '--ckpt', rel, '--out', str(r / 'recursion_steps4'), '--steps', '4']),
        ('natural_pair', ['-m', 'artifacts.experiments.20261109_p1_hud_change.natural_pair', rel, str(r / 'natural_pair.json')]),
        ('belief_probe', ['-m', f'{F4F}.belief_probe', rel, str(r / 'belief_probe.json')]),
        ('belief_path_probe', ['-m', f'{F4F}.belief_path_probe', rel, str(r / 'belief_path_probe.json')]),
    ]


def main():
    assert os.environ.get('MAMBA_DETERMINISTIC') == '1'
    logs = HERE / 'readings/logs'
    logs.mkdir(parents=True, exist_ok=True)
    status = HERE / 'readings/STATUS.json'
    done = json.loads(status.read_text()) if status.exists() else {}
    sha = TR.file_sha256(Path(sys.argv[1]).resolve())
    if done.setdefault('checkpoint_sha256', sha) != sha:
        raise RuntimeError('readout status belongs to another checkpoint')
    for name, args in steps(sys.argv[1]):
        if done.get(name) == 'complete':
            continue
        done[name] = 'running'
        status.write_text(json.dumps(done, indent=1))
        env = dict(os.environ, OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', CUDA_VISIBLE_DEVICES=os.environ.get('CUDA_VISIBLE_DEVICES', '0'))
        with open(logs / f'{name}.log', 'a') as log:
            subprocess.run([sys.executable, '-u', *args], cwd=TR.ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
        done[name] = 'complete'
        status.write_text(json.dumps(done, indent=1))


if __name__ == '__main__':
    main()
