"""P1-F3 morning readout (PLAN.md): one subprocess per instrument, in order, on the selected checkpoint; each logs to
readings/logs/NAME.log; a failure stops the chain (the worker records it). Instruments are the consequence trace's
(../20261105_consequence_trace/), compared with F2's readings there; sampler reads at 3 and 16 steps."""
import os
os.environ.setdefault('JAX_PLATFORMS', 'cpu')
import json
from pathlib import Path
import subprocess
import sys
from saccade import train as TR

HERE = Path(__file__).resolve().parent
CT = 'artifacts.experiments.20261105_consequence_trace'


def steps(ckpt):
    ck = Path(ckpt).resolve(); rel = str(ck.relative_to(TR.ROOT)); r = HERE / 'readings'
    trace = r / 'trace'
    return [
        ('s1', ['-m', 'saccade.eval.s1', str(ck), str(ck.parent / 's1/results.json')]),
        ('trace', ['-m', f'{CT}.trace', str(trace), rel]),
        ('sampler_steps', ['-m', f'{CT}.sampler_steps', str(r / 'sampler_steps'), rel, str(trace), str(trace)]),
        ('interface_steps', ['-m', f'{CT}.interface_steps', str(r / 'interface_steps.json'), rel, 'f3']),
        ('recursion_3', ['-m', f'{CT}.recursion_steps', '--ckpt', rel, '--out', str(r / 'recursion_steps3'), '--steps', '3']),
        ('recursion_16', ['-m', f'{CT}.recursion_steps', '--ckpt', rel, '--out', str(r / 'recursion_steps16'), '--steps', '16']),
        ('glimpse', ['-m', f'{CT}.glimpse_check', str(r / 'glimpse.json'), rel]),
        ('readings', [str(TR.ROOT / 'artifacts/experiments/20261105_consequence_trace/readings.py'), str(trace / 'results.json')]),
        ('posthoc', [str(TR.ROOT / 'artifacts/experiments/20261105_consequence_trace/posthoc.py'), str(trace)]),
        ('allocation', [str(TR.ROOT / 'artifacts/experiments/20261105_consequence_trace/weighting_allocation.py'), str(trace)]),
    ]


def main():
    assert os.environ.get('MAMBA_DETERMINISTIC') == '1'
    logs = HERE / 'readings/logs'; logs.mkdir(parents=True, exist_ok=True)
    status = HERE / 'readings/STATUS.json'; done = json.loads(status.read_text()) if status.exists() else {}
    for name, args in steps(sys.argv[1]):
        if done.get(name) == 'complete':
            continue
        done[name] = 'running'; status.write_text(json.dumps(done, indent=1))
        env = dict(os.environ, OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', CUDA_VISIBLE_DEVICES=os.environ.get('CUDA_VISIBLE_DEVICES', '0'))
        if name in ('readings', 'posthoc', 'allocation'):
            env['CUDA_VISIBLE_DEVICES'] = ''
        with open(logs / f'{name}.log', 'a') as log:
            subprocess.run([sys.executable, '-u', *args], cwd=TR.ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
        done[name] = 'complete'; status.write_text(json.dumps(done, indent=1))


if __name__ == '__main__':
    main()
