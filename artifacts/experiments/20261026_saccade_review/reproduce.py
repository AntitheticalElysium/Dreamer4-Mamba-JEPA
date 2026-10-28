"""Reproduce the existing harness into this review directory, without changing it."""
import json
from pathlib import Path
import runpy
import subprocess
import sys
import time

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]
PYTHON = ROOT / '.venv/bin/python'
SOURCE = ROOT / 'artifacts/eda/s0'


def run(module, args):
    start = time.monotonic()
    with (OUT / f'{module}.log').open('w') as log:
        result = subprocess.run([str(PYTHON), '-m', f'saccade.eval.{module}', *map(str, args)],
                                cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
    record = {'module': module, 'seconds': time.monotonic() - start, 'exit_code': result.returncode}
    print(json.dumps(record), flush=True)
    if result.returncode:
        raise SystemExit(f'{module} failed; see {OUT / (module + ".log")}')
    return record


def cpu():
    dev, train = OUT / 's0_dev_labels.npz', OUT / 's0_train400_labels.npz'
    commands = [('labels', [dev, 'dev']), ('labels', [train, 'train', 400]),
                ('gates', [dev, OUT / 's0_gates.json']), ('revisit', [dev, OUT / 's0_revisit.json']),
                ('oracle', [dev, OUT / 's0_oracle.json']), ('calib', [train, dev, OUT / 's0_calib.json'])]
    records = []
    for module, args in commands:
        records.append(run(module, args))
        (OUT / 'cpu_reproduction.json').write_text(json.dumps(records, indent=2))


def gpu():
    import torch
    torch.cuda.reset_peak_memory_stats()
    sys.argv = ['saccade.eval.probe', str(SOURCE / 's0_train400_labels.npz'),
                str(SOURCE / 's0_dev_labels.npz'), str(OUT / 's0_probe.json')]
    start = time.monotonic()
    runpy.run_module('saccade.eval.probe', run_name='__main__')
    result = {'seconds': time.monotonic() - start, 'torch': torch.__version__,
              'cuda': torch.version.cuda, 'gpu': torch.cuda.get_device_name(),
              'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
              'peak_reserved_bytes': torch.cuda.max_memory_reserved()}
    (OUT / 'gpu_reproduction.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    {'cpu': cpu, 'gpu': gpu}[sys.argv[1]]()
