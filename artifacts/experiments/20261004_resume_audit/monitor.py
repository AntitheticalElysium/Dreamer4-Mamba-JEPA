"""Lightweight live status notes; no GPU computations, verdict changes, or automatic new experiments."""
import datetime
import json
import subprocess
import time
from pathlib import Path

OUT = Path(__file__).parent
LOG = Path('artifacts/eda/levers_logs')
UNITS = ['d4mj-oct04-resume', 'd4mj-oct04-e17', 'd4mj-oct04-e18', 'd4mj-oct04-e16-fixed']


def read_command(args):
    p = subprocess.run(args, capture_output=True, text=True, timeout=15)
    return {'exit_code': p.returncode, 'stdout': p.stdout.strip(), 'stderr': p.stderr.strip()}


def snapshot():
    out = {'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
           'note': 'Runtime/status only. Active unit Result is not a scientific verdict. A missing training log row does not mean zero compute.',
           'units': {u: read_command(['systemctl', '--user', 'show', u, '-p', 'ActiveState', '-p', 'Result', '-p', 'MemoryCurrent']) for u in UNITS},
           'python_processes': read_command(['ps', '-C', 'python', '-o', 'pid,etimes,pcpu,args']),
           'gpu': read_command(['nvidia-smi', '--query-gpu=memory.used,memory.free,utilization.gpu', '--format=csv,noheader']),
           'recent_events': (LOG / 'lanes.log').read_text().splitlines()[-15:]}
    out['report_present'] = {name: (OUT / name).exists() for name in
                             ('sighting_results.json', 'carry_results.json', 'delta_results.json', 'e16_fixed_result.json')}
    out['recent_logs'] = {}
    for name in ['oct04_h16traj_m6', 'tworld_c6_s7', 'tworld_c6_s8', 'e17_M16_s7', 'e17_M16_s8']:
        p = LOG / f'{name}.log'
        out['recent_logs'][name] = p.read_text().splitlines()[-3:] if p.exists() else None
    return out


for _ in range(2880):  # up to 24 hours; stop once all managed lanes stop
    state = snapshot()
    tmp = OUT / 'live_status.tmp'
    tmp.write_text(json.dumps(state, indent=2) + '\n')
    tmp.replace(OUT / 'live_status.json')
    with (OUT / 'status_events.jsonl').open('a') as f:
        f.write(json.dumps(state) + '\n')
    if not any('ActiveState=active' in v['stdout'] or 'ActiveState=activating' in v['stdout'] for v in state['units'].values()):
        break
    time.sleep(30)
