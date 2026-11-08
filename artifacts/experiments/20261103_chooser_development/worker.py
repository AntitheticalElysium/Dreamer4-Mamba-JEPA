"""One-shot queue after the approved F1.6 evaluation; never controls another job."""
import os
import argparse
from datetime import datetime, timezone
import importlib
import json
import subprocess
import sys
import time
import traceback

C=importlib.import_module('artifacts.experiments.20261103_chooser_development.common')
PREDECESSOR=C.HERE.parent/'20261102_p1f16/STATUS.json'


def now():
    return datetime.now(timezone.utc).isoformat()


def predecessor():
    out=subprocess.run(['systemctl','--user','show','saccade-p1f16.service',
                        '--property=ActiveState','--value'],check=True,capture_output=True,text=True)
    phase=json.loads(PREDECESSOR.read_text())['phase']
    if phase=='failed':
        raise RuntimeError('Predecessor failed; manual review required, not bypassed')
    return out.stdout.strip() in ('inactive','failed') and phase=='complete'


def gpu_available():
    out=subprocess.run(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory',
                        '--format=csv,noheader,nounits'],check=True,capture_output=True,text=True)
    for line in out.stdout.splitlines():
        if not line.strip():continue
        pid,name,memory=[x.strip() for x in line.split(',',2)]
        # NVML reports this desktop notification renderer as a compute client.
        if name.rsplit('/',1)[-1]=='swaync' and float(memory)<=32:continue
        raise RuntimeError('GPU occupied; no process touched: '+line)


def verify():
    pin=json.loads((C.HERE/'SOURCE_PINS.json').read_text())
    for rel, sha in pin['files'].items():
        if C.sha(C.ROOT/rel)!=sha:
            raise RuntimeError('Pinned source/input changed: '+rel)


def main():
    p=argparse.ArgumentParser();p.add_argument('--dry-run',action='store_true');args=p.parse_args()
    verify()
    if args.dry_run:
        print(json.dumps(dict(phases=['smoke','e1','e2','e3','report'],
            predecessor=str(PREDECESSOR),max_worker_seconds=2700)));return
    status=C.HERE/'STATUS.json'
    if status.exists():raise RuntimeError('One-shot worker already attempted; manual review required')
    state=dict(phase='waiting_for_p1f16_evaluation',queued_at=now(),events=[])
    C.write(status,state)
    try:
        while not predecessor():
            time.sleep(30)
        verify();gpu_available();deadline=time.monotonic()+2700
        for phase in ('smoke','e1','e2','e3'):
            verify();gpu_available()
            state.update(phase=phase,phase_started_at=now());C.write(status,state)
            env=dict(os.environ,JAX_PLATFORMS='cpu',MAMBA_DETERMINISTIC='1',
                     OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
            with open(C.HERE/(phase+'.log'),'a') as log:
                subprocess.run([sys.executable,'-u','-m',
                    'artifacts.experiments.20261103_chooser_development.campaign',phase],
                    cwd=C.ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,
                    check=True,timeout=max(1,deadline-time.monotonic()))
            state['events'].append(dict(phase=phase,completed_at=now()))
        verify()
        with open(C.HERE/'report.log','a') as log:
            subprocess.run([sys.executable,'-u','-m',
                'artifacts.experiments.20261103_chooser_development.report'],cwd=C.ROOT,
                stdout=log,stderr=subprocess.STDOUT,check=True,timeout=max(1,deadline-time.monotonic()))
        state.update(phase='complete',completed_at=now());C.write(status,state)
    except BaseException as e:
        state.update(failed_phase=state['phase'],phase='failed',failed_at=now(),error=repr(e))
        C.write(status,state);traceback.print_exc();raise


if __name__=='__main__':main()
