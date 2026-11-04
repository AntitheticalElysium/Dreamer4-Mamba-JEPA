"""Bounded one-shot approved head fits and own-history runtime measurement."""
import os
from datetime import datetime,timezone
import importlib
import json
from pathlib import Path
import subprocess
import sys
import time
import traceback

C=importlib.import_module('artifacts.experiments.20261103_chooser_development.common')
HERE=Path(__file__).resolve().parent


def now():return datetime.now(timezone.utc).isoformat()


def available():
    r=subprocess.run(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory',
                      '--format=csv,noheader,nounits'],capture_output=True,text=True,check=True)
    for row in r.stdout.splitlines():
        if not row.strip():continue
        _,name,mem=[s.strip() for s in row.split(',',2)]
        if name.rsplit('/',1)[-1]=='swaync' and float(mem)<=32:continue
        raise RuntimeError('GPU occupied; no process touched: '+row)


def main():
    status=HERE/'STATUS.json'
    if status.exists():raise RuntimeError('One-shot campaign already attempted; preserve results and review')
    pins=json.loads((C.HERE/'SOURCE_PINS.json').read_text())['files']
    own=['PLAN.md','run.py','worker.py']
    # bench.py/report.py are added and pinned before their phases, while fits execute.
    pins.update({str((HERE/p).relative_to(C.ROOT)):C.sha(HERE/p) for p in own})
    C.write(HERE/'SOURCE_PINS.json',dict(files=pins))
    env=dict(os.environ,JAX_PLATFORMS='cpu',MAMBA_DETERMINISTIC='1',
             OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
    state=dict(phase='starting',started_at=now(),events=[]);C.write(status,state)
    deadline=time.monotonic()+2700
    phases=[('smoke','run',['smoke']),('1C_A','run',['fit','1C','611601']),
            ('1B_A','run',['fit','1B','611601']),('cost','bench',[]),
            ('1C_B','run',['fit','1C','611602']),('1B_B','run',['fit','1B','611602']),
            ('8C_A','run',['fit','8C','611601']),('8B_A','run',['fit','8B','611601']),
            ('8C_B','run',['fit','8C','611602']),('8B_B','run',['fit','8B','611602']),
            ('report','report',[])]
    try:
        for name,mod,args in phases:
            for rel,sha in pins.items():
                if C.sha(C.ROOT/rel)!=sha:raise RuntimeError('Pinned source/input changed: '+rel)
            available()
            script=HERE/(mod+'.py')
            if not script.exists():raise RuntimeError('Phase script not ready; preserved completed work: '+str(script))
            pins[str(script.relative_to(C.ROOT))]=C.sha(script)
            C.write(HERE/'SOURCE_PINS.json',dict(files=pins))
            state.update(phase=name,phase_started_at=now());C.write(status,state)
            with open(HERE/(name+'.log'),'a') as log:
                subprocess.run([sys.executable,'-u','-m','artifacts.experiments.20261103_chooser_history.'+mod,*args],
                               cwd=C.ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,check=True,
                               timeout=min(420 if name=='cost' else 900,max(1,deadline-time.monotonic())))
            state['events'].append(dict(phase=name,completed_at=now()))
            C.write(status,state)
        state.update(phase='complete',completed_at=now());C.write(status,state)
    except BaseException as e:
        state.update(failed_phase=state['phase'],phase='failed',error=repr(e),failed_at=now());C.write(status,state)
        traceback.print_exc();raise


if __name__=='__main__':main()
