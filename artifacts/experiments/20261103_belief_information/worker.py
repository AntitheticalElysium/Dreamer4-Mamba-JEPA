"""One-shot bounded worker. Successful fits do not launch another experiment."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2]


def main():
    def validate():
        for p,h in json.loads((HERE/'SOURCE_PINS.json').read_text()).items():
            with (ROOT/p).open('rb') as f:assert hashlib.file_digest(f,'sha256').hexdigest()==h,p
    validate();state=HERE/'RUN_STATUS.json';assert not state.exists(),'One-shot job already attempted'
    status=dict(phase='predictive_readers',started_unix=time.time())
    def save():state.write_text(json.dumps(status,indent=2)+'\n')
    save()
    try:
        for stage,module in (('predictive_readers','audit'),('genuine_look_native_comparison','paired'),('reporting','report')):
            validate();status['phase']=stage;save()
            env=os.environ.copy()
            if module=='report':env['CUDA_VISIBLE_DEVICES']=''
            cmd=[sys.executable,'-u','-m',f'artifacts.experiments.20261103_belief_information.{module}']
            if module=='audit':cmd+=['run']
            with (HERE/f'{module}.log').open('w') as log:
                subprocess.run(cmd,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
        validate();status.update(phase='complete',completed_unix=time.time());save()
    except BaseException as e:
        status.update(phase='failed',error=repr(e),failed_unix=time.time());save();traceback.print_exc();raise


if __name__=='__main__':main()
