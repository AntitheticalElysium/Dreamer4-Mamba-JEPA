"""One-shot frozen audit and CPU report; no retry loop or downstream world job."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]


def sha(path):
    with open(path,'rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def write(path,x):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(x,indent=2));tmp.replace(path)


def main():
    manifest=json.loads((HERE/'SOURCE_PINS.json').read_text())
    def validate():
        for file,expected in manifest.items():assert sha(ROOT/file)==expected,file
    validate()
    status=HERE/'RUN_STATUS.json'
    assert not status.exists(),'One-shot worker already attempted; inspect before manual continuation'
    state=dict(phase='fitting',started_unix=time.time(),source_manifest_sha256=sha(HERE/'SOURCE_PINS.json'))
    write(status,state)
    try:
        with open(HERE/'run.log','a') as log:
            subprocess.run([sys.executable,'-u','-m','artifacts.experiments.20261103_fusion_retention.audit','run'],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
        validate();state.update(phase='reporting',fits_completed_unix=time.time());write(status,state)
        env={**os.environ,'CUDA_VISIBLE_DEVICES':''}
        with open(HERE/'report.log','a') as log:
            subprocess.run([sys.executable,'-u','-m','artifacts.experiments.20261103_fusion_retention.report'],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
        validate();state.update(phase='complete',completed_unix=time.time());write(status,state)
    except BaseException as e:
        state.update(phase='failed',failed_unix=time.time(),error=repr(e));write(status,state);traceback.print_exc();raise


if __name__=='__main__':main()
