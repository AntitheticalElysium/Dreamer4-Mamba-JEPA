"""Resumable E20 coordinator; logs/status only in EDA, never NOTEBOOK.

Scientific validity failures hold the affected arm. A reader failure holds C but
does not discard the independent A/B replicate. Inputs and numerical sources are
sealed before the queue; every subprocess has its own atomic scientific resume.
"""
import fcntl
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[3]
L=ROOT/'artifacts/experiments/20260927_levers'
OUT=ROOT/'artifacts/eda/levers_e20_v1'
LOG=ROOT/'artifacts/eda/levers_logs'
PY=ROOT/'.venv/bin/python'
W=ROOT/'artifacts/eda/levers_tworlds_v1'


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb')as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def write(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    with tmp.open('w')as f:
        json.dump(data,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
    tmp.replace(path)


def main():
    os.chdir(ROOT)
    os.environ.update(TRITON_F32_DEFAULT='ieee',PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True',JAX_PLATFORMS='cpu')
    LOG.mkdir(parents=True,exist_ok=True);OUT.mkdir(parents=True,exist_ok=True)
    with (OUT/'coordinator.lock').open('a')as writer:
        fcntl.flock(writer,fcntl.LOCK_EX|fcntl.LOCK_NB)
        files=set(ROOT.joinpath('d4mj').rglob('*.py'))
        files.update(L/n for n in ('e20_driver.py','e20_data.py','e20_labels.py','e20_prepare.py','e20_train.py',
                                   'e20_verify.py','E20.md','e19.py','e19_eval.py','check_damage.py','check_recall.py',
                                   'tworld.py','teval.py','scroll.py','h16_resume.py'))
        files.add(ROOT/'artifacts/experiments/20260921_readout_ladder/spatial.py')
        files.add(ROOT/'artifacts/experiments/20261005_recovery/teval_export.py')
        pins={str(p):sha(p)for p in sorted(files)}
        contract=OUT/'driver_pins.json'
        if contract.exists():
            if json.loads(contract.read_text())!=pins:raise RuntimeError('E20 driver source contract changed')
        else:write(contract,pins)
        def check():
            if any(not Path(p).exists()or sha(p)!=digest for p,digest in pins.items()):
                raise RuntimeError('E20 numerical source changed during queue')
        def stage(name,script,*args,gpu=False):
            check()
            logpath=LOG/(name+'.log')
            lease=None
            if gpu:
                lease=(LOG/'gpu.lock').open('a');fcntl.flock(lease,fcntl.LOCK_EX)
                while True:
                    free=int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).splitlines()[0])
                    if free>=3656:break
                    write(OUT/'live.json',{'stage':name,'state':'waiting_gpu','free_mib':free,'unix':time.time()})
                    time.sleep(20)
            command=[str(PY),'-B',str(script),*map(str,args)]
            with logpath.open('a')as log:
                log.write(json.dumps({'event':'driver_start','command':command,'unix':time.time()})+'\n');log.flush()
                process=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,cwd=ROOT)
                started=time.time()
                while process.poll()is None:
                    write(OUT/'live.json',{'stage':name,'state':'running','pid':process.pid,
                        'seconds':time.time()-started,'unix':time.time(),'log':str(logpath)})
                    time.sleep(10)
                if lease is not None:lease.close()
                write(OUT/'live.json',{'stage':name,'state':'finished'if process.returncode==0 else'failed',
                                     'returncode':process.returncode,'unix':time.time(),'log':str(logpath)})
                if process.returncode:raise RuntimeError(f'{name} failed: {process.returncode}; see {logpath}')
        # User-directed scheduling changes do not invalidate already measured
        # numerical admission. Verify its bound inputs instead of rerunning a
        # timing-bearing proof and trying to overwrite its immutable report.
        proof_path=OUT/'mechanics.json'
        reference=OUT/'states/e20_A_s7_fmamba_fromM16/contract.json'
        if proof_path.exists() and reference.exists():
            proof=json.loads(proof_path.read_text());bound=json.loads(reference.read_text())
            if not proof['passed']:raise RuntimeError('Prior mechanics proof failed')
            for p,h in proof['sources'].items():
                if sha(p)!=h:raise RuntimeError('Measured mechanics source changed')
            for key,p in (('mechanics_sha256',proof_path),('pool_manifest_sha256',OUT/'pool.json'),
                          ('pool_labels_sha256',OUT/'pool/labels.pt'),('reader_sha256',OUT/'health_reader.pt')):
                if sha(p)!=bound[key]:raise RuntimeError(f'Admission input changed: {p}')
            pool=json.loads((OUT/'pool.json').read_text())
            for p,h in pool['contract']['inputs'].items():
                if sha(p)!=h:raise RuntimeError(f'Prepared pool input changed: {p}')
            with (LOG/'e20_mechanics.log').open('a')as log:
                log.write(json.dumps({'event':'verified_admission_reuse','mechanics_sha256':sha(proof_path),
                                      'source_difference':0,'input_difference':0,'unix':time.time()})+'\n')
        else:
            stage('e20_source_labels',L/'e20_data.py')
            stage('e20_prepare',L/'e20_prepare.py',gpu=True)
            stage('e20_reader',L/'e20_verify.py','reader')
            stage('e20_mechanics',L/'e20_verify.py','mechanics',gpu=True)
        cvalid=json.loads((OUT/'health_reader.json').read_text())['passed']
        arms=['A','B','C']if cvalid else['A','B']
        if not cvalid:write(OUT/'C_held.json',{'reason':'factual loss reader fails declared validity',
                                              'reader':json.loads((OUT/'health_reader.json').read_text())})
        review=OUT/'schedule_review.json'
        if not review.exists():write(review,{'approved_seeds':[7],'held_seeds':[8],
                                             'reason':'User requires seed7 review before replication'})
        requested=json.loads(review.read_text())['approved_seeds']
        if requested not in ([7],[7,8]):raise RuntimeError('Unexpected authorized seed schedule')
        completed=[]
        for seed in requested:
            for arm in arms:stage(f'e20_{arm}_s{seed}_train',L/'e20_train.py','--arm',arm,'--seed',seed,gpu=True)
            worlds=[W/f'e20_{arm}_s{seed}_fmamba_fromM16.pt'for arm in arms]
            stage(f'e20_s{seed}_health_strict',L/'e19_eval.py',*worlds,gpu=True)
            for window in (4,5,15):
                stage(f'e20_s{seed}_health_w{window}',L/'check_damage.py',*worlds,'--window',window,gpu=True)
            for window in (5,15):
                stage(f'e20_s{seed}_world_w{window}',ROOT/'artifacts/experiments/20261005_recovery/teval_export.py',
                      *worlds,'--window',window,gpu=True)
            stage(f'e20_s{seed}_recall',L/'check_recall.py','--futures',*worlds,gpu=True)
            completed.append(seed)
        write(OUT/'live.json',{'stage':'complete','state':'finished','arms':arms,
                              'completed_seeds':completed,'held_seeds':[s for s in (7,8)if s not in completed],
                              'review_required':8 not in completed,'unix':time.time()})


if __name__=='__main__':main()
