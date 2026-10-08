"""Persistent E20 endpoint completion queue; waits for the scientific main queue."""
import fcntl
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


def main():
    import sys
    sys.path.insert(0,str(L))
    from e20_driver import sha,write
    os.chdir(ROOT)
    os.environ.update(TRITON_F32_DEFAULT='ieee',PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True',JAX_PLATFORMS='cpu')
    with (OUT/'post.lock').open('a')as writer:
        fcntl.flock(writer,fcntl.LOCK_EX|fcntl.LOCK_NB)
        files=set(ROOT.joinpath('d4mj').rglob('*.py'))
        files.update(L/n for n in ('e20_post.py','e20_endpoints.py','E20_ENDPOINTS.md','e20_driver.py',
                                   'e20_train.py','e20_data.py','e20_labels.py','e20_verify.py','e19.py','e19_eval.py',
                                   'teval.py','tworld.py','scroll.py','h16_resume.py','check_recall.py'))
        files.add(ROOT/'artifacts/experiments/20261005_recovery/teval_export.py')
        files.add(ROOT/'artifacts/experiments/20260921_readout_ladder/spatial.py')
        pins={str(p):sha(p)for p in sorted(files)}
        path=OUT/'post_pins.json'
        if path.exists():
            if json.loads(path.read_text())!=pins:raise RuntimeError('Post queue sources changed')
        else:write(path,pins)
        def check():
            if any(not Path(p).exists()or sha(p)!=h for p,h in pins.items()):raise RuntimeError('Endpoint source changed')
        while True:
            check()
            live=json.loads((OUT/'live.json').read_text())
            if live['stage']=='complete'and live['state']=='finished':break
            if live['state']=='failed':raise RuntimeError('Main queue failed; endpoints held')
            write(OUT/'post_live.json',{'stage':'waiting_main_endpoints','main_stage':live['stage'],'unix':time.time()})
            time.sleep(30)
        def stage(name,script,*args,gpu=False):
            check()
            lease=None
            if gpu:
                lease=(LOG/'gpu.lock').open('a');fcntl.flock(lease,fcntl.LOCK_EX)
                while int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).splitlines()[0])<3656:
                    time.sleep(20)
            command=[str(PY),'-B',str(script),*map(str,args)]
            with (LOG/(name+'.log')).open('a')as log:
                log.write(json.dumps({'event':'driver_start','command':command,'unix':time.time()})+'\n');log.flush()
                p=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,cwd=ROOT)
                while p.poll()is None:
                    write(OUT/'post_live.json',{'stage':name,'state':'running','pid':p.pid,'unix':time.time()})
                    time.sleep(10)
                if lease is not None:lease.close()
                if p.returncode:raise RuntimeError(f'{name} failed ({p.returncode}); evidence retained')
        for seed in live['completed_seeds']:
            parent=W/f'corrt_rawlong_teacher_s{seed}_fmamba_L16b40_from36000.pt'
            arms=[W/f'e20_{arm}_s{seed}_fmamba_fromM16.pt'for arm in live['arms']]
            paths=[parent,*arms]
            for window in (5,15):
                stage(f'e20_post_s{seed}_parent_w{window}',ROOT/'artifacts/experiments/20261005_recovery/teval_export.py',
                      parent,'--window',window,gpu=True)
                stage(f'e20_post_s{seed}_paired_w{window}',L/'e20_endpoints.py','compare',*paths,'--window',window)
            stage(f'e20_post_s{seed}_dual_w4',L/'e20_endpoints.py','four',*paths,gpu=True)
            stage(f'e20_post_s{seed}_parent_position',L/'e19_eval.py',parent,gpu=True)
            rows=[L/'evals'/f'{p.stem}__e19_health_per_root.pt'for p in paths]
            stage(f'e20_post_s{seed}_paired_position',L/'e19_eval.py','--compare',*rows)
        write(OUT/'post_live.json',{'stage':'seed7_review_ready'if live['review_required']else'complete',
                                   'state':'finished','completed_seeds':live['completed_seeds'],
                                   'held_seeds':live['held_seeds'],'review_required':live['review_required'],
                                   'unix':time.time()})


if __name__=='__main__':main()
