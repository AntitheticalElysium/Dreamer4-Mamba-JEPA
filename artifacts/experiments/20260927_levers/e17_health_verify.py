"""Strict preflight for resuming the E17 frozen-health wrappers."""
import json
from pathlib import Path
import h16_resume as R


def main():
    count=0
    for seed in (7,8):
        p=next(Path('artifacts/eda/frozen_eval_resume_v1').glob(f'corrt_rawlong_teacher_s{seed}_fmamba_L16b40_from36000__damage_w15__*'))
        c=json.loads((p/'contract.json').read_text())
        for kind in ('sources','checkpoints'):
            for f,h in c[kind].items():assert R.file_hash(f)==h,f;count+=1
    p=Path('artifacts/experiments/20260927_levers/evals/resume/e17_health_clock_b16/contract.json')
    if p.exists():
        c=json.loads(p.read_text())
        for kind in ('sources','inputs','checkpoints'):
            for f,h in c[kind].items():assert R.file_hash(f)==h,f;count+=1
    print(json.dumps({'health_preflight_bound_file_hashes':count,'all_verified':True}),flush=True)


if __name__=='__main__':main()
