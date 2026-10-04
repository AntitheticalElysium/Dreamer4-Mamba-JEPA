"""Exact resume and legacy-recipe equivalence check before 18k fmamba+corrg launch."""
import json
import os
import subprocess
import sys
from pathlib import Path

import torch

ROOT=Path('/home/antithetical/EPITA/PERSO/DynamicHorizons-Mamba-JEPA')
HERE=Path(__file__).resolve().parent
OUT=ROOT/'artifacts/eda/levers_mamba_integration_smoke_v1'
TRAIN=HERE/'short_train.py'
NAME='int_corrg_raw_suffix_s7_fmamba_u18000.resume.pt'

def run(out,until):
    subprocess.run([sys.executable,str(TRAIN),'--pool','raw','--out',str(out),'--until',str(until)],
                   cwd=ROOT,check=True,env=os.environ|{'TRITON_F32_DEFAULT':'ieee'})

def delta(a,b):
    assert a.keys()==b.keys()
    return max(float((a[k].float()-b[k].float()).abs().max()) for k in a)

def main():
    split=OUT/'split';direct=OUT/'direct'
    assert not split.exists() and not direct.exists(), 'smoke directory must be new'
    run(split,3);run(split,6);run(direct,6)
    A=torch.load(split/NAME,map_location='cpu',weights_only=False)
    B=torch.load(direct/NAME,map_location='cpu',weights_only=False)
    resume_diff=delta(A['world'],B['world'])
    assert resume_diff==0.0, f'resume drift: {resume_diff}'
    assert torch.equal(A['order_rng'],B['order_rng'])
    assert A['update']==B['update']==6
    sys.path[:0]=[str(ROOT),str(ROOT/'artifacts/experiments/20260927_levers'),
                  str(ROOT/'artifacts/experiments/20260921_readout_ladder')]
    import tworld as T
    old,_,_=T.train('corrg','raw','suffix',7,6,torch.device('cuda'),lambda **kw: None,backbone='fmamba')
    old_diff=delta({k:v.cpu() for k,v in old.state_dict().items()},B['world'])
    report={'resume_max_abs_parameter_diff':resume_diff,'legacy_max_abs_parameter_diff':old_diff,
            'resume_order_rng_equal':True,'updates':6,'pool':'raw','backbone':'fmamba','head':'corrg'}
    (HERE/'verify_short.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report),flush=True)
    assert old_diff<=1e-6,f'new trainer differs from legacy recipe: {old_diff}'

if __name__=='__main__':main()
