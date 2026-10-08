"""Completed short/long Mamba8 H16 comparison on identical DEV-B roots.

Original averaged-head decision rows; assert labels/root order/opportunity masks
and numerical sources match. Parent-to-long changes window, world training budget,
corpus pool and optimizer; this is not a causal context-only or memory-only effect.
"""
import json
from pathlib import Path
import torch
import h16_resume as R
import e17_h16_diagnose as D

HERE=Path(__file__).resolve().parent


def main():
    root=Path('artifacts/eda/deepeval_v1/h16-resume-v1');paths={
        'short':root/'corrt_raw_teacher_s8_fmamba_u36000__w5__det__861588baea475c7a',
        'long':root/'corrt_rawlong_teacher_s8_fmamba_L16b40_from36000__w15__det__cb1596225c7af346'}
    contracts={};results={}
    for k,p in paths.items():
        c=json.loads((p/'contract.json').read_text());contracts[k]=c
        for f,h in c['sources'].items():assert R.file_hash(f)==h,f
        results[k]=R.Store(p,c).load('result');assert results[k] is not None
    assert contracts['short']['sources']==contracts['long']['sources']
    for k in ('seed_devB','opportunity_devB','P_devB'):assert torch.equal(results['short'][k],results['long'][k]),k
    ids=results['short']['seed_devB'][results['short']['opportunity_devB']].numpy()
    contrasts={k:D.paired(results['long']['safe_rows'][k].numpy(),results['short']['safe_rows'][k].numpy(),ids) for k in ('trajectory','snapshot')}
    result={'scope':__doc__,'contracts':contracts,'sources':{f:R.file_hash(f) for f in (__file__,R.__file__,D.__file__)},
            'short':results['short']['res'],'long':results['long']['res'],'long_minus_short':contrasts}
    R.atomic_json(HERE/'evals/e17_h16_parent_compare.json',result)
    print(json.dumps({'short':result['short'],'long':result['long'],'long_minus_short':contrasts},indent=2),flush=True)


if __name__=='__main__':main()
