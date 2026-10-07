"""Paired seed-cluster contrasts for completed E17 fixed-clock interventions."""
import json
from pathlib import Path

import numpy as np
import torch

import e17_clock_control as C
import h16_resume as R

HERE=Path(__file__).resolve().parent
out={'scope':__doc__,'contrasts':{},'inputs':{},'source_sha256':R.file_hash(__file__)}
for seed in (7,8):
    names=[f'corrt_rawlong_teacher_s{seed}{suffix}_L16b40_from36000'for suffix in ('','_fmamba')]
    paths=[HERE/'evals'/(n+'__e17_fixed_clock_rows.pt')for n in names]
    a,b=[torch.load(p,map_location='cpu',weights_only=False)for p in paths]
    assert torch.equal(a['roots'],b['roots'])and torch.equal(a['seed'],b['seed'])
    out['inputs'].update({str(p):R.file_hash(p)for p in paths})
    groups=[np.flatnonzero(a['seed'].numpy()==s)for s in a['seed'].unique().tolist()]
    rng=np.random.default_rng(20261005)
    res={}
    for g in ('same','moved'):
        av,bv=a['raw'][g],b['raw'][g]
        assert np.array_equal(av[:,[0,3,4]],bv[:,[0,3,4]])
        point=C.stats(bv)['history_gain']-C.stats(av)['history_gain'];diff=[]
        for _ in range(2000):
            idx=np.concatenate([groups[i]for i in rng.integers(len(groups),size=len(groups))])
            if av[idx].sum(0)[4]>av[idx].sum(0)[3]:
                diff.append(C.stats(bv[idx])['history_gain']-C.stats(av[idx])['history_gain'])
        res[g]={'mamba_minus_attention_history_gain':point,'interval95':np.quantile(diff,[.025,.975]).tolist(),
                'episode_seeds':len(groups),'cells':int(av[:,0].sum())}
    out['contrasts'][str(seed)]=res
R.atomic_json(HERE/'evals/e17_fixed_clock_contrasts.json',out)
print(json.dumps(out,indent=2))
