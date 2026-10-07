"""Data-only audit of the proposed E19 mask/layout; not a treatment-effect result."""
import hashlib
import json
from pathlib import Path

import torch

torch.set_num_threads(2)
HERE=Path(__file__).parent
p=Path('artifacts/eda/spatial_pool_v1/pool.pt')
lab=Path('artifacts/eda/hpctx_labels_v1.pt')
d=torch.load(p,weights_only=False,mmap=True)
z=torch.load(lab,weights_only=False,mmap=True)['beside']
assert z.shape==d['dh'].shape and z.dtype==torch.bool
main=torch.where(~d['terminal'])[0]
held=main[torch.randperm(len(main),generator=torch.Generator().manual_seed(1))[:2048]]
rows=torch.cat([main[~torch.isin(main,held)],torch.where(d['terminal'])[0]])
dh, mask, alive=d['dh'][rows],z[rows],d['alive'][rows]
term=d['terminal'][rows]
death=alive[:,:-1]&~alive[:,1:]
valid=alive[:,:-1]
alive_successor=valid&alive[:,1:]

def table(sel):
    n=int(sel.sum())
    classes={name:int((sel&m).sum()) for name,m in
             [('drop_le2',dh<=-2),('starvation_minus1',dh==-1),('unchanged',dh==0),('recovery_plus1',dh>=1)]}
    return {'n':n,'health':classes,'drop_le2_rate':classes['drop_le2']/n if n else None,
            'zombie_mask':int((sel&mask).sum())}

out={'scope':'Exact TRAIN rows used by six-frame recipe. Counts describe selection, not causal effect of retraining.',
     'train_windows':len(rows),'heldout_windows':len(held),'terminal_windows':int(term.sum()),
     'all_alive_input':table(valid),'ordinary_alive_successor':table(alive_successor),
     'ordinary_masked':table(alive_successor&mask),'ordinary_unmasked':table(alive_successor&~mask),
     'terminal_health_drop_rows':[int(((dh<=-2)&term[:,None])[:,r].sum()) for r in range(5)],
     'all_health_drop_rows':[int((dh<=-2)[:,r].sum()) for r in range(5)],
     'actual_death_rows':[int(death[:,r].sum()) for r in range(5)],
     'terminal_already_dead_input_transitions':int((term[:,None]&~valid).sum()),
     'mask_rows':[int(mask[:,r].sum()) for r in range(5)],
     'death_transitions':int(death.sum()),
     'ordinary_drop_le2_mask_coverage':float(mask[alive_successor&(dh<=-2)].float().mean()),
     'ordinary_drop_le2_health_token_fraction':float((alive_successor&(dh<=-2)).sum())/(dh.numel()*81),
     'lambda1_masked_health_token_multiplier':1+dh.numel()*81/int(mask.sum()),
     'unweighted_drop_le2_rate':float((dh<=-2).float().mean()),
     'weighted_drop_le2_rate_lambda1':float(((dh<=-2).float()*(1+mask.float()*dh.numel()*81/mask.sum())).sum()
                                         /(1+mask.float()*dh.numel()*81/mask.sum()).sum()),
     'note':'Mask normalization weights selected health coordinates; scalar multiplier is batch-dependent in training. '
            'Global multiplier/rate describe an exact whole-pool loss, not every minibatch or the conditional optimum.'}
assert sum(out['actual_death_rows'])==int(death.sum())
# Exact expectation of the notebook's proposed crop/right-pad transform, enumerating all five offsets.
# Terminal death input row r retains original transitions [4-r,...,4], then masks padded targets.
retained=torch.ones_like(dh,dtype=torch.float64)
retained[term]=torch.arange(1,6,dtype=torch.float64)/5
main_death=death[~term].sum(0).double()
term_death=torch.zeros(5,dtype=torch.float64)
term_drop=torch.zeros(5,dtype=torch.float64)
for r in range(5):
    start=4-r
    term_death[:r+1]+=death[term,start:].sum(0).double()/5
    term_drop[:r+1]+=(dh[term,start:]<=-2).sum(0).double()/5
out['proposed_dealign_expectation']={
    'valid_targets':float(retained.sum()),
    'valid_target_change_fraction':float(retained.sum()/dh.numel()-1),
    'death_rows':(main_death+term_death).tolist(),
    'death_rate_before':float(death.sum()/valid.sum()),
    'death_rate_after':float((death.double()*retained).sum()/retained.sum()),
    'ordinary_damage_targets_before':int((alive_successor&(dh<=-2)).sum()),
    'ordinary_damage_targets_after':float(((alive_successor&(dh<=-2)).double()*retained).sum()),
    'ordinary_damage_target_change_fraction':float(((alive_successor&(dh<=-2)).double()*retained).sum()/(alive_successor&(dh<=-2)).sum()-1),
    'drop_le2_rows':((dh[~term]<=-2).sum(0).double()+term_drop).tolist(),
    'note':'This candidate changes target counts/context lengths as well as terminal position. '
           'Any de-alignment-only claim must account for these changes before launch; not a measured retrain effect.'}
tmp=HERE/'health_design.tmp'
tmp.write_text(json.dumps(out,indent=2)+'\n')
tmp.replace(HERE/'health_design.json')
print(json.dumps(out,indent=2))
