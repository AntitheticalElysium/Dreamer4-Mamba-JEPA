"""CPU-only exact TRAIN target/dose accounting, and frozen router paired contrasts.

The error allocation here is the measured error of COPYING true health token63,
not the trained model's loss or its parameter gradients. Exposure and coefficients
come from the actual 6000-update ledger. No inferred motion labels are called truth.
Atomic source/input-bound stages resume independently; output stays in this campaign.
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import e19 as E
import e19_eval as D
R = E.R


def accounting():
    poolpath = E.T.POOLS['raw'] / 'pool.pt'
    root = E.T.OUT / 'state/e19_C_s7_fmamba_from36000'
    spec = json.loads((root / 'contract.json').read_text())
    training = R.Store(root, spec).load('train')
    assert training['update'] == 6000
    inputs = {'pool': R.file_hash(poolpath), 'mask': R.file_hash(E.MASK),
              'training': R.file_hash(root / json.loads((root / 'train.json').read_text())['file']),
              'source': R.file_hash(__file__), 'scope': __doc__}
    store = R.Store(HERE/'evals/resume/e19_actual_dose', inputs)
    with store.lock():
        result = store.load('result')
        if result is not None:
            return result
        pool = torch.load(poolpath, mmap=True, map_location='cpu', weights_only=False)
        mask = torch.load(E.MASK, weights_only=False)['beside'].bool()
        rows = E.train_rows(pool)
        dh, alive = pool['dh'], pool['alive'][:,1:]
        classes = {'death': ~alive, 'ordinary_ge2': alive & (dh <= -2),
                   'starvation_net_minus1': alive & (dh == -1),
                   'unchanged': alive & (dh == 0), 'recovery': alive & (dh >= 1)}
        assert sum(v.int() for v in classes.values()).eq(1).all()
        selected = torch.zeros_like(mask, dtype=torch.float64)
        seen = torch.zeros_like(mask, dtype=torch.int64)
        ledger = training['ledger'].long()
        for update in ledger:
            seen.index_add_(0, update, torch.ones(len(update),5,dtype=torch.int64))
            m = mask[update]
            selected.index_add_(0, update, m.double()/m.sum().clamp_min(1))
        assert int(seen.sum()) == 6000*40*5
        assert abs(float(selected.sum())-6000) < 1e-8
        copy = torch.empty_like(dh, dtype=torch.float32)
        for i in range(0,len(mask),256):
            token = pool['tokens'][i:i+256,:,63].float()
            copy[i:i+256] = (token[:,1:]-token[:,:-1]).abs().mean(-1)
        weighted_copy = selected*copy
        result = {'contract':inputs, 'updates':6000, 'windows':len(rows),
                  'targets_seen':int(seen.sum()), 'selected_seen':int((seen*mask).sum()),
                  'mask_count':int(mask[rows].sum()), 'classes':{}}
        for name, use in classes.items():
            chosen = use & mask
            result['classes'][name] = {
                'pool_targets':int(use[rows].sum()), 'pool_selected':int(chosen[rows].sum()),
                'sampled_targets':int(seen[use].sum()), 'sampled_selected':int(seen[chosen].sum()),
                'extra_loss_coefficient_share':float(selected[use].sum()/selected.sum()),
                'extra_copy_loss_share':float(weighted_copy[use].sum()/weighted_copy.sum()),
                'copy_token63_L1_mean':float(copy[use & (seen>0)].mean()),
                'copy_token63_L1_median':float(copy[use & (seen>0)].median()),
            }
        factor = selected[mask & (seen>0)]/seen[mask & (seen>0)]*(40*5*81)
        result['extra_over_teacher_coordinate_coefficient_quantiles'] = factor.quantile(
            torch.tensor([0.,.1,.5,.9,1.],dtype=torch.float64)).tolist()
        R.atomic_json(HERE/'evals/e19_actual_dose.json', result)
        store.save('result',result,1)
        return result


def router():
    meta, fit, seeds = D.T.split()
    masks = {k:v[:,0] for k,v in D.DR.masks(meta).items()}
    # Exact shifts from the simulator's recorded tile classes, same as frozen trace.
    from e19_diagnose import true_shifts
    shift, ok = true_shifts(meta)
    valid = masks['valid'] & masks['k3']
    assert (ok | ~valid).all()
    selected = {'hit':valid & masks['drop2'],
                'fresh_hit':valid & masks['drop2'] & masks['adjacent'] & ~masks['win'] & ~masks['adjwin'],
                'scroll_hit':valid & masks['drop2'] & (shift!=0),
                'stationary_hit':valid & masks['drop2'] & (shift==0),
                'unchanged':valid & (masks['dh']==0)}
    result = {'scope':'post-hoc frozen composition interventions; not a deployment repair', 'arms':{}}
    for arm in ('B','C'):
        path = HERE/'evals'/f'e19_{arm}_s7_fmamba_from36000__e19_router_trace.pt'
        trace = torch.load(path, map_location='cpu', weights_only=False)
        assert torch.equal(trace['seed'],meta['seed'])
        deltas = trace['delta']; out = {}
        for label, use in selected.items():
            out[label] = {name:D.cluster_contrast(deltas['actual'] < -1.5, delta < -1.5,
                                                 meta['seed'],use)
                          for name,delta in deltas.items() if name != 'actual'}
        result['arms'][arm] = {'trace_sha256':R.file_hash(path), 'contrasts':out}
    R.atomic_json(HERE/'evals/e19_router_contrasts.json',result)
    return result


if __name__=='__main__':
    torch.set_num_threads(3)
    print(json.dumps({'dose':accounting(),'router':router()},indent=2),flush=True)
