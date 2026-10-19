"""CPU exploratory temperature calibration of frozen E22M code priors.

Select temperature on validation episode clusters and score disjoint test
clusters. Previously inspected diagnosis panel; this is mechanism evidence.
Posterior targets are frozen. Temperature cannot change argmax accuracy.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['OMP_NUM_THREADS'] = '2'
os.environ['MKL_NUM_THREADS'] = '2'
import sys, json, time
from pathlib import Path
import torch
import torch.nn.functional as F
import numpy as np
torch.set_num_threads(2)
torch.set_num_interop_threads(1)
ROOT = Path('/home/antithetical/EPITA/PERSO/DynamicHorizons-Mamba-JEPA')
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT/'artifacts/experiments/20260927_levers'))
import dworld as DW
start = time.time()
rows = torch.load('/home/antithetical/EPITA/PERSO/DynamicHorizons-Mamba-JEPA/artifacts/eda/health_chain_v1/d4mj_prior_conditional_rows_20261009.pt', map_location='cpu', weights_only=False)
cache = torch.load(ROOT/'artifacts/eda/levers_futures_tokens_raw.pt', map_location='cpu', mmap=True, weights_only=False)
frames = torch.cat([cache['ctx'], cache['fut']], 1).float()
acts = torch.cat([cache['ctx_a'], cache['fut_a']], 1)
ys, seeds = rows['targets'], rows['seed']
meta = torch.load(ROOT/'artifacts/eda/diagnosis_rollouts_v1/meta.pt', map_location='cpu', mmap=True, weights_only=False)
# Score real future transitions through the first terminal outcome, excluding
# the observed context and all frames after death.
before_alive = torch.cat([torch.ones(len(seeds),1,dtype=torch.bool),~meta['future_dead'][:,0].cumsum(1).bool()[:,:-1]],1)
live_mask = torch.cat([torch.zeros(len(seeds),3,dtype=torch.bool),before_alive],1)
unique = seeds.unique()
unique = unique[torch.randperm(len(unique), generator=torch.Generator().manual_seed(0))]
train = unique[:int(.7*len(unique))]
validation = train[-max(1,len(train)//5):]
test = unique[len(train):]
va, te = torch.isin(seeds, validation), torch.isin(seeds, test)
temperatures = (.5,.75,1.,1.5,2.,3.,4.,6.,8.)
scores = {}
live_scores = {}
live_counts = live_mask.sum(1)
out = {'protocol':__doc__, 'roots':len(seeds), 'validation_seeds':validation.tolist(),
       'test_seeds':test.tolist(), 'temperature_grid':temperatures, 'priors':{}, 'cuda_available':torch.cuda.is_available()}
base = 'dworld_a_s7_from2000_revalways_mwauto_L16_di_b16_u2000'
for updates in (20000,60000):
    payload = torch.load(ROOT/f'artifacts/eda/levers_tworlds_v1/{base}_prior_u{updates}.pt', map_location='cpu', mmap=True, weights_only=False)
    prior = DW.Prior().eval()
    prior.load_state_dict(payload['prior'])
    root_ce = torch.empty(len(seeds),len(temperatures))
    live_sum = torch.empty_like(root_ce)
    with torch.no_grad():
        for i in range(0,len(seeds),8):
            logits = prior(frames[i:i+8], F.pad(acts[i:i+8],(0,1)), F.pad(ys[i:i+8],(0,0,0,1)))[0][:,:-1].float()
            targets = ys[i:i+8]
            for j,t in enumerate(temperatures):
                transition_ce = F.cross_entropy((logits/t).reshape(-1,DW.CODES), targets.flatten(), reduction='none').view(len(targets),19,4).mean(-1)
                root_ce[i:i+len(targets),j] = transition_ce.mean(1)
                live_sum[i:i+len(targets),j] = (transition_ce*live_mask[i:i+len(targets)]).sum(1)
            if i%256 == 0: print(json.dumps({'updates':updates,'roots':i+len(targets),'seconds':round(time.time()-start,1)}),flush=True)
    chosen = int(root_ce[va].mean(0).argmin())
    scores[updates] = root_ce
    live_scores[updates] = live_sum
    out['priors'][str(updates)] = {'chosen_temperature':temperatures[chosen], 'validation_ce_by_temperature':root_ce[va].mean(0).tolist(),
        'test_raw_ce':float(root_ce[te,2].mean()), 'test_calibrated_ce':float(root_ce[te,chosen].mean()),
        'test_accuracy':float(rows['correct'][updates][te].float().mean())}
rng = np.random.default_rng(20261009)
indices = rng.integers(0,len(test),(2000,len(test)))
def paired_interval(differences):
    sums = np.array([[float(differences[seeds==s].sum()),int((seeds==s).sum())] for s in test])
    boot = sums[indices].sum(1)
    return {'mean':float(sums[:,0].sum()/sums[:,1].sum()),'ci95':np.quantile(boot[:,0]/boot[:,1],[.025,.975]).tolist()}
selected = {u:int(scores[u][va].mean(0).argmin()) for u in scores}
out['contrasts'] = {'raw_60k_minus_20k_ce':paired_interval(scores[60000][:,2]-scores[20000][:,2]),
    'calibrated_60k_minus_20k_ce':paired_interval(scores[60000][:,selected[60000]]-scores[20000][:,selected[20000]]),
    '60k_calibration_minus_raw_ce':paired_interval(scores[60000][:,selected[60000]]-scores[60000][:,2])}
def live_interval(differences):
    sums = np.array([[float(differences[seeds==s].sum()),int(live_counts[seeds==s].sum())] for s in test])
    boot = sums[indices].sum(1)
    return {'mean':float(sums[:,0].sum()/sums[:,1].sum()),'ci95':np.quantile(boot[:,0]/boot[:,1],[.025,.975]).tolist()}
live_selected = {u:int(live_scores[u][va].sum(0).argmin()) for u in live_scores}
out['alive_future'] = {'definition':'Future transitions only, including first terminal target, excluding later frames; 4 codes per transition',
    'test_transitions':int(live_counts[te].sum()),'priors':{},'contrasts':{}}
for u in live_scores:
    j = live_selected[u]
    out['alive_future']['priors'][str(u)] = {'chosen_temperature':temperatures[j],
        'validation_ce_by_temperature':(live_scores[u][va].sum(0)/live_counts[va].sum()).tolist(),
        'test_raw_ce':float(live_scores[u][te,2].sum()/live_counts[te].sum()),
        'test_calibrated_ce':float(live_scores[u][te,j].sum()/live_counts[te].sum()),
        'test_accuracy':float(rows['correct'][u][te][live_mask[te]].float().mean())}
out['alive_future']['contrasts'] = {'raw_60k_minus_20k_ce':live_interval(live_scores[60000][:,2]-live_scores[20000][:,2]),
    'calibrated_60k_minus_20k_ce':live_interval(live_scores[60000][:,live_selected[60000]]-live_scores[20000][:,live_selected[20000]]),
    '60k_calibration_minus_raw_ce':live_interval(live_scores[60000][:,live_selected[60000]]-live_scores[60000][:,2])}
out['seconds'] = time.time()-start
Path('/home/antithetical/EPITA/PERSO/DynamicHorizons-Mamba-JEPA/artifacts/eda/health_chain_v1/d4mj_prior_calibration_20261009.json').write_text(json.dumps(out,indent=2))
torch.save({'seed':seeds,'temperatures':temperatures,'ce_by_temperature':scores,'live_ce_sums_by_temperature':live_scores,'live_transitions':live_counts},'/home/antithetical/EPITA/PERSO/DynamicHorizons-Mamba-JEPA/artifacts/eda/health_chain_v1/d4mj_prior_calibration_rows_20261009.pt')
print(json.dumps(out,indent=2),flush=True)
