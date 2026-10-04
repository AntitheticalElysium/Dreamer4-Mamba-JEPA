"""Post-hoc, exploratory 54k action-choice check of the factual frozen-H2 refit.

The 54k block was already inspected for another model. No fit or selection uses it.
This tests whether the improved factual terminal discrimination transfers to the
all-17-action decision target; it is not a sealed result or a canonical gate.
"""
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).parent
LADDER = ROOT / 'artifacts/experiments/20260921_readout_ladder'
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(LADDER))
from boundary import judge_store
from frozen_ladder import strata
from frozen_refit import Probe, CHECKPOINT, CACHE
from d4mj.data import _sha256
from d4mj.experiments import _load_bridge_parent
from d4mj.train import autocast_context

STORE = ROOT / 'artifacts/eda/observe_fresh_v5'
SEEDS = (7, 11, 19)
N_ACTIONS = 17
DOWN, SLEEP = 4, 6


def cluster_ci(values, seed_ids, draws=2000):
    # Equal resampling probability per episode seed, root-weighted statistic.
    unique, inv = np.unique(seed_ids, return_inverse=True)
    sums = np.bincount(inv, weights=values, minlength=len(unique))
    counts = np.bincount(inv, minlength=len(unique))
    take = np.random.default_rng(20261006).integers(0, len(unique), (draws, len(unique)))
    means = sums[take].sum(1) / counts[take].sum(1)
    return np.quantile(means, [.025, .975]).tolist()


@torch.no_grad()
def main():
    t0 = time.time()
    assert _sha256(CHECKPOINT) == 'ffb852c1f650a106dd94943c0458615cb8bc42be0181c2c465663db82a9eb20e'
    assert _sha256(CACHE / 'manifest.json') == '80d3d60521424ddb61318e6ebda9020e304cbcd3a96a7f2d07b1055bb1c30618'
    bundle, heads, _ = _load_bridge_parent(CHECKPOINT)
    bundle.encoder.freeze()
    bundle.world.eval()
    heads.eval()
    device = bundle.device
    rows, manifest, files = judge_store(STORE)
    n = len(rows['seed'])
    print(json.dumps({'stage':'loaded', 'roots':n, 'files':files, 'seconds':round(time.time()-t0,1)}),flush=True)
    probes=[]
    for seed in SEEDS:
        payload=torch.load(HERE / f'head_generated_{seed}.pt',map_location='cpu',weights_only=False)
        assert payload['checkpoint_sha256'] == _sha256(CHECKPOINT)
        p=Probe(len(payload['mean'])).to(device).eval()
        p.load_state_dict(payload['state'])
        probes.append((p,payload['mean'].to(device),payload['std'].to(device)))
    scores={'old':[], **{f'refit_{s}':[] for s in SEEDS}}
    for i in range(0,n,16):
        f=rows['frames'][i:i+16,-4:].to(device)
        past=rows['actions'][i:i+16,-3:].argmax(-1).to(device)
        assert bool((past < N_ACTIONS).all())
        with autocast_context(bundle.config):
            z=bundle.encoder(f)
            state=bundle.world.teacher(z,past).state
            acts=torch.arange(N_ACTIONS,device=device).repeat(len(f))[:,None]
            _,g=bundle.advance(bundle.repeat_state(state,N_ACTIONS),acts)
            gf=g[:,-1,0].float()
            old=1-torch.sigmoid(heads(g[:,-1:])['continuation'][:,0,0].float())
            scores['old'].append(old.reshape(len(f),N_ACTIONS).cpu())
            for seed,(p,mean,std) in zip(SEEDS,probes):
                risk=torch.sigmoid(p((gf-mean)/std))
                scores[f'refit_{seed}'].append(risk.reshape(len(f),N_ACTIONS).cpu())
        if i and i%1024 < 16:
            print(json.dumps({'stage':'encode','roots_done':i,'seconds':round(time.time()-t0,1)}),flush=True)
    scores={k:torch.cat(v) for k,v in scores.items()}
    truth=rows['p_death1']
    opp=truth.amax(1)>truth.amin(1)
    zombies=strata(rows['visible'])['zombie_adjacent']
    seed_ids=rows['seed'].numpy()
    safety={}
    choices={}
    for name,risk in scores.items():
        chosen=risk.argmin(1)
        safe=1-truth.gather(1,chosen[:,None]).squeeze(1)
        safety[name]=safe.numpy()
        choices[name]=chosen.numpy()
    prior=(1-truth[:,DOWN]).numpy()
    safety['down']=prior
    masks={'overall':opp.numpy(),'zombie':(opp&zombies).numpy()}
    report={'status':'exploratory_preinspected_54k', 'checkpoint_sha256':_sha256(CHECKPOINT),
            'judge_manifest_sha256':manifest,'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'roots':n,'opportunity_roots':int(opp.sum()),'rows':{}}
    for stratum,mask in masks.items():
        report['rows'][stratum]={'count':int(mask.sum()),'scores':{}}
        for name,safe in safety.items():
            selected=safe[mask]
            line={'expected_safe':float(selected.mean()),'ci95':cluster_ci(selected,seed_ids[mask])}
            if name in choices:
                pick=choices[name][mask]
                line['chosen_down']=int((pick==DOWN).sum())
                line['chosen_sleep']=int((pick==SLEEP).sum())
            report['rows'][stratum]['scores'][name]=line
        for name in scores:
            diff=safety[name][mask]-prior[mask]
            report['rows'][stratum]['scores'][name]['vs_down']={
                'difference':float(diff.mean()),'ci95':cluster_ci(diff,seed_ids[mask])}
    report['seconds']=round(time.time()-t0,1)
    (HERE/'fork_check.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'stage':'done','seconds':report['seconds'],'overall':{k:round(v['expected_safe'],4) for k,v in report['rows']['overall']['scores'].items()},'zombie':{k:round(v['expected_safe'],4) for k,v in report['rows']['zombie']['scores'].items()}}),flush=True)


if __name__=='__main__':
    main()
