"""Measurement controls, alignment and immutable-input checks."""
import importlib
import json
from pathlib import Path
import numpy as np
import torch

A=importlib.import_module('artifacts.experiments.20261103_gaze_belief.analyze')
C=A.C


def main():
    result={}
    z=torch.load(C.OLD/'dev_600_coarse.pt',map_location='cpu',weights_only=False)
    assert np.array_equal(z['episode_keys'],np.stack([A.L['ep_shard'][:600],A.L['ep_slot'][:600]],-1))
    assert np.array_equal(z['offsets'],A.OFF[:601])
    result['label_cache_episode_alignment']=True
    for cohort in ('saved','confirmation'):
        bank=A.models(cohort); ref=bank['native']; m=A.metadata(ref)
        target=m['masks']['any_entity'][:,C.WIN].any(-1)
        ix=(ref['cls']=='random') & target.any(-1)
        for name in ('native','E1_611601','E2_carry_611602'):
            p=C.policies(bank[name]['C'],bank[name]['S'])['existing']
            cover=(p*target).sum(-1)
            uniform=target.mean(-1)
            shuffled=np.mean([(np.roll(p,k,axis=-1)*target).sum(-1) for k in range(49)],0)
            assert np.max(abs(shuffled-uniform))<1e-12
            assert cover[ix].mean()>shuffled[ix].mean()+.15
        result[cohort+'_spatial_scramble_power']=True
        assert np.allclose(sum(m['categories'].values()),1)
        assert (m['masks']['entering_map'][m['action']==0].sum()==0)
        # Offset t by one must materially change the physical state used by the instrument.
        before=np.stack([A.E[e]['grid'][t-1] for e,t in zip(ref['ep'],ref['t'])])
        after=np.stack([A.E[e]['grid'][t] for e,t in zip(ref['ep'],ref['t'])])
        result[cohort+'_frames_with_changed_entity_grid']=int((before!=after).any((1,2)).sum())
    r=A.estimates(np.zeros(12),np.ones(12),np.repeat(np.arange(4),3))[0]
    assert r['mean']==r['low']==r['high']==0
    result['paired_zero_control']=True
    for r in json.loads((A.HERE/'decomposition_audit.json').read_text()).values():
        assert abs(r['global_reduction']-r['sum_disjoint_category_contributions'])<1e-8
    result['error_decomposition_closes']=True
    for path in A.HERE.glob('*_static.npz'):
        a=dict(np.load(path)); assert np.array_equal(a['C'][:,0],a['C'][:,3])
        for p in C.policies(a['C'].reshape(-1,49),a['S'].reshape(-1,49)).values():
            assert np.allclose(p.sum(-1),1) and (p>=.2/49-1e-10).all()
    result['fixed_history_contingency_invariant_and_policy_valid']=True
    # Pins record exact inputs used by this dissection, not an inference of original lineage.
    paths=[A.HERE/'PLAN.md',A.HERE/'inference.py',A.HERE/'analyze.py',A.HERE/'verify.py',
           A.QREF,C.OLD/'native_confirmation_and_noise.npz',
           A.HERE.parent/'20261102_p1f15_readiness/static_repeats.npz',
           A.HERE.parent/'20261102_p1f15_readiness/static_repeats.json',
           A.HERE.parent/'20261102_p1f15_readiness/gaze_checks/static_10.npz']
    for cohort in ('saved','confirmation'):
        paths.extend(C.HERE/f'e3_{kind}_{seed}_endpoint_{cohort}.npz'
                     for kind in ('coarse','rich') for seed in A.SEEDS)
        for seed in A.SEEDS:
            paths.extend([C.HERE/f'e1_{seed}_u4000_{cohort}.npz',
                          C.HERE/f'e2_carry_{seed}_ck10_s4000_{cohort}.npz',
                          C.HERE/f'e2_fresh_{seed}_ck10_endpoint_{cohort}.npz'])
    paths.extend([Path('saccade/model.py'),Path('saccade/eval/gates.py'),
                  C.HERE/'campaign.py',C.HERE/'common.py',C.OLD/'direct.py'])
    result['sha256']={str(p):C.sha(p) for p in paths}
    C.write(A.HERE/'CHECKS.json',result)
    print(json.dumps({k:v for k,v in result.items() if k!='sha256'},indent=2))


if __name__=='__main__':main()
