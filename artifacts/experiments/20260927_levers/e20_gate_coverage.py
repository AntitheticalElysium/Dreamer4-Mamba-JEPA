"""Recover factual TRAIN/DEV terminal-calibration addresses without FINAL access.

This is evaluation data plumbing, not a measured pass or a new gate threshold.
All rows have four actual observed frames and one recorded successor. Build a
balanced diagnostic panel with one dead and one alive transition per episode;
same-depth evaluation removes generated-depth alias in this calibration panel.
No world is fitted or run. Gate label gaps remain explicit until integrated.
"""
import json
import sys
from pathlib import Path

import torch

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import e20_data as D
import e20_labels as L
import h16_resume as R


def main():
    torch.set_num_threads(3)
    episodes,contract=D.corpus()
    pictures=L.templates()
    selected=[];frames=[];actions=[];health=[];labels=[]
    for split in ('train','dev'):
        eligible=sorted((e for e in episodes if e.split==split and e.uniform_eligible and
                         len(e)>=14 and bool(e.terminated[-1])),key=lambda e:e.episode_id)
        permutation=torch.randperm(len(eligible),generator=torch.Generator().manual_seed(20261007))
        for i in permutation[:128].tolist():
            e=eligible[i]
            for dead,t in ((True,len(e)-1),(False,len(e)-11)):
                if t<3 or bool(e.terminated[t])!=dead:raise RuntimeError('Invalid evaluation address')
                src=e.observations[t-3:t+2].clone()
                hp=L.decode(src,pictures)
                frames.append(src);actions.append(e.actions_taken[t-3:t+1].clone())
                health.append(hp);labels.append(dead)
                selected.append({'episode_id':e.episode_id,'split':split,'outgoing_step':t,
                                 'context_frames':4,'generated_depth_to_evaluate':1,'dead':dead})
    assert len(selected)==512 and not any(r['split']=='final'for r in selected)
    trainids={r['episode_id']for r in selected if r['split']=='train'}
    devids={r['episode_id']for r in selected if r['split']=='dev'}
    assert trainids.isdisjoint(devids)
    path=D.OUT/'gate_calibration_candidates.pt'
    payload={'addresses':selected,'frames':torch.stack(frames),'actions':torch.stack(actions),
             'health':torch.stack(health),'dead':torch.tensor(labels),'dataset_contract':contract['sha256']}
    if path.exists():
        old=torch.load(path,map_location='cpu',weights_only=False)
        assert old['addresses']==selected
        for k in ('frames','actions','health','dead'):assert torch.equal(old[k],payload[k])
    else:R.atomic_torch(path,payload)
    baseline=D.ROOT/'artifacts/lewm_m4_canonical/raw/gates/h2'
    retention=json.loads((baseline/'semantic_retention.json').read_text())
    continuation=json.loads((baseline/'outcome_calibration.json').read_text())['continuation']
    report={'dataset_contract':contract['sha256'],'source':{str(Path(__file__)):R.file_hash(__file__)},
            'candidate_panel_sha256':R.file_hash(path),'addresses':len(selected),
            'counts':{s:{'alive':128,'dead':128,'episodes':128}for s in ('train','dev')},
            'baseline_continuation_coverage':continuation,
            'unsupported_retention_labels':[r['label']for r in retention['coverage']if not r['supported']],
            'missing_numeric_suite':['health','food','drink','energy','12 inventory quantities'],
            'FINAL_frames_inspected':0,'world_forward_calls':0,
            'scope':'recovered balanced factual calibration inputs, not population calibration or gate authorization',
            'not_completed':'numeric retention integration and simulator-verified expansion of missing predicates'}
    R.atomic_json(D.OUT/'gate_coverage.json',report,immutable=True)
    print(json.dumps(report),flush=True)


if __name__=='__main__':main()
