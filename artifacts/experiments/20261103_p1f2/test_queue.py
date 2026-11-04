"""Selection, failure dispatch and estimator/output isolation, without long evaluation fits."""
import importlib
import json
from pathlib import Path
import sys
import numpy as np
import pytest
import torch
from saccade import train as TR
W=importlib.import_module('artifacts.experiments.20261103_p1f2.readings_codex')
R=importlib.import_module('artifacts.experiments.20261103_p1f2.run')
F=importlib.import_module('artifacts.experiments.20261103_p1f2.fovea_readings')


def test_sampler_wrapper_isolated_and_relabels(tmp_path,monkeypatch):
    monkeypatch.setattr(W,'HERE',tmp_path);monkeypatch.setattr(sys,'argv',['wrapper','sampler','candidate.pt'])
    def fake(path):
        assert str(path)=='candidate.pt'
        (W.OLD.BR.OUT/'sampler_results.json').write_text(json.dumps(dict(models=dict(p1f15=dict(marker=17)))))
    monkeypatch.setattr(W.OLD,'sampler',fake);W.main()
    assert json.loads((tmp_path/'readings/codex/sampler_results.json').read_text())['models']==dict(p1f2=dict(marker=17))


def test_panel_wrapper_uses_original_entrypoint(tmp_path,monkeypatch):
    monkeypatch.setattr(W,'HERE',tmp_path);monkeypatch.setattr(sys,'argv',['wrapper','panel','candidate.pt'])
    seen=[];monkeypatch.setattr(W.OLD.BR,'main',lambda:seen.append((sys.argv.copy(),W.OLD.BR.WORLDS.copy(),W.OLD.BR.OUT)))
    W.main();args,worlds,out=seen[0]
    assert args[1:]==['panel','p1f2'] and str(worlds['p1f2'])=='candidate.pt'
    assert out==tmp_path/'readings/codex'


def test_genuine_look_summary_identity_and_paired_cohort(tmp_path):
    source=F.BASE/'p1f16/local.npz'
    for name in ('p1f16','p1f2'):
        (tmp_path/name).mkdir();(tmp_path/name/'local.npz').symlink_to(source)
    s=F.summary(tmp_path)
    assert s['worlds']['p1f16']==s['worlds']['p1f2']
    assert all(q['mean']==0 for q in s['paired_fraction_change'].values())
    z=dict(np.load(source));z['t']=z['t']+1
    (tmp_path/'p1f2/local.npz').unlink();np.savez(tmp_path/'p1f2/local.npz',**z)
    with pytest.raises(AssertionError,match='t'):F.summary(tmp_path)


def test_selection_saved_rule_complete_and_one_shot(tmp_path):
    cfg=R.L.check_recipe(R.L.recipe(TR.load_cfg()));prov=TR.run_provenance(cfg)
    ck=dict(cfg=cfg,world_seed=1,state=dict(frames=cfg['p1']['frames']),provenance=prov,
            fixation_controller=dict(config=R.L.C.config(),frames=cfg['p1']['frames']))
    records=[]
    for i in range(1,11):
        torch.save(ck,tmp_path/f'ckpt_{i:02d}.pt')
        records.append(dict(checkpoint=i,frames=i*cfg['p1']['frames']/10,monitor=dict(g1=1/i,windows=cfg['p1']['monitor']['windows'])))
    (tmp_path/'log.jsonl').write_text('\n'.join(json.dumps(r) for r in records))
    path,decision=R.select(tmp_path)
    assert path.name=='ckpt_10.pt' and not decision['three_increases_triggered']
    with pytest.raises(RuntimeError,match='already attempted'):R.select(tmp_path)


@pytest.mark.parametrize('failure',[False,True])
def test_one_shot_dispatch_no_successor_after_failure(tmp_path,monkeypatch,failure):
    monkeypatch.setattr(R,'HERE',tmp_path);monkeypatch.setattr(R,'validate_sources',lambda:{'files':{}})
    monkeypatch.setattr(sys,'argv',['run']);monkeypatch.setenv('MAMBA_DETERMINISTIC','1')
    (tmp_path/'SOURCE_PINS.json').write_text('{}');(tmp_path/'checks.json').write_text('{"passed":true}')
    selected=tmp_path/'run/ckpt_10.pt';calls=[]
    monkeypatch.setattr(R,'select',lambda run:(selected,dict(checkpoint=10)))
    def run(args,**kwargs):
        calls.append(args)
        if failure:raise RuntimeError('intentional training failure')
    monkeypatch.setattr(R.subprocess,'run',run)
    if failure:
        with pytest.raises(RuntimeError,match='intentional'):R.main()
    else:R.main()
    state=json.loads((tmp_path/'STATUS.json').read_text())
    assert state['phase']==('failed' if failure else 'complete')
    assert len(calls)==(1 if failure else 2)
    if not failure:assert calls[1]==['bash',str(tmp_path/'queue.sh'),str(selected)]
    assert not any('p2' in str(arg).lower() or 'actor.train' in str(arg) for call in calls for arg in call)
