"""Output isolation, labels and one-shot checkpoint selection; no evaluation fits."""
import importlib
import json
import sys
import pytest
import torch
from saccade import train as TR

W=importlib.import_module('artifacts.experiments.20261102_p1f16.readings_codex')
R=importlib.import_module('artifacts.experiments.20261102_p1f16.run')


def test_sampler_wrapper_keeps_outputs_isolated_and_relabels_only(tmp_path,monkeypatch):
    monkeypatch.setattr(W,'HERE',tmp_path)
    monkeypatch.setattr(sys,'argv',['wrapper','sampler','candidate.pt'])
    def fake(path):
        assert str(path)=='candidate.pt'
        out=W.OLD.BR.OUT
        (out/'sampler_results.json').write_text(json.dumps(dict(models=dict(p1f15=dict(marker=17)))))
    monkeypatch.setattr(W.OLD,'sampler',fake)
    W.main()
    out=json.loads((tmp_path/'readings/codex/sampler_results.json').read_text())
    assert out['models']==dict(p1f16=dict(marker=17))
    assert W.OLD.BR.OUT==tmp_path/'readings/codex'


def test_panel_wrapper_calls_unchanged_reader_with_new_label(tmp_path,monkeypatch):
    monkeypatch.setattr(W,'HERE',tmp_path)
    monkeypatch.setattr(sys,'argv',['wrapper','panel','candidate.pt'])
    seen=[]
    monkeypatch.setattr(W.OLD.BR,'main',lambda:seen.append((sys.argv.copy(),dict(W.OLD.BR.WORLDS),W.OLD.BR.OUT)))
    W.main()
    assert seen[0][0][1:]==['panel','p1f16']
    assert str(seen[0][1]['p1f16'])=='candidate.pt'
    assert seen[0][2]==tmp_path/'readings/codex'


def test_selection_uses_saved_rule_and_refuses_repeated_evaluation(tmp_path):
    cfg=R.L.check_recipe(R.L.recipe(TR.load_cfg()))
    ck=dict(cfg=cfg,world_seed=1,state=dict(frames=cfg['p1']['frames']),provenance=TR.run_provenance(cfg))
    records=[]
    for i in range(1,11):
        torch.save(ck,tmp_path/f'ckpt_{i:02d}.pt')
        records.append(dict(checkpoint=i,frames=i*cfg['p1']['frames']/10,
            monitor=dict(g1=1/i,windows=cfg['p1']['monitor']['windows'])))
    (tmp_path/'log.jsonl').write_text('\n'.join(json.dumps(r) for r in records))
    path,decision=R.select(tmp_path)
    assert path.name=='ckpt_10.pt' and not decision['three_increases_triggered']
    assert (tmp_path/'s1/selection.json').exists()
    with pytest.raises(RuntimeError,match='already attempted'):R.select(tmp_path)
