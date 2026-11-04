"""Qualification of the data/evaluation handoff; no training or architecture changes."""
import importlib
import json
import os
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pytest

E = importlib.import_module('artifacts.experiments.20261104_p1f21.evaluation')
R = importlib.import_module('artifacts.experiments.20261104_p1f21.run')
V = importlib.import_module('artifacts.experiments.20261104_p1f21.eval_tools')


def test_queue_selected_checkpoint_and_output_isolation():
    ck = E.HERE/'run/ckpt_07.pt'
    queue = E.commands(ck)
    assert len(queue) == 20 and len({s[0] for s in queue}) == 20
    for name, cmd, cpu in queue:
        assert cmd[3] == E.TOOLS
        if name not in ('m05_summary', 'geometry'):
            assert any(str(ck) in arg for arg in cmd), name
    assert [n for n, _, _ in queue[-3:]] == ['own_dev', 'own_coverage', 'own_risk']
    assert all(any(str(E.BASE) in arg for arg in c) for n, c, _ in queue if n.startswith('interface_'))
    # Every explicit output belongs to the new arm, never an archived run.
    for _, c, _ in queue:
        for arg in c:
            if arg.endswith('.json') or arg.endswith('/uptake') or arg.endswith('/consistency'):
                assert arg.startswith(str(E.HERE)), arg


@pytest.mark.parametrize('failure', [False, True])
def test_worker_evaluation_after_selection_and_failure_status(tmp_path, failure):
    (tmp_path/'SOURCE_PINS.json').write_text('{}')
    (tmp_path/'gate.json').write_text('{"passed":true}')
    calls = []
    def child(cmd, **kw):
        calls.append(cmd)
        if failure and len(calls) == 2:raise RuntimeError('evaluation failed')
    selected = {'checkpoint_path': str(tmp_path/'run/ckpt_07.pt')}
    with patch.dict(os.environ, MAMBA_DETERMINISTIC='1'), patch.object(R, 'validate_sources'), \
         patch.object(R, 'select', return_value=selected), patch.object(R.subprocess, 'run', side_effect=child):
        if failure:
            with pytest.raises(RuntimeError, match='evaluation failed'):
                R.main(here=tmp_path, gates=('gate.json',), evaluation_module='the.evaluation')
        else:R.main(here=tmp_path, gates=('gate.json',), evaluation_module='the.evaluation')
    assert calls[1][-1] == selected['checkpoint_path']
    state = json.loads((tmp_path/'STATUS.json').read_text())
    assert state['phase'] == ('failed' if failure else 'complete')
    if failure:assert state['failed_phase'] == 'evaluation'


def test_union_eval_membership_and_heldout_labels():
    V.setup()
    from saccade import data as D, train as TR
    union = D.load_split(V.U.cfg_union())
    old = D.load_split(TR.load_cfg())
    assert len(union['train']) == 22267 and len(union['monitor']) == 397
    assert len(old['train']) == 7672 and len(old['dev']) == 1009
    key = lambda eps: {(e.shard, e.slot) for e in eps}
    assert all(e.shard < 420 for split in old.values() for e in split)
    assert key(old['monitor']) == key(union['monitor'])
    train = key(union['train']); held = key(union['held_out'])
    assert not (train & held)
    both = set()
    for source, episodes, transitions in [('coverage', 140, 168701), ('risk', 625, 142840)]:
        z = dict(np.load(E.HERE/f'heldout_{source}.npz'))
        keys = set(zip(z['ep_shard'], z['ep_slot']))
        assert len(keys) == episodes and not (keys & train)
        both |= keys
        assert int(z['ep_T'].sum()) == transitions
        assert len(z['s_body']) == transitions + episodes
        assert len(z['t_action']) == transitions
        # Direct reads: labels use the original collection's local slot and
        # union shard offset, with the exact recorded actions/rewards/termination.
        off = np.r_[0, np.cumsum(z['ep_T'])]
        for i in [0, len(keys)//2, len(keys)-1]:
            fields = D.RP.episode_fields(int(z['ep_shard'][i]), int(z['ep_slot'][i]))
            for label, field in [('t_action', 'actions_taken'), ('t_reward', 'rewards'), ('t_terminated', 'terminated')]:
                np.testing.assert_array_equal(z[label][off[i]:off[i+1]], fields[field].numpy())
    assert both == held
