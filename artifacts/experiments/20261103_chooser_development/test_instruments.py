"""Mechanism checks for the new instruments, not duplicate implementation assertions."""
import importlib
import numpy as np
import torch

C = importlib.import_module('artifacts.experiments.20261103_chooser_development.common')
E = importlib.import_module('artifacts.experiments.20261103_chooser_development.campaign')


def toy():
    torch.manual_seed(611760)
    return dict(p=torch.randn(24,81,16), v=torch.randn(24,8), reward=torch.randn(24),
                terminated=torch.zeros(24,dtype=torch.bool), prev_action=torch.arange(24)%17,
                episode=torch.tensor([0]*12+[1]*12), t=torch.tensor(list(range(12))*2),
                monitor=torch.zeros(24,dtype=torch.bool), offsets=np.array([0,12,24]))


def test_policy_contracts():
    assert all(C.checks().values())


def test_existing_rule_matches_archived_implementation():
    r = np.random.default_rng(611761)
    c, s = r.random((9,49)), r.random((9,49))
    old = E.Q.policy(torch.tensor(c), torch.tensor(s)).numpy()
    assert np.allclose(old, C.policies(c,s)['existing'], atol=1e-12)


def test_bootstrap_pairing_and_missing_counter():
    meta = dict(cls=np.array(['drink']*4+['random']*4), ep=np.array([0,0,1,1]*2),
                counter=np.array([65]*4+[-1]*4), faced=np.array([31]*8), light=np.ones(8))
    c=np.ones((8,49)); s=c.copy()
    out=C.summarize(meta,c,s,dict(counter_mse=np.arange(8),factual_mse=np.ones(8)),draws=80)
    for q in out.values():
        for p in q['policies'].values():
            for k in ('remote_mass','contact_mass','union_mass'):
                assert abs(p[k]['minus_existing']['mean']) < 1e-12
                assert abs(p[k]['minus_existing']['low']) < 1e-12
    assert 'counter_mse' not in out['random']['errors']


def test_forward_auxiliary_excludes_target_and_reset():
    h=E.AuxHead(); torch.nn.init.normal_(h.output.weight,std=.02)
    for i in (2,13,21):
        z=toy(); aux=E.sketch(z)
        data=E.AuxData(z,torch.device('cpu'),aux,torch.zeros(256),torch.ones(256))
        ix=torch.tensor([i]); before=h(data.inputs(ix)).detach()
        # Test independently: a later query is allowed to see an earlier frame.
        z['p'][i]=900; z['v'][i]=900; z['reward'][i]=900; z['terminated'][i]=True
        assert torch.equal(before,h(data.inputs(ix)).detach())
    assert torch.count_nonzero(aux[[0,12]])==0
    p,ctx,act,a=data.inputs(torch.tensor([13]))
    assert torch.count_nonzero(p[:,:-1])==0
    # Identical histories and queried NOOP must give exactly zero contrast.
    data.z['prev_action'][ix]=0
    pr,_,_=E.predict(h,data,ix)
    assert not np.any(pr['C'])
    h(data.inputs(ix)).square().mean().backward()
    assert h.aux.weight.grad.norm()>0


def test_forecast_instrument_matches_original():
    z=toy(); data=E.Q.Data(z,torch.device('cpu')); h=E.Q.Head()
    torch.nn.init.normal_(h.output.weight,std=.02)
    ix=torch.tensor([2,13,21]); new,_,_=E.predict(h,data,ix); old=E.Q.predict(h,data,ix)
    assert all(np.array_equal(new[k],old[k]) for k in old)


def test_auxiliary_uses_training_only_statistics():
    z=toy(); z['episode'][12:]=360; z['monitor'][10]=True
    aux=torch.randn(24,256); a,b=E.aux_stats(aux,z)
    aux[12:]=1e5; aux[10]=-1e5
    aa,bb=E.aux_stats(aux,z)
    assert torch.equal(a,aa) and torch.equal(b,bb)


def test_carry_schedule_does_not_restart():
    assert E.lr_at(4000,12000) > E.lr_at(8000,12000)
    assert E.lr_at(0,12000) < E.lr_at(4000,12000)
    # Original E1 schedule retained exactly; E2 is intentionally a different clock.
    assert E.lr_at(3999,4000) < E.lr_at(3999,12000)
