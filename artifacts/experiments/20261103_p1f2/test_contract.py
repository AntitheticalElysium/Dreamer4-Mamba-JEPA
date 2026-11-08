"""Mechanical isolation, causal inputs, historical fixation and policy contracts."""
import ast
import copy
import importlib
import inspect
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from saccade import model as M, train as TR
C=importlib.import_module('artifacts.experiments.20261103_p1f2.chooser')
T=importlib.import_module('artifacts.experiments.20261103_p1f2.trainer')
L=importlib.import_module('artifacts.experiments.20261103_p1f2.train_p1f2')


def test_world_loop_ast_matches_canonical_except_declared_hooks():
    a=ast.parse(inspect.getsource(TR.p1)).body[0]
    b=ast.parse(inspect.getsource(T.p1)).body[0];b.args.args.pop();b.args.defaults.pop()
    class Strip(ast.NodeTransformer):
        def visit_Attribute(self,n):
            if isinstance(n.value,ast.Name) and n.value.id=='TR':return ast.copy_location(ast.Name(n.attr,ast.Load()),n)
            return self.generic_visit(n)
        def visit_Assign(self,n):
            if any(isinstance(t,ast.Name) and t.id=='controller' for t in n.targets):return None
            if any(isinstance(t,ast.Subscript) and isinstance(t.value,ast.Name) and t.value.id=='ck' and isinstance(t.slice,ast.Constant) and t.slice.value=='fixation_controller' for t in n.targets):return None
            return self.generic_visit(n)
        def visit_Expr(self,n):
            if isinstance(n.value,ast.Call) and isinstance(n.value.func,ast.Attribute) and isinstance(n.value.func.value,ast.Name) and n.value.func.value.id=='controller':return None
            return self.generic_visit(n)
    assert ast.dump(a,include_attributes=False)==ast.dump(Strip().visit(b),include_attributes=False)


def synthetic():
    rng=np.random.default_rng(8);n,L=2,20
    c=dict(frames=rng.integers(256,size=(n,L,63,63,3),dtype=np.uint8),
        prev_action=rng.integers(17,size=(n,L)),reward=rng.random((n,L),dtype=np.float32),
        terminated=np.zeros((n,L),bool),valid=np.ones((n,L),bool),
        episode=np.array([[1]*12+[2]*8,[3]*20]),target=np.arange(n*L).reshape(n,L),
        fixation=rng.integers(1,8,size=(n,L,2)))
    c['prev_action'][0,12]=-1
    return c


def test_online_inputs_match_original_head_history_and_reset():
    c=synthetic();p=torch.randn(2,20,81,16);v=torch.randn(2,20,8);x,y=C.inputs(p,v,c,8)
    # Compare full original preprocessing independently on each same-episode sequence.
    for row in range(2):
        for t in range(8,20):
            if c['prev_action'][row,t]<0:continue
            begin=0 if row or t<12 else 12
            z={k:torch.as_tensor(c[k][row,begin:]) for k in ('prev_action','reward','terminated')}
            z.update(p=p[row,begin:],v=v[row,begin:],t=torch.arange(20-begin),offsets=np.array([0,20-begin]))
            old=C.Q.Data(z,torch.device('cpu')).inputs(torch.tensor([t-begin]));i=row*12+t-8
            assert all(torch.equal(a[i:i+1],b) for a,b in zip(x,old))
    before=tuple(a[0].clone() for a in x)
    p[:,8:]+=100;v[:,8:]+=100;c['reward'][:,8:]+=100;c['terminated'][:,8:]=True
    now,_=C.inputs(p,v,c,8)
    assert all(torch.equal(a,b[0]) for a,b in zip(before,now))
    assert y.shape==(24,81,16)


def test_coarse_is_exact_and_never_uses_fine(monkeypatch):
    torch.set_num_threads(2);pc=M.Perception(TR.load_cfg());frames=torch.randint(256,(2,63,63,3),dtype=torch.uint8)
    full=pc(frames,torch.tensor([[2,3],[5,6]]));p,v=C.coarse(pc,frames)
    assert torch.equal(p,full.p) and torch.equal(v,full.v)
    def forbidden(*a,**k):raise AssertionError('fine crop computed by chooser')
    monkeypatch.setattr(M,'crop',forbidden);C.coarse(pc,frames)


def test_policy_reference_uniform_floor_and_noop():
    torch.manual_seed(14);h=C.Q.Head();torch.nn.init.normal_(h.output.weight,std=.01)
    x=(torch.randn(3,8,81,16),torch.randn(3,232),torch.zeros(3,dtype=torch.long))
    assert torch.equal(h(x),h((x[0],x[1],torch.zeros_like(x[2]))))
    c,s=torch.rand(3,49),torch.rand(3,49)
    ref=.8*torch.softmax(.5*c/c.mean(-1,keepdim=True)+.5*s/s.mean(-1,keepdim=True),-1)+.2/49
    assert torch.equal(C.Q.policy(c,s),ref)
    assert torch.all(C.Q.policy(c,s)>=.2/49)
    assert torch.allclose(C.Q.policy(torch.zeros_like(c),torch.zeros_like(s)),torch.full_like(c,1/49))


def test_controller_rng_gradients_prefix_and_resume(tmp_path):
    torch.set_num_threads(2);cfg=TR.load_cfg();world=SimpleNamespace(perception=M.Perception(cfg))
    cfg['p1']['frames']=4096;pol=C.config();pol['world_frames_per_head_update']=1;pol['batch_size']=4
    # Saturate update clock to two rather than 24 updates in this mechanics check.
    ctrl=C.Controller(world,cfg,pol,tmp_path,'cpu');ctrl.frames=-22
    c=synthetic();hist=[[tuple([0,1,t,*c['fixation'][i,t]]) for t in range(12,20)] for i in range(2)]
    streams=SimpleNamespace(hist=hist);before=copy.deepcopy(c);rng=torch.get_rng_state().clone()
    ctrl.prepare(c,streams)
    assert torch.equal(rng,torch.get_rng_state())
    for k in c:
        if k!='fixation':assert np.array_equal(c[k],before[k]),k
    assert np.array_equal(c['fixation'][:,:8],before['fixation'][:,:8])
    assert all(tuple(streams.hist[i][j][3:])==tuple(c['fixation'][i,12+j]) for i in range(2) for j in range(8))
    for p in world.perception.parameters():p.grad=None
    ctrl.learn(24);assert ctrl.updates==2
    assert all(p.grad is None for p in world.perception.parameters())
    state=copy.deepcopy(ctrl.state());(tmp_path/'latest.pt').touch()
    data=ctrl.trace.read_bytes();log=ctrl.log.read_bytes()
    ctrl.trace.write_bytes(data+b'bad-tail');ctrl.log.write_bytes(log+b'bad-tail')
    again=C.Controller(world,cfg,pol,tmp_path,'cpu');again.load_state(state)
    assert again.trace.read_bytes()==data and again.log.read_bytes()==log
    assert torch.equal(again.gaze.get_state(),ctrl.gaze.get_state())
    assert torch.equal(again.batch.get_state(),ctrl.batch.get_state())
    changed=copy.deepcopy(state);changed['config']['uniform_floor']=.3
    with pytest.raises(AssertionError,match='recipe'):again.load_state(changed)


def test_recipe_is_actual_f16_configuration():
    L.check_recipe(L.recipe(TR.load_cfg()))
