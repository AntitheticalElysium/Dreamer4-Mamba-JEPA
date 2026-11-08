"""P1-F1.6 joint fusion route checks (canonical since F2; the unfused belief runs from commit fa6edd1a)."""
import numpy as np
import pytest
import torch
from saccade import data as D, model as M, train as TR


def percept(shape=(3, 4)):
    return M.Percept(torch.randn(*shape, 81, 16), torch.randn(*shape, 9, 64),
                     torch.randn(*shape, 8), torch.randint(1, 8, (*shape, 2)))


def test_fusion_draws_are_its_own_and_reload_exact():
    cfg = TR.load_cfg()
    torch.manual_seed(1); a = M.M0(cfg)
    torch.manual_seed(2); b = M.M0(cfg)
    fa, fb = a.belief.spatial_fusion.state_dict(), b.belief.spatial_fusion.state_dict()
    assert all(torch.equal(fa[k], fb[k]) for k in fa)                  # its own seed, whatever the global stream
    assert sum(p.numel() for p in b.belief.spatial_fusion.parameters()) == 19648
    clone = M.M0(cfg); clone.load_state_dict(b.state_dict(), strict=True)
    for shape in ((), (7,), (3,4)):
        P = percept(shape)
        assert torch.equal(b.belief.percept_embed(P), clone.belief.percept_embed(P))


def test_fusion_is_shared_local_scale_bounded_and_observation_immutable():
    cfg = TR.load_cfg(); bel = M.Belief(cfg)
    P = percept((49,)); P.fix = torch.cartesian_prod(torch.arange(1,8),torch.arange(1,8))
    old = [x.clone() for x in P.parts()]
    cells = M.window(P.fix); out = bel.spatial_fusion(P,cells)
    assert out.shape == P.g.shape and torch.isfinite(out).all()
    assert float(out.detach().abs().max()) < cfg['perception']['periphery']['tanh_clamp']
    assert all(torch.equal(a,b) for a,b in zip(old,P.parts()))
    alt = P.p.clone(); mask = torch.zeros(49,81,dtype=torch.bool).scatter(1,cells,True)
    alt[~mask] = 100
    assert torch.equal(out,bel.spatial_fusion(M.Percept(alt,P.g,P.v,P.fix),cells))
    fix = torch.tensor([[1,1],[6,6]]); cell=M.window(fix); idx=cell[...,None].expand(2,9,16)
    p=torch.zeros(2,81,16).scatter(1,idx,torch.randn(1,9,16).expand(2,-1,-1))
    pair=M.Percept(p,torch.randn(1,9,64).expand(2,-1,-1),torch.randn(1,8).expand(2,-1),fix)
    joint=bel.spatial_fusion(pair,cell)
    assert torch.equal(joint[0],joint[1])


def test_no_raw_fine_bypass_into_belief(monkeypatch):
    cfg=TR.load_cfg(); bel=M.Belief(cfg); P=percept((2,))
    monkeypatch.setattr(bel.spatial_fusion,'forward',lambda p,c:torch.zeros_like(p.g))
    alternative=M.Percept(P.p,P.g+10,P.v,P.fix)
    assert torch.equal(bel.percept_embed(P),bel.percept_embed(alternative))


def test_fusion_connects_all_three_inputs_and_parameters():
    cfg=TR.load_cfg(); bel=M.Belief(cfg); P=percept((2,))
    for t in P.parts():t.requires_grad_(True)
    bel.spatial_fusion(P,M.window(P.fix))[...,0].square().sum().backward()
    assert all(t.grad is not None and torch.isfinite(t.grad).all() and t.grad.abs().sum()>0 for t in P.parts())
    assert all(t.grad is not None and torch.isfinite(t.grad).all() and t.grad.abs().sum()>0
               for t in bel.spatial_fusion.parameters())


@pytest.mark.skipif(not torch.cuda.is_available(),reason='needs native CUDA Mamba')
def test_native_future_gradient_after_real_full_objective_warmup():
    cfg=TR.load_cfg(); dev=torch.device('cuda'); torch.manual_seed(1)
    b=M.M0(cfg).to(dev); rng=torch.cuda.get_rng_state()
    M.SpatialFusion(16,64,8,128,3.)
    assert torch.equal(rng,torch.cuda.get_rng_state())
    split=D.load_split(cfg); streams=D.Streams(split['train'][:16],2,8,seed=1)
    opt=torch.optim.AdamW(b.param_groups(),lr=cfg['optimizer']['lr'])
    carry=b.belief.initial(2,dev); cache=torch.zeros(2,9,256,device=dev)
    for _ in range(8):
        terms,carry,cache=TR.p1_losses(b,streams.next(8),carry,cache,None,cfg,np.zeros(2,bool),0)
        opt.zero_grad(set_to_none=True)
        (b.harmoniser(TR.groups(terms))+terms['noise']).backward()
        torch.nn.utils.clip_grad_norm_(b.parameters(),cfg['optimizer']['grad_clip']);opt.step()
        # Match the production trainer's carry_put boundary; do not backpropagate through prior chunks twice.
        carry=tuple(M.MambaCarry(c.conv.detach(),c.ssm.detach()) for c in carry)
    terms,_,_=TR.p1_losses(b,streams.next(8),carry,cache,None,cfg,np.zeros(2,bool),0)
    opt.zero_grad(set_to_none=True);terms['g1'].backward()
    for p in b.belief.spatial_fusion.parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum()>0
    assert all(p.requires_grad for p in b.parameters())
