"""CPU chronology/detachment contract checks for generated-prefix training.

No model training, GPU use, pool reads, or notebook writes. Synthetic exact
transitions expose off-by-one actions/outcomes, crossed rows and short windows.
The trainer's --stop-after path supplies the separate real GPU resume check.
"""
from __future__ import annotations

import json
from types import SimpleNamespace

import torch
import torch.nn as nn

import prefix_intervention as P


class FakeCond(nn.Module):
    def __init__(self):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(()))

    def forward(self, layout, frames):
        assert layout.shape[1] == frames == 15
        out = layout.new_zeros(len(layout), frames, 81, 4)
        out[:, :, :63] = layout
        out[..., 3] = self.bias
        return out


class FakeWorld(nn.Module):
    def __init__(self):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(()))
        self.calls = []
        self.delta = None

    def forward(self, frames, actions):
        assert frames.shape[1] == 15
        assert torch.equal(actions.float(), frames[:, :, 0, 0]), 'outgoing action off by one'
        assert torch.equal(self.delta[:, :, 0, 0], frames[:, :, 0, 0]+1), 'successor oracle off by one'
        assert torch.equal(self.delta[:, :, 0, 1], frames[:, :, 0, 1]), 'source rows crossed'
        self.calls.append({'gradient':torch.is_grad_enabled(), 'last':frames[:, -1, 0, 0].tolist()})
        out = frames.clone()
        out[..., 0] += 1
        out[..., 2] += 2
        out[..., 3] += self.bias+self.delta[..., 3]
        return out, None, None


def main():
    torch.set_num_threads(2)
    time = torch.arange(64, dtype=torch.float32)[None, :, None].expand(2,64,81)
    row = torch.arange(2, dtype=torch.float32)[:, None, None].expand(2,64,81)
    stream = torch.stack([time,row,2*time+row,torch.zeros_like(time)], -1)
    actions = torch.arange(63)[None].expand(2,63)
    layouts = stream[:, :, :63].clone()
    targets = torch.tensor([30,63])
    config = SimpleNamespace(runtime=SimpleNamespace(device='cpu',precision='fp32'))
    results = []
    for exposure in (0,*P.EXPOSURES):
        world,cond = FakeWorld(),FakeCond()
        teacher,target,_ = P.endpoint(world,cond,stream,actions,layouts,targets,exposure,False,config)
        world.calls.clear()
        pred,second_target,inputs = P.endpoint(world,cond,stream,actions,layouts,targets,exposure,True,config)
        assert torch.equal(teacher,pred) and torch.equal(target,second_target)
        assert torch.equal(inputs,P.window(stream,targets-15,15)), 'perfect generated prefixes must match real history'
        assert not inputs.requires_grad
        assert len(world.calls) == exposure+1
        assert all(not call['gradient'] for call in world.calls[:-1])
        assert world.calls[-1]['gradient']
        assert world.calls[-1]['last'] == [29.,62.]
        # Perturb only an unused target coordinate so alignment stays exact,
        # and verify final completion sends real gradients to both modules.
        offset_target = target.clone(); offset_target[...,3] += .3
        loss = P.LI.objective(P.LI.components(pred[:,None],offset_target[:,None]),'D')
        loss.backward()
        assert world.bias.grad is not None and float(world.bias.grad.abs()) > 0
        assert cond.bias.grad is not None and float(cond.bias.grad.abs()) > 0
        assert world.delta is None
        results.append({'exposure':exposure, 'prefix_calls':exposure, 'completion_calls':1,
                        'max_perfect_prefix_error':float((teacher-pred).detach().abs().max()),
                        'world_gradient':float(world.bias.grad),'conditioner_gradient':float(cond.bias.grad),
                        'endpoint_context':15,'last_input_frames':[29,62],'target_frames':[30,63]})
    # The exposure stream never touches the old main-row sampler stream.
    original = torch.Generator().manual_seed(11)
    matched = torch.Generator().manual_seed(11)
    extra = torch.Generator().manual_seed(20261012)
    for _ in range(20):
        left = torch.randint(100,(16,),generator=original)
        right = torch.randint(100,(16,),generator=matched)
        torch.randint(30,64,(16,),generator=extra); torch.randint(5,(),generator=extra)
        assert torch.equal(left,right), 'exposure draws changed main sampler'
    print(json.dumps({'status':'pass','checks':results,'main_sampler_stream_preserved':True,
                      'scope':'synthetic chronology, full-prefix detach and final gradient; not a real-model resume test'},indent=2))


if __name__ == '__main__': main()
