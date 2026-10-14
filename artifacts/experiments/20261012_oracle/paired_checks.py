"""Real-parent causal/live-noise and interrupted B16 resume proofs.

These checks validate execution. They do not establish physical correctness or
that the paired event parameterization has learned coherent movement.
"""
import argparse
import gc
import json
from pathlib import Path

import torch

import direct_model as DIRECT
import direct_train as BASE
import paired_model as M
import paired_train as T
import loss_intervention as L
from d4mj.train import autocast_context


def maxdiff(a, b):
    if isinstance(a, torch.Tensor):
        assert isinstance(b, torch.Tensor) and a.shape == b.shape and a.dtype == b.dtype
        return float((a.detach().cpu().double() - b.detach().cpu().double()).abs().max()) if a.numel() else 0.
    if isinstance(a, dict):
        assert a.keys() == b.keys()
        return max((maxdiff(a[k], b[k]) for k in a), default=0.)
    if isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        return max((maxdiff(x, y) for x, y in zip(a, b)), default=0.)
    assert a == b
    return 0.


def sources():
    paths = (Path(__file__), Path(M.__file__), Path(T.__file__), Path(DIRECT.__file__), Path(BASE.__file__))
    return {str(path.relative_to(L.ROOT)): L.sha(path) for path in paths}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--resume-proof', action='store_true')
    args = parser.parse_args()
    config = T.settings()
    manifest, windows, actions, tokens = T.load_data()
    worlds = T.initialized_worlds('cuda')
    assert tuple(worlds) == ('direct', 'paired')
    states = {arm: L.cpu_state(world) for arm, world in worlds.items()}
    common = states['direct'].keys() & states['paired'].keys()
    # Every common tensor is intentionally shared, including the six-output
    # routing readout whose interpretation changes in the paired arm.
    shared = sorted(common)
    assert shared and all(torch.equal(states['direct'][name], states['paired'][name]) for name in shared)
    original = torch.load(BASE.OUT / 'training/energy_u0000.pt', map_location='cpu', weights_only=False)
    direct_initial = L.restored_equal(states['direct'], original['world'])
    assert direct_initial, 'direct control must preserve historical initialization exactly'
    frames, ac, _ = T.batch(tokens, actions, windows, torch.tensor([0, 1]))
    saved_noise = DIRECT.draw_noise(frames.shape[:2], torch.Generator().manual_seed(918)).cuda()
    assert not torch.equal(saved_noise[0], saved_noise[1])
    changed = frames.clone()
    changed[:, -1] = torch.randn_like(changed[:, -1])
    execution = {}
    for arm in M.ARMS:
        world = worlds[arm].eval()
        noise = saved_noise.detach().clone().requires_grad_()
        with autocast_context(config):
            h, ha = world.backbone_full(frames, ac)
            prediction = torch.stack([world.complete(frames, ac, h, ha, n)[0] for n in noise])
        assert prediction.shape == (2, 2, 16, 81, 192) and bool(torch.isfinite(prediction).all())
        values = DIRECT.losses(prediction[:, :, :15], frames[:, 1:])
        values['energy'].backward()
        draw_gradients = [float(noise.grad[k].float().norm()) for k in range(2)]
        local_gradients = [float(noise.grad[k, ..., :4].float().norm()) for k in range(2)]
        shared_gradients = [float(noise.grad[k, ..., 4:].float().norm()) for k in range(2)]
        assert all(value > 0 for value in draw_gradients + local_gradients + shared_gradients)
        assert bool(torch.isfinite(noise.grad).all())
        backbone_gradient = sum(float(param.grad.float().square().sum()) for name, param in world.named_parameters()
                                if name.startswith('layers.') and param.grad is not None) ** .5
        assert backbone_gradient > 0
        output_effect = float((prediction[0].detach() - prediction[1].detach())[..., DIRECT.INTERIOR, :].abs().max())
        assert output_effect > 1e-6, 'noise must affect scored post-LN outputs'
        with torch.no_grad(), autocast_context(config):
            repeat = torch.stack([world.complete(frames, ac, h.detach(), ha.detach(), n)[0] for n in noise.detach()])
            hc, hac = world.backbone_full(changed, ac)
            changed_prediction = torch.stack([world.complete(changed, ac, hc, hac, n)[0] for n in noise.detach()])
        replay_error = float((repeat - prediction.detach()).abs().max())
        causal_error = float((changed_prediction[:, :, :15] - prediction.detach()[:, :, :15]).abs().max())
        assert replay_error == 0 and causal_error == 0
        assert torch.count_nonzero(DIRECT.distance(frames, frames)) == 0
        assert torch.equal(DIRECT.distance(prediction[0], prediction[1]), DIRECT.distance(prediction[1], prediction[0]))
        first_power = torch.allclose(DIRECT.distance(2 * prediction[0], 2 * prediction[1]),
                                     2 * DIRECT.distance(prediction[0], prediction[1]), atol=1e-7, rtol=1e-6)
        assert first_power
        execution[arm] = {'noise_replay_maxabs': replay_error, 'future_mutation_current_output_maxabs': causal_error,
                          'independent_draw_output_maxabs': output_effect, 'draw_gradient_norms': draw_gradients,
                          'local_noise_gradient_norms': local_gradients, 'shared_noise_gradient_norms': shared_gradients,
                          'backbone_gradient_norm': backbone_gradient, 'first_power_linear_scaling': bool(first_power),
                          'parameters': sum(param.numel() for param in world.parameters()),
                          'completion_parameters': sum(param.numel() for param in world.completion.parameters())}
        world.zero_grad(set_to_none=True)
        del prediction, repeat, changed_prediction, noise, h, ha, hc, hac
    proof = {'passed': False, 'shared_backbone_and_trunk_exact': True, 'shared_tensors': sorted(shared),
             'historical_direct_initial_state_exact': direct_initial, 'arms': execution,
             'data_contract_id': manifest['contract_id'],
             'precision': 'actual real-parent BF16/IEEE, gradients and post-LayerNorm outputs',
             'sources': sources()}
    del worlds
    gc.collect()
    torch.cuda.empty_cache()
    if args.resume_proof:
        resumed = torch.load(T.OUT / 'smoke/state.pt', map_location='cpu', weights_only=False)
        reference = torch.load(T.OUT / 'smoke_reference/state.pt', map_location='cpu', weights_only=False)
        assert resumed['contract_id'] == reference['contract_id'] and resumed['update'] == reference['update'] == 4
        streams = {name: L.restored_equal(resumed[name], reference[name]) for name in
                   ('order', 'noise_rng', 'torch_rng', 'cuda_rng', 'ledger_chain')}
        differences = {name: maxdiff(resumed[name], reference[name]) for name in ('worlds', 'optimizers')}
        assert all(streams.values()) and max(differences.values()) < 1e-4
        peak = max(value['peak_allocated_gib'] for payload in (resumed, reference)
                   for value in payload['totals'].values())
        assert peak < 6
        proof.update(passed=True, resume_streams_exact=streams, resume_endpoint_maxabs=differences,
                     real_batch16_peak_allocated_gib=peak, resumed_vs_uninterrupted_updates=4)
    L.atomic(proof, T.OUT / ('validation.json' if args.resume_proof else 'initial_validation.json'), True)
    print(json.dumps(proof), flush=True)


if __name__ == '__main__':
    main()
