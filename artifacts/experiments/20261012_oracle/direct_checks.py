"""Real-parent independent-noise, causal-boundary and durable-resume proofs."""
import argparse
import json
from pathlib import Path

import torch

import direct_model as M
import direct_train as T
import loss_intervention as L
from d4mj.train import autocast_context


def maxdiff(a, b):
    if isinstance(a, torch.Tensor):
        return float((a.detach().cpu().double() - b.detach().cpu().double()).abs().max()) if a.numel() else 0.
    if isinstance(a, dict):
        assert a.keys() == b.keys()
        return max((maxdiff(a[k], b[k]) for k in a), default=0.)
    if isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        return max((maxdiff(x, y) for x, y in zip(a, b)), default=0.)
    assert a == b
    return 0.


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--resume-proof', action='store_true')
    a = p.parse_args()
    config = T.settings()
    manifest, windows, actions, tokens = T.load_data()
    worlds = T.initialized_worlds('cuda')
    assert L.restored_equal(L.cpu_state(worlds['point']), L.cpu_state(worlds['energy']))
    frames, ac, _ = T.batch(tokens, actions, windows, torch.tensor([0, 1]))
    noise = M.draw_noise(frames.shape[:2], torch.Generator().manual_seed(918)).cuda().requires_grad_()
    assert not torch.equal(noise[0], noise[1])
    world = worlds['energy'].eval()
    with autocast_context(config):
        predictions, routes, generated = world.samples(frames, ac, noise)
    with torch.no_grad(), autocast_context(config):
        twin = worlds['point'].eval().samples(frames, ac, noise.detach())[0]
    parity = float((predictions.detach() - twin).abs().max())
    assert parity == 0, parity
    scores = M.losses(predictions[:, :, :15], frames[:, 1:])
    scores['energy'].backward()
    gradients = [float(noise.grad[k].norm()) for k in range(2)]
    assert all(x > 0 for x in gradients) and torch.isfinite(noise.grad).all()
    backbone_gradient = sum(float(param.grad.float().square().sum()) for name, param in world.named_parameters()
                            if name.startswith('layers.') and param.grad is not None) ** .5
    assert backbone_gradient > 0
    sample_effect = float((predictions[0].detach() - predictions[1].detach())[..., M.INTERIOR, :].abs().max())
    assert sample_effect > 1e-6
    changed = frames.clone()
    changed[:, -1] = torch.randn_like(changed[:, -1])
    with torch.no_grad(), autocast_context(config):
        changed_pred = world.samples(changed, ac, noise.detach())[0]
        h, ha = world.backbone_full(frames, ac)
        base = world.complete(frames, ac, h, ha, noise[0].detach())[0]
        route_swap = world.complete(frames, ac, h, ha, noise[0].detach(), routing_noise=noise[1].detach())[0]
        gen_swap = world.complete(frames, ac, h, ha, noise[1].detach(), routing_noise=noise[0].detach())[0]
    causal_error = float((changed_pred[:, :, :15] - predictions.detach()[:, :, :15]).abs().max())
    routing_effect = float((base - route_swap)[..., M.INTERIOR, :].abs().max())
    generation_effect = float((base - gen_swap)[..., M.INTERIOR, :].abs().max())
    assert causal_error == 0 and routing_effect > 1e-6 and generation_effect > 1e-6
    assert torch.count_nonzero(M.distance(frames, frames)) == 0
    assert torch.equal(M.distance(predictions[0], predictions[1]), M.distance(predictions[1], predictions[0]))
    proof = {'matched_initial_output_maxabs': parity, 'future_mutation_current_output_maxabs': causal_error,
             'independent_draw_output_maxabs': sample_effect, 'draw_gradient_norms': gradients,
             'backbone_gradient_norm': backbone_gradient, 'routing_only_noise_effect_maxabs': routing_effect,
             'generation_only_noise_effect_maxabs': generation_effect, 'first_power_identity_symmetry_exact': True,
             'data_contract_id': manifest['contract_id'], 'precision': 'actual real-parent BF16, post-LayerNorm outputs',
             'sources': {str(Path(x).relative_to(L.ROOT)): L.sha(x) for x in (__file__, M.__file__, T.__file__)}}
    if a.resume_proof:
        resumed = torch.load(T.OUT / 'smoke/state.pt', map_location='cpu', weights_only=False)
        reference = torch.load(T.OUT / 'smoke_reference/state.pt', map_location='cpu', weights_only=False)
        assert resumed['contract_id'] == reference['contract_id'] and resumed['update'] == reference['update'] == 4
        streams = {name: L.restored_equal(resumed[name], reference[name]) for name in
                   ('order', 'noise_rng', 'torch_rng', 'cuda_rng', 'ledger_chain')}
        differences = {name: maxdiff(resumed[name], reference[name]) for name in ('worlds', 'optimizers')}
        assert all(streams.values()) and max(differences.values()) < 1e-4
        peak = max(row['peak_allocated_gib'] for row in resumed['history'] + reference['history'])
        assert peak < 6
        proof.update(resume_streams_exact=streams, resume_endpoint_maxabs=differences,
                     real_batch16_peak_allocated_gib=peak, resumed_vs_uninterrupted_updates=4)
    L.atomic(proof, T.OUT / ('validation.json' if a.resume_proof else 'initial_validation.json'), True)
    print(json.dumps(proof), flush=True)


if __name__ == '__main__':
    main()
