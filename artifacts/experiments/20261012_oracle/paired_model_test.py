"""CPU mechanism invariants; synthetic observation tensors, no semantic labels.

These prove implementation properties and live optimization paths, not physical
validity. The real-parent CUDA/resume proof and semantic evaluation are separate.
"""
import json

import torch
from torch import nn
from torch.nn import functional as F

import direct_model as D
import paired_model as P


class ZeroCompletion(nn.Module):
    def forward(self, belief, noise):
        return belief.new_zeros(*belief.shape[:-1], 192), belief.new_zeros(*belief.shape[:-1], 6)


def toy_world(completion):
    """Exercise the unchanged corrt readout without constructing/scanning Mamba."""
    world = P.PairedWorld.__new__(P.PairedWorld)
    nn.Module.__init__(world)
    world.proj = nn.Linear(256, 192)
    world.choose = nn.Linear(256, 6)
    world.frame = nn.Linear(256, 1)
    world.target_gate = nn.Linear(256, 1)
    world.completion = completion
    return world


def fixed_packet(shape, source=None, edge=0):
    event = torch.randn(*shape, 256)
    gate = torch.zeros(*shape, 1)
    routes = torch.zeros(*shape, 5)
    routes[..., 0] = 1
    if source is not None:
        gate[..., source, :] = 1
        routes[..., source, :] = 0
        routes[..., source, edge] = 1
    return {'event': event, 'gate': gate, 'routes': routes}


def main():
    torch.set_num_threads(4)
    torch.manual_seed(1193)
    completion = P.PairedCompletion().eval()
    # A near-parameter-matched endpoint readout must not restrict every learned
    # edit to a fixed narrow output span. This checks expressibility, not whether
    # training ultimately uses the available span.
    write_rank = int(torch.linalg.matrix_rank(completion.write.weight.detach().float()))
    assert tuple(completion.write.weight.shape) == (192, 256) and write_rank == 192
    world = toy_world(completion).eval()
    belief = torch.randn(2, 2, 81, 256)
    frames = F.layer_norm(torch.randn(2, 2, 81, 192), (192,))
    actions = torch.tensor([[0, 1], [3, 4]])
    action_belief = torch.randn(2, 2, 256)
    noise = P.draw_noise((2, 2), torch.Generator().manual_seed(713)).requires_grad_()

    # Parent reference parity includes movement-action camera/target logits.
    reference = world.reference(frames, actions, belief, action_belief)
    control = toy_world(ZeroCompletion())
    for name in ('proj', 'choose', 'frame', 'target_gate'):
        getattr(control, name).load_state_dict(getattr(world, name).state_dict())
    parent = D.DirectWorld.complete(control, frames, actions, belief, action_belief, noise[0])[0]
    assert torch.equal(parent, F.layer_norm(reference, (192,)))

    packets = completion(belief, noise[0])
    assert torch.allclose(packets['routes'].sum(-1), torch.ones(2, 2, 81), atol=1e-7, rtol=1e-6)
    assert torch.count_nonzero(packets['routes'][..., ~completion.destination_mask]) == 0
    zero = {key: value.detach().clone() for key, value in packets.items()}
    zero['gate'].zero_()
    delta, _ = completion.edits(belief, zero)
    assert torch.count_nonzero(delta) == 0
    event_off = world.complete_from_packets(frames, actions, belief, action_belief, zero)[0]
    assert torch.equal(event_off, parent)
    self_only = {key: value.detach().clone() for key, value in packets.items()}
    self_only['routes'].zero_()
    self_only['routes'][..., 0] = 1
    delta, _ = completion.edits(belief, self_only)
    assert torch.count_nonzero(delta) == 0

    # Forward scatter agrees with an explicit coordinate implementation. In
    # particular, the right edge never wraps into the next row.
    grid = torch.arange(81 * 3).reshape(81, 3).float()
    for dr, dc in P.NEIGHBOURS:
        expected = torch.zeros_like(grid)
        for source in range(81):
            r, c = divmod(source, 9)
            if 0 <= r + dr < 9 and 0 <= c + dc < 9:
                expected[(r + dr) * 9 + c + dc] = grid[source]
        assert torch.equal(P.shift_grid(grid, dr, dc), expected)

    source = 40
    endpoint_errors = []
    for edge, (dr, dc) in enumerate(P.NEIGHBOURS, 1):
        packet = fixed_packet((2, 2, 81), source, edge)
        delta, aux = completion.edits(belief, packet, trace=True)
        recipient = source + 9 * dr + dc
        support = delta.detach().abs().sum(-1) > 0
        expected_support = torch.zeros_like(support)
        expected_support[..., source] = True
        expected_support[..., recipient] = True
        assert torch.equal(support, expected_support)
        event_read = completion.event_read(packet['event'])[..., source, :]
        context_read = completion.context_read(belief)
        removal = -completion.endpoint(event_read, context_read[..., source, :], 0, (0, 0))
        arrival = completion.endpoint(event_read, context_read[..., recipient, :], 1, (dr, dc))
        assert torch.equal(delta[..., source, :], removal)
        assert torch.equal(delta[..., recipient, :], arrival)
        assert torch.equal(aux['arrivals'][..., source, edge, :], arrival)
        assert torch.equal(aux['incoming'][..., recipient, :], arrival)
        endpoint_errors.append(float((delta[..., recipient, :] - arrival).detach().abs().max()))

    # Swap the whole post-attention packet, leaving reference and all other
    # packets fixed. Full and path-broken swaps have the expected additive edits.
    original = completion(belief, noise[0])
    other = completion(belief, noise[1])
    swapped = P.swapped_packet(original, other, source)
    for key in original:
        keep = torch.arange(81) != source
        assert torch.equal(swapped[key][..., keep, :], original[key][..., keep, :])
        assert torch.equal(swapped[key][..., source, :], other[key][..., source, :])
    base_delta, _ = completion.edits(belief, original)
    full_delta, _ = completion.edits(belief, swapped)
    removal_delta, _ = completion.edits(belief, original, removal_packets=swapped)
    arrival_delta, _ = completion.edits(belief, original, arrival_packets=swapped)
    assert torch.allclose(full_delta - base_delta,
                          (removal_delta - base_delta) + (arrival_delta - base_delta),
                          atol=1e-8, rtol=1e-5)
    changed = (full_delta - base_delta).detach().abs().sum(-1) > 1e-9
    allowed = torch.zeros_like(changed)
    allowed[..., source] = True
    for dr, dc in P.NEIGHBOURS:
        allowed[..., source + 9 * dr + dc] = True
    assert not bool((changed & ~allowed).any())

    # Both independent samples and both noise channels survive output LN.
    predictions = torch.stack([world.complete(frames, actions, belief, action_belief, n)[0]
                               for n in noise])
    targets = F.layer_norm(torch.randn_like(frames), (192,))
    energy = P.losses(predictions, targets)['energy']
    energy.backward()
    assert torch.isfinite(noise.grad).all()
    gradient_norms = [[float(noise.grad[k, ..., :4].norm()),
                       float(noise.grad[k, ..., 4:].norm())] for k in range(2)]
    assert all(value > 0 for draw in gradient_norms for value in draw)
    parameter_norms = {name: float(parameter.grad.norm()) for name, parameter in completion.named_parameters()
                       if parameter.grad is not None}
    assert all(torch.isfinite(parameter.grad).all() for parameter in completion.parameters()
               if parameter.grad is not None)
    assert parameter_norms['write.weight'] > 0 and parameter_norms['route.weight'] > 0
    sample_effect = float((predictions[0] - predictions[1]).detach().abs().max())
    assert sample_effect > 1e-6
    with torch.no_grad():
        base = predictions[0]
        only_local = noise[0].clone()
        only_local[..., :4] = noise[1, ..., :4]
        only_shared = noise[0].clone()
        only_shared[..., 4:] = noise[1, ..., 4:]
        channel_effects = [float((world.complete(frames, actions, belief, action_belief, altered)[0] - base).abs().max())
                           for altered in (only_local, only_shared)]
    assert min(channel_effects) > 1e-6
    print(json.dumps({'device': 'CPU FP32', 'batch': 2, 'frames': 2,
                      'parent_reference_parity_exact': True, 'zero_gate_identity_exact': True,
                      'self_route_identity_exact': True, 'boundary_mask_and_scatter_exact': True,
                      'endpoint_context_maxabs': endpoint_errors,
                      'single_packet_support_exact': True, 'broken_path_additivity': True,
                      'draw_local_shared_gradient_norms': gradient_norms,
                      'draw_output_effect_maxabs': sample_effect,
                      'local_shared_swap_maxabs': channel_effects,
                      'endpoint_write_shape': list(completion.write.weight.shape),
                      'initial_endpoint_write_rank': write_rank,
                      'parameters_completion': sum(p.numel() for p in completion.parameters())}), flush=True)


if __name__ == '__main__':
    main()
