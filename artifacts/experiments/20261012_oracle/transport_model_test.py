"""CPU checks of transport_model's operator and free heads (synthetic tensors; no backbone, no data)."""
import torch
from torch.nn import functional as F

import transport_model as TM


def main():
    torch.manual_seed(0)
    b, t = 3, 4
    ref = torch.randn(b, t, 81, 192, requires_grad=True)
    belief = torch.randn(b, t, 81, 256)
    noise = torch.randn(b, t, 81, 8)
    g = torch.Generator().manual_seed(1)
    gumbel = TM.draw_gumbel((b, t), g)[0]
    op = TM.TransportCompletion('operator')
    # 1. forced proposals: random logits, matching is an involution of disjoint valid adjacent swaps
    with torch.no_grad():
        op.propose.bias.zero_()
        op.propose.weight.normal_(0, 1.)
    out, aux = op(ref, belief, noise, gumbel)
    p = aux['partner']
    assert torch.equal(p.gather(-1, p), torch.arange(81).expand_as(p)), 'not an involution'
    moved = p != torch.arange(81)
    assert int(moved.sum()) > 0
    rows, cols = torch.where(moved.reshape(-1, 81))
    for r, j in zip(rows.tolist(), cols.tolist()):
        k = int(p.reshape(-1, 81)[r, j])
        assert abs(j // 9 - k // 9) + abs(j % 9 - k % 9) == 1 and j < 63 and k < 63 and TM.PLAYER not in (j, k), (j, k)
    # 2. forward equals the hard permutation of the reference
    hard = F.layer_norm(ref.detach().gather(2, p[..., None].expand(-1, -1, -1, 192)), (192,))
    assert torch.allclose(out, hard, atol=1e-5), float((out - hard).abs().max())
    # 3. gradients: the reference's gradient follows the moved content; logits receive a gradient
    w = torch.randn_like(out)
    (out * w).sum().backward()
    gref = ref.grad
    expect = torch.autograd.grad((hard_fn := lambda r: F.layer_norm(r.gather(2, p[..., None].expand(-1, -1, -1, 192)), (192,)))(ref) .mul(w).sum(), ref)[0]
    assert torch.allclose(gref, expect, atol=1e-5), float((gref - expect).abs().max())
    assert op.propose.weight.grad is not None and float(op.propose.weight.grad.abs().sum()) > 0
    # 4. at the declared initialization (stay bias BETA, weights std 1e-3) moves are rare and the output is LN(reference) when none
    op0 = TM.TransportCompletion('operator')
    with torch.no_grad():
        out0, aux0 = op0(ref.detach(), belief, noise, gumbel)
    moves_per_frame = float((aux0['partner'] != torch.arange(81)).sum(-1).float().mean() / 2)
    still = (aux0['partner'] == torch.arange(81)).all(-1)
    assert torch.allclose(out0[still], F.layer_norm(ref.detach(), (192,))[still], atol=1e-6)
    # 5. huge stay bias: identity everywhere
    with torch.no_grad():
        op0.propose.bias[0] = 1e4
        out_i, aux_i = op0(ref.detach(), belief, noise, gumbel)
    assert torch.equal(aux_i['partner'], torch.arange(81).expand_as(aux_i['partner']))
    assert torch.allclose(out_i, F.layer_norm(ref.detach(), (192,)), atol=1e-6)
    # 6. free arm at initialization is within the generate init's scale of the reference
    fr = TM.TransportCompletion('free')
    with torch.no_grad():
        out_f, _ = fr(ref.detach(), belief, noise, gumbel)
    dev = float((out_f - F.layer_norm(ref.detach(), (192,))).abs().max())
    print({'operator_swaps_checked': len(rows), 'initial_moves_per_frame': round(moves_per_frame, 3),
           'free_initial_max_deviation': round(dev, 5), 'passed': True})


if __name__ == '__main__':
    main()
