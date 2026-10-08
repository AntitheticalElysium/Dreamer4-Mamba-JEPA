"""Numerical E19 mechanics checks; real parents, data, gradients and serialized resume."""
import argparse
import copy
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e19 as E
T, R = E.T, E.R


def difference(a, b):
    if torch.is_tensor(a):
        return float((a.cpu() - b.cpu()).abs().max()) if a.numel() else 0.0
    if isinstance(a, dict):
        assert a.keys() == b.keys()
        return max([difference(a[k], b[k]) for k in a] + [0.0])
    if isinstance(a, (tuple, list)):
        assert len(a) == len(b)
        return max([difference(x, y) for x, y in zip(a, b)] + [0.0])
    assert a == b, (a, b)
    return 0.0


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    p.add_argument('--batch', type=int, default=10)
    args = p.parse_args()
    torch.set_num_threads(4)
    device = torch.device(args.device)
    pool = torch.load(T.POOLS['raw'] / 'pool.pt', mmap=True, weights_only=False)
    mask = torch.load(E.MASK, weights_only=False)['beside']
    rows = E.train_rows(pool)
    backbone = 'full' if device.type == 'cpu' else 'fmamba'
    parent = T.OUT / ('corrt_raw_teacher_s7' + ('_fmamba' if backbone == 'fmamba' else '') + '_u36000.pt')
    w = E.make_world(parent, backbone, device)
    idx = torch.cat([torch.where(~pool['terminal'])[0][:args.batch // 2],
                     torch.where(pool['terminal'])[0][:args.batch - args.batch // 2]])
    b = T.S.batch_of(pool, idx, 'tokens', device)
    s, a, mm = b['s'], b['actions'], mask[idx].to(device)
    offsets = (torch.arange(args.batch) % 5).to(device)
    out = {'device': args.device, 'backbone': backbone, 'batch': args.batch,
           'parent_sha256': R.file_hash(parent), 'source': E.numeric_sources()}
    for r in range(5):
        pairs = [(i, i + 1, i) for lo, hi in E.segments(r) for i in range(lo, hi)]
        assert pairs == [(i, i + 1, i) for i in range(5)]
    out['all_offsets_exact_action_target_mask_pairs'] = True
    from d4mj.train import autocast_context, phase_optimizer, optimizer_step, _phase_lr
    from contextlib import nullcontext
    cfg = E.config()
    ctx = lambda: autocast_context(cfg) if device.type == 'cuda' else nullcontext()
    w.train()
    with ctx():
        legacy = T.rollout_losses(w, s, a, 'teacher')
    legacy.backward()
    grad = {k: q.grad.detach().cpu().clone() for k, q in w.named_parameters() if q.grad is not None}
    w.zero_grad(set_to_none=True)
    with ctx():
        control, _ = E.objective(w, s, a, mm, offsets, 'A')
    control.backward()
    out['A_loss_difference'] = float(abs(legacy.detach() - control.detach()))
    out['A_gradient_max_abs'] = max(float((q.grad.cpu() - grad[k]).abs().max()) for k, q in w.named_parameters() if k in grad)
    assert out['A_loss_difference'] == 0 and out['A_gradient_max_abs'] <= 1e-6
    w.zero_grad(set_to_none=True)
    with ctx():
        value, _ = E.objective(w, s, a, mm, offsets, 'B')
    value.backward()
    assert all(torch.isfinite(q.grad).all() for q in w.parameters() if q.grad is not None)
    out['B_finite_loss_and_gradients'] = bool(torch.isfinite(value))
    w.zero_grad(set_to_none=True)
    with ctx():
        bc, bp = E.objective(w, s, a, mm, offsets, 'B')
        cc, cp = E.objective(w, s, a, mm, offsets, 'C')
    out['C_minus_B_minus_health_abs'] = abs(float(cc - bc) - cp['health'])
    out['B_C_teacher_difference'] = abs(bp['teacher'] - cp['teacher'])
    assert out['B_C_teacher_difference'] <= 1e-6
    assert out['C_minus_B_minus_health_abs'] <= 1e-6
    del legacy, control, value, bc, cc, grad
    w.zero_grad(set_to_none=True)
    # Causality: changing future frames/actions cannot alter a prior target.
    with torch.no_grad():
        aa = F.pad(a, (0, 1))
        pad = s.clone(); pad[:, 3:] += 5
        ap = aa.clone(); ap[:, 3:] = (ap[:, 3:] + 7) % 17
        x = w(s, aa)[0][:, :3]
        y = w(pad, ap)[0][:, :3]
        out['future_perturbation_max_abs'] = float((x - y).abs().max())
    assert out['future_perturbation_max_abs'] <= 1e-5
    del w, x, y
    if device.type == 'cuda':
        torch.cuda.empty_cache()
        resume = {}
        for arm in ('A', 'B', 'C'):
            def run(n, state=None):
                torch.manual_seed(7); torch.cuda.manual_seed_all(7)
                model = E.make_world(parent, backbone, device).train()
                opt = phase_optimizer([model], cfg)
                params = [q for group in opt.param_groups for q in group['params']]
                order = torch.Generator().manual_seed(11)
                boundaries = torch.Generator().manual_seed(19)
                start = 0
                if state is not None:
                    model.load_state_dict(state['world']); opt.load_state_dict(state['optimizer'])
                    order.set_state(state['order']); boundaries.set_state(state['boundaries'])
                    R.restore_rng(state['rng'], device); start = state['update']
                for u in range(start, n):
                    ii = rows[torch.randint(len(rows), (args.batch,), generator=order)]
                    off = torch.randint(5, (args.batch,), generator=boundaries).to(device)
                    bb = T.S.batch_of(pool, ii, 'tokens', device)
                    with ctx():
                        v, _ = E.objective(model, bb['s'], bb['actions'], mask[ii].to(device), off, arm)
                    optimizer_step(opt, v, params, learning_rate=_phase_lr(cfg, u),
                                   grad_clip=cfg.agent.grad_clip, strict=True, zero_grad=True)
                # Actual disk serialization, not a shallow alias of optimizer tensors.
                path = T.OUT / 'state' / f'e19_verify_{arm}_{n}.pt'
                path.parent.mkdir(exist_ok=True)
                R.atomic_torch(path, {'world': model.state_dict(), 'optimizer': opt.state_dict(),
                        'order': order.get_state(), 'boundaries': boundaries.get_state(), 'rng': R.rng_state(device), 'update': n})
                result = torch.load(path, map_location='cpu', weights_only=False)
                del model, opt, params
                torch.cuda.empty_cache()
                return result
            full = run(4)
            split = run(4, run(2))
            metrics = {'parameter_max_abs': difference(full['world'], split['world']),
                       'optimizer_max_abs': difference(full['optimizer'], split['optimizer']),
                       'order_equal': torch.equal(full['order'], split['order']),
                       'boundaries_equal': torch.equal(full['boundaries'], split['boundaries']),
                       'cpu_rng_equal': torch.equal(full['rng']['cpu'], split['rng']['cpu']),
                       'cuda_rng_equal': torch.equal(full['rng']['cuda'], split['rng']['cuda'])}
            assert metrics['parameter_max_abs'] <= 1e-6 and metrics['optimizer_max_abs'] <= 1e-6
            assert all(metrics[k] for k in ('order_equal', 'boundaries_equal', 'cpu_rng_equal', 'cuda_rng_equal'))
            resume[arm] = metrics
        out['four_updates_vs_two_plus_two'] = resume
        out['peak_allocated_gb'] = torch.cuda.max_memory_allocated() / 1e9
    # Storage guards: changed recipe and corrupted committed payload must be rejected.
    store = R.Store(T.OUT / 'state' / ('e19_verify_store_' + args.device), {'version': 1, 'device': args.device})
    store.save('proof', {'value': 1}, 1)
    assert store.load('proof')['value'] == 1
    try:
        R.Store(store.root, {'version': 2, 'device': args.device})
    except RuntimeError:
        out['changed_contract_rejected'] = True
    else:
        raise AssertionError('Changed recipe accepted')
    with (store.root / 'proof-1.pt').open('ab') as f:
        f.write(b'corruption-test')
    try:
        store.load('proof')
    except RuntimeError:
        out['corrupt_payload_rejected'] = True
    else:
        raise AssertionError('Corrupt payload accepted')
    R.atomic_json(T.HERE / f'e19_verify_{args.device}.json', out)
    print(json.dumps({k: v for k, v in out.items() if k != 'source'}, indent=2))


if __name__ == '__main__':
    main()
