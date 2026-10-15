"""Train the PI world (pi_model.PIWorld; TRANSPORT_PI_PREREG.md) from the exact D parent with the TRANSPORT_IC streams.

Per update (batch 16, L16), from the live backbone:
  scroll  cross-entropy of the global scroll logits toward the parameter-free scroll estimate of the real (t, t+1) frames;
          the copy uses that estimated scroll (teacher forcing)
  chance  provisional afterstate (the gate's own decisions) -> IC posterior with the chance head's own prior -> T_hat;
          canonical soft cross-entropy on the proposal head (transport_ic_train.infer)
  edits   pre-chance target z_pre = T_hat^-1 z; q = sigmoid((|copy - z_pre| - |LN(gen) - z_pre|) / s + gate logit);
          edit = q > 0.5 or entering; gate trained by BCE toward q (entering cells excluded)
  point   the parent's D loss on LN(T_hat a), a = edit ? gen : copy
No simulator labels. Streams: windows / Gaussian / Gumbel generators seeded exactly as TRANSPORT_IC seed sets 1 / 2.
Usage: pi_train.py --seed-set 1|2 [--updates 3000] [--stop-after N] [--directory DIR]
"""
import argparse
import copy as cp
import hashlib
import json
import os
import sys
from pathlib import Path
import time

os.environ.setdefault('TRITON_F32_DEFAULT', 'ieee')
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / '20261012_oracle'))
sys.path.insert(0, str(HERE))

import torch
from torch.nn import functional as F

import audit_s5_direct as A5
import direct_model as DM
import direct_train as BASE
import loss_intervention as L
import transport_model as TM
import transport_ic_train as IC
import pi_model as PI
from d4mj.train import autocast_context, optimizer_step, phase_optimizer, _phase_lr

OUT = L.OUT / 'transport'
PREREG = HERE.parent / '20261012_oracle/TRANSPORT_PI_PREREG.md'
PARENT = L.OUT / 'paired_event/training/paired_u0000.pt'


def losses(w, frames, ac, noise, config, scroll):
    est = scroll.estimate(frames[:, :-1].float(), frames[:, 1:].float())                  # [B,15] label-free scroll target
    est16 = F.pad(est, (0, 1))
    with autocast_context(config):
        belief, ha = w.backbone_full(frames, ac)
        copy, gen, glogit, enter, sl = w.afterstate_parts(frames, ac, belief, ha, scroll=est16)
        logits = w.completion.propose(w.completion.trunk(belief, noise)).float()
    copy, gen, glogit, enter, sl, logits = (x[:, :15] for x in (copy, gen, glogit, enter, sl, logits))
    logits = logits.masked_fill(~w.completion.valid, -torch.inf)
    z = frames[:, 1:].float()
    with torch.no_grad():
        a0 = torch.where(((glogit > 0) | enter)[..., None], gen, copy)
        partner, target, nsw, _, _, _ = IC.infer(F.layer_norm(a0, (192,)), z, IC.pair_prior_logit(logits))
        z_pre = z.gather(2, partner[..., None].expand(-1, -1, -1, 192))                     # swaps are involutions
        res_copy = (F.layer_norm(copy, (192,)) - z_pre).norm(dim=-1)
        gain = res_copy - (F.layer_norm(gen, (192,)) - z_pre).norm(dim=-1)
        s = res_copy[..., :PI.MAP].mean(-1, keepdim=True).clamp_min(1e-6)
        q = torch.sigmoid(gain / s + glogit)
        edit = (q > .5) | enter
    after = torch.where(edit[..., None], gen, copy)
    out = F.layer_norm(after.gather(2, partner[..., None].expand(-1, -1, -1, 192)), (192,))
    point = L.objective(L.components(out, z), 'D')
    logp = logits.log_softmax(-1)
    ce = -torch.where(target > 0, target * logp, torch.zeros_like(logp)).sum(-1)[..., IC.MAPTILES].mean()
    gate = F.binary_cross_entropy_with_logits(glogit[~enter], q[~enter])
    possible = sl.gather(-1, est[..., None])[..., 0] > -10                 # estimator outputs impossible under the action are dropped
    scr = F.cross_entropy(sl[possible], est[possible]) if bool(possible.any()) else sl.sum() * 0
    stats = {'point': float(point.detach()), 'chance_ce': float(ce.detach()), 'gate_bce': float(gate.detach()), 'scroll_ce': float(scr.detach()),
             'posterior_swaps_per_frame': float(nsw.mean()), 'edits_per_frame_nonentering': float((edit & ~enter).float().sum(-1).mean()),
             'gate_fires_per_frame': float(((glogit > 0) & ~enter).float().sum(-1).mean()),
             'scroll_acc': float((sl.argmax(-1) == est).float().mean())}
    return point + ce + gate + scr, stats


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--seed-set', type=int, required=True, choices=(1, 2))
    ap.add_argument('--updates', type=int, default=3000)
    ap.add_argument('--stop-after', type=int)
    ap.add_argument('--directory', type=Path)
    a = ap.parse_args()
    assert PREREG.exists()
    config = BASE.settings()
    manifest, windows, actions, tokens = BASE.load_data()
    init_seed, window_seed, noise_seed = {1: (7, 20261014, 2026101401), 2: (17, 20261016, 2026101501)}[a.seed_set]
    parent = torch.load(PARENT, map_location='cpu', weights_only=False)
    w = PI.PIWorld(init_seed=init_seed)
    res = w.load_state_dict(parent['world'], strict=False)
    assert all(k.startswith(('completion.', 'gate.')) for k in res.missing_keys), res.missing_keys
    w = w.cuda().train()
    opt = phase_optimizer([w], config)
    scroll = A5.load_scroll()
    d = a.directory or OUT / f'pi_seed{a.seed_set}'; d.mkdir(parents=True, exist_ok=True)
    source = [Path(__file__), Path(PI.__file__), Path(IC.__file__), Path(TM.__file__), PREREG, PARENT]
    contract = {'format': 'transport_pi_v1', 'sources_inputs': {str(p.relative_to(L.ROOT)): L.sha(p) for p in source},
                'data_contract_id': manifest['contract_id'], 'seed_set': a.seed_set, 'updates': a.updates,
                'seeds': {'initialization': init_seed, 'windows': window_seed, 'noise': noise_seed},
                'parameters': sum(p.numel() for p in w.parameters()), 'training_semantic_or_simulator_labels': False}
    cid = hashlib.sha256(json.dumps(contract, sort_keys=True).encode()).hexdigest()
    if (d / 'contract.json').exists():
        assert json.loads((d / 'contract.json').read_text()) == contract, 'source/input drift'
    else:
        L.atomic(contract, d / 'contract.json', True)
    order = torch.Generator().manual_seed(window_seed); noise_rng = torch.Generator().manual_seed(noise_seed)
    start, history = 0, []
    if (d / 'state.pt').exists():
        st = torch.load(d / 'state.pt', map_location='cpu', weights_only=False)
        assert st['contract_id'] == cid
        w.load_state_dict(st['world']); opt.load_state_dict(st['optimizer'])
        order.set_state(st['order']); noise_rng.set_state(st['noise_rng'])
        torch.set_rng_state(st['torch_rng']); torch.cuda.set_rng_state_all(st['cuda_rng'])
        start, history = st['update'], st['history']
        print(json.dumps({'stage': 'resume', 'update': start}), flush=True)

    def save(update, snap=False):
        state = L.cpu_state(w)
        L.atomic({'contract_id': cid, 'update': update, 'world': state, 'optimizer': cp.deepcopy(opt.state_dict()),
                  'order': order.get_state(), 'noise_rng': noise_rng.get_state(), 'torch_rng': torch.get_rng_state(),
                  'cuda_rng': torch.cuda.get_rng_state_all(), 'history': history}, d / 'state.pt')
        if snap:
            L.atomic({'contract_id': cid, 'update': update, 'arm': 'pi', 'world': state}, d / f'pi_u{update:04d}.pt')
    if start == 0:
        save(0, True)
    began = time.perf_counter()
    for update in range(start, a.updates):
        idx = torch.randint(len(windows), (16,), generator=order)
        frames, ac, _ = BASE.batch(tokens, actions, windows, idx)
        noise = DM.draw_noise(frames.shape[:2], noise_rng).cuda()
        gumbel = TM.draw_gumbel(frames.shape[:2], noise_rng).cuda()
        loss, stats = losses(w, frames, ac, noise[0], config, scroll)
        norm = float(optimizer_step(opt, loss, list(w.parameters()), learning_rate=_phase_lr(config, update),
                                    grad_clip=config.agent.grad_clip, strict=True, zero_grad=True))
        if (update + 1) % 100 == 0 or update == start:
            with torch.no_grad(), autocast_context(config):
                belief, ha = w.backbone_full(frames, ac)
                _, aux = w.complete(frames, ac, belief, ha, noise[0], gumbel[0])
            stats['sampled_swaps_per_frame'] = float((aux['partner'][:, :15] != torch.arange(81, device='cuda')).sum(-1).float().mean() / 2)
            stats['own_edits_per_frame'] = float(aux['edits'][:, :15].float().sum(-1).mean())
            history.append({'update': update + 1, 'stats': {k: round(v, 5) for k, v in stats.items()}, 'gradient_norm': round(norm, 4),
                            'elapsed': round(time.perf_counter() - began, 1), 'peak_allocated_gib': round(torch.cuda.max_memory_allocated() / 2**30, 3)})
            print(json.dumps(history[-1]), flush=True)
        snap = update + 1 in (500, 1500, a.updates)
        if (update + 1) % 25 == 0 or snap:
            save(update + 1, snap)
        if a.stop_after and update + 1 >= a.stop_after:
            save(update + 1)
            return
    L.atomic({'contract_id': cid, 'update': a.updates, 'history': history, 'completed': True}, d / 'training.json', True)
    print(json.dumps({'stage': 'done', 'update': a.updates}), flush=True)


if __name__ == '__main__':
    main()
