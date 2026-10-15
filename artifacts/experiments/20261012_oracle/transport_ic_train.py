"""Inferred-chance (IC) vs energy-score (ES) training of the identical reassignment operator. TRANSPORT_IC_PREREG.md governs.

Copied from transport_train.py (unchanged infrastructure: parent, construction, decay-group check, streams, resume). Arms:
  operator  the ES operator exactly as in Stage 3 (in-process replicate, the matched baseline)
  ic        the same model. Per update, from the current deterministic afterstate a = LN(b) (detached) and the observed
            successor z, a label-free posterior over adjacent interior swaps:
              gain(j,k) = |a_j - z_j| + |a_k - z_k| - |a_k - z_j| - |a_j - z_k|          (first-power distances)
              candidates: parallel greedy one-to-one matching over pairs with gain > 0 (the sampler's matching rule)
              v2 (TRANSPORT_IC_PREREG addendum): Laplace likelihood with scale s = the frame's mean interior residual |a - z|
              and prior pi = running label-free posterior mass per pair:
                T_hat = candidates with gain / s > log((1 - pi) / pi)     (MAP: a swap must pay its description cost)
                q(j,k) = sigmoid(gain / s + logit(pi)) on candidates     (Bayesian marginal; v1 had the sign of logit flipped)
            point loss: the parent's D loss on LN(T_hat b) vs z over all 81 tokens (frames 1..15)
            chance loss: cross-entropy of the proposal head (masked as in sampling) toward canonical soft targets: pair
            (j, j+1) labelled only at j as RIGHT, pair (j, j+9) only at j as DOWN (one endpoint per unordered pair keeps
            the sampler calibrated); weight 1; averaged over map tiles and frames.

Both arms: the exact D parent (paired_u0000 weights), fresh model construction + load_state_dict (keeps upstream Mamba
no-decay flags), phase AdamW groups verified identical by parameter name on every shared tensor, the training ledger's
uniform eligible windows (seed 20261014), two independent noise draws per update (Gaussian seed 2026101401, then Gumbel from
the same generator: identical streams in both arms), energy = .5d(X1,Y)+.5d(X2,Y)-.5d(X1,X2) on the 35 scored tokens
(direct_model.losses), batch 16, L16, 3,000 updates, phase LR schedule, grad clip 1. Exact resumable state every 25 updates.
No simulator labels, layouts, readers or evaluation data enter optimization.
"""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import time

os.environ.setdefault('TRITON_F32_DEFAULT', 'ieee')

import numpy as np
import torch
from torch.nn import functional as F

import direct_model as DM
import direct_train as BASE
import loss_intervention as L
import transport_model as TM
from d4mj.train import autocast_context, optimizer_step, phase_optimizer, _phase_lr

OUT = L.OUT / 'transport'
PREREG = Path(__file__).with_name('TRANSPORT_IC_PREREG.md')
COST_PREREG = Path(__file__).with_name('TRANSPORT_COST_PREREG.md')
INTERIOR = [r * 9 + c for r in range(1, 6) for c in range(1, 8)]
MAPTILES = [x for x in range(63) if x != 31]
PAIRS = [(x, x + d) for x in INTERIOR for d in (1, 9) if x + d in INTERIOR and 31 not in (x, x + d)]


def infer(a, z, pi):
    """a, z [B,T,81,192] -> partner [B,T,81] (ML matching), soft canonical targets [B,T,81,5], accepted swaps [B,T], s [B,T]"""
    dev = a.device
    j = torch.tensor([p[0] for p in PAIRS], device=dev); k = torch.tensor([p[1] for p in PAIRS], device=dev)
    res = (a - z).norm(dim=-1)                                               # [B,T,81]
    cross = (a[..., k, :] - z[..., j, :]).norm(dim=-1) + (a[..., j, :] - z[..., k, :]).norm(dim=-1)
    gain = res[..., j] + res[..., k] - cross                                 # [B,T,P]
    s = res[..., INTERIOR].mean(-1)                                          # [B,T]
    shape = gain.shape[:-1]
    g = gain.reshape(-1, len(PAIRS)); m = len(g)
    live = g > 0
    free = torch.ones(m, 81, dtype=torch.bool, device=dev)
    won = torch.zeros_like(live)
    neg = torch.finfo(g.dtype).min
    src, dst = j.expand(m, -1), k.expand(m, -1)
    for _ in range(32):
        live = live & free.gather(1, src) & free.gather(1, dst)
        if not bool(live.any()):
            break
        sc = torch.where(live, g, torch.full_like(g, neg))
        best = torch.full((m, 81), neg, device=dev, dtype=g.dtype)
        best = best.scatter_reduce(1, src, sc, 'amax').scatter_reduce(1, dst, sc, 'amax')
        win = live & (sc >= best.gather(1, src)) & (sc >= best.gather(1, dst))
        rows, ps = torch.where(win)
        free[rows, j[ps]] = False; free[rows, k[ps]] = False
        won |= win
        live = live & ~win
    logit_pi = float(np.log(pi / (1 - pi)))
    lr = g / s.reshape(-1, 1)                                                # log-likelihood ratio of swap vs keep
    accepted = won & (lr + logit_pi > 0)                                     # MAP: posterior odds > 1
    partner = torch.arange(81, device=dev).repeat(m, 1)
    rows, ps = torch.where(accepted)
    partner[rows, j[ps]] = k[ps]; partner[rows, k[ps]] = j[ps]
    q = torch.sigmoid(lr + logit_pi) * won.float()
    right = ((k - j) == 1).float()
    target = torch.zeros(m, 81, 5, device=dev)
    target[:, :, 4] = target[:, :, 4].index_add(1, j, q * right)
    target[:, :, 2] = target[:, :, 2].index_add(1, j, q * (1 - right))
    target[:, :, 0] = (1 - target[:, :, 2] - target[:, :, 4]).clamp_min(0)
    return (partner.view(*shape, 81), target.view(*shape, 81, 5), accepted.view(*shape, -1).sum(-1).float(), s,
            won.view(*shape, -1).sum(-1).float(), q.view(*shape, -1).sum(-1))
PARENT = L.OUT / 'paired_event/training/paired_u0000.pt'


def groups(world, optimizer):
    names = {id(p): n for n, p in world.named_parameters()}
    return [{'weight_decay': g['weight_decay'], 'names': sorted(names[id(p)] for p in g['params'])} for g in optimizer.param_groups]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--updates', type=int, default=3000)
    ap.add_argument('--batch', type=int, default=16)
    ap.add_argument('--stop-after', type=int)
    ap.add_argument('--directory', type=Path, default=OUT / 'training')
    ap.add_argument('--arms', default='operator,ic')
    ap.add_argument('--seed-set', type=int, default=1, choices=(1, 2))
    ap.add_argument('--cost', type=float, default=0.0, help='transition cost weight for the operator_cost arm')
    a = ap.parse_args()
    arms = tuple(a.arms.split(','))
    assert all(x in ('operator', 'ic') for x in arms) and a.cost == 0
    init_seed, window_seed, noise_seed = {1: (7, 20261014, 2026101401), 2: (17, 20261016, 2026101501)}[a.seed_set]
    assert PREREG.exists()
    config = BASE.settings()
    manifest, windows, actions, tokens = BASE.load_data()
    parent = torch.load(PARENT, map_location='cpu', weights_only=False)
    assert parent['update'] == 0 and parent['arm'] == 'paired'
    worlds, shared = TM.initialized_worlds(parent['world'], 'cuda', arms, init_seed)
    optimizers = {arm: phase_optimizer([w], config) for arm, w in worlds.items()}
    gs = {arm: groups(worlds[arm], optimizers[arm]) for arm in arms}
    for other in arms[1:]:
        shared_params = {n for n, _ in worlds[arms[0]].named_parameters()} & {n for n, _ in worlds[other].named_parameters()}
        for wd in (0.01, 0.0):
            f = {n for g in gs[arms[0]] if g['weight_decay'] == wd for n in g['names']} & shared_params
            o = {n for g in gs[other] if g['weight_decay'] == wd for n in g['names']} & shared_params
            assert f == o, f'decay-group mismatch on shared parameters at wd={wd}: {sorted(f ^ o)[:5]}'
    source = [Path(__file__), Path(TM.__file__), Path(DM.__file__), Path(BASE.__file__), Path(L.__file__), Path(L.TW.__file__),
              L.ROOT / 'd4mj/train.py', L.ROOT / 'd4mj/mamba_recurrence.py', PREREG, PARENT, L.TW.S.CHECKPOINT,
              *([COST_PREREG] if COST_PREREG.exists() else []),
              BASE.D.OUT / 'manifest.json', BASE.D.OUT / 'windows.pt']
    contract = {'format': 'transport_ic_v2', 'sources_inputs': {str(p.relative_to(L.ROOT)): L.sha(p) for p in source},
                'data_contract_id': manifest['contract_id'], 'updates': a.updates, 'batch': a.batch, 'arms': list(arms),
                'parameters': {arm: {'total': sum(p.numel() for p in w.parameters()),
                                     'completion': sum(p.numel() for p in w.completion.parameters())} for arm, w in worlds.items()},
                'decay_groups': {arm: [{'weight_decay': g['weight_decay'], 'count': len(g['names'])} for g in gs[arm]] for arm in arms},
                'seeds': {'initialization': init_seed, 'windows': window_seed, 'noise': noise_seed}, 'beta': TM.BETA,
                **({'seed_set': a.seed_set, 'transition_cost': a.cost} if (a.seed_set != 1 or a.cost or arms != TM.ARMS) else {}),
                'loss': {'operator': 'direct_model.losses energy; 35 scored tokens',
                         'ic': 'D loss on LN(T_hat b) over 81 tokens + soft canonical CE on proposals (infer())'},
                'training_semantic_or_simulator_labels': False}
    contract = json.loads(json.dumps(contract))
    cid = hashlib.sha256(json.dumps(contract, sort_keys=True).encode()).hexdigest()
    d = a.directory; d.mkdir(parents=True, exist_ok=True)
    if (d / 'contract.json').exists():
        assert json.loads((d / 'contract.json').read_text()) == contract, 'source/input drift'
    else:
        L.atomic(contract, d / 'contract.json', True)
    order = torch.Generator().manual_seed(window_seed)
    noise_rng = torch.Generator().manual_seed(noise_seed)
    start, chain, history = 0, 'initial', []
    pi = None                                                              # running label-free swap rate per pair (IC prior)
    state = d / 'state.pt'
    if state.exists():
        saved = torch.load(state, map_location='cpu', weights_only=False)
        assert saved['contract_id'] == cid
        for arm in arms:
            worlds[arm].load_state_dict(saved['worlds'][arm]); optimizers[arm].load_state_dict(saved['optimizers'][arm])
        order.set_state(saved['order']); noise_rng.set_state(saved['noise_rng'])
        torch.set_rng_state(saved['torch_rng']); torch.cuda.set_rng_state_all(saved['cuda_rng'])
        start, chain, history, pi = saved['update'], saved['ledger_chain'], saved['history'], saved['pi']
        print(json.dumps({'stage': 'resume', 'update': start}), flush=True)

    def save(update, snapshot=False):
        states = {arm: L.cpu_state(w) for arm, w in worlds.items()}
        L.atomic({'contract_id': cid, 'update': update, 'worlds': states,
                  'optimizers': {arm: copy.deepcopy(o.state_dict()) for arm, o in optimizers.items()},
                  'order': order.get_state(), 'noise_rng': noise_rng.get_state(), 'torch_rng': torch.get_rng_state(),
                  'cuda_rng': torch.cuda.get_rng_state_all(), 'ledger_chain': chain, 'history': history, 'pi': pi}, state)
        if snapshot:
            for arm in arms:
                L.atomic({'contract_id': cid, 'update': update, 'arm': arm, 'world': states[arm], 'ledger_chain': chain},
                         d / f'{arm}_u{update:04d}.pt')
    if start == 0:
        save(0, snapshot=True)
    began = time.perf_counter()
    for update in range(start, a.updates):
        idx = torch.randint(len(windows), (a.batch,), generator=order)
        frames, ac, choices = BASE.batch(tokens, actions, windows, idx)
        noise = DM.draw_noise(frames.shape[:2], noise_rng).cuda()
        gumbel = TM.draw_gumbel(frames.shape[:2], noise_rng).cuda()
        chain = hashlib.sha256((b'' if chain == 'initial' else bytes.fromhex(chain)) + choices.numpy().tobytes()
                               + noise.cpu().numpy().tobytes() + gumbel.cpu().numpy().tobytes()).hexdigest()
        cpu_rng, cuda_rng = torch.get_rng_state(), torch.cuda.get_rng_state_all()
        records = {}
        for arm in arms:
            torch.set_rng_state(cpu_rng); torch.cuda.set_rng_state_all(cuda_rng)
            w = worlds[arm]
            torch.cuda.synchronize(); t0 = time.perf_counter()
            if arm == 'ic':
                z = frames[:, 1:].float()
                with autocast_context(config):
                    belief, ha = w.backbone_full(frames, ac)
                    b = w.reference(frames, ac, belief, ha).float()[:, :15]
                    logits = w.completion.propose(w.completion.trunk(belief, noise[0])).float()[:, :15]
                logits = logits.masked_fill(~w.completion.valid, -torch.inf)
                with torch.no_grad():
                    if pi is None:
                        pi = 0.01                                               # ~0.5 swaps per frame over 54 pairs
                    partner, target, nsw, s_, ncand, qmass = infer(F.layer_norm(b.detach(), (192,)), z, pi)
                    pi = min(max(0.99 * pi + 0.01 * float(qmass.mean()) / len(PAIRS), 1e-4), 0.5)
                out = F.layer_norm(b.gather(2, partner[..., None].expand(-1, -1, -1, 192)), (192,))
                point = L.objective(L.components(out, z), 'D')
                logp = logits.log_softmax(-1)
                ce = -torch.where(target > 0, target * logp, torch.zeros_like(logp)).sum(-1)[..., MAPTILES].mean()
                objective = point + ce
                norm = float(optimizer_step(optimizers[arm], objective, list(w.parameters()), learning_rate=_phase_lr(config, update),
                                            grad_clip=config.agent.grad_clip, strict=True, zero_grad=True))
                torch.cuda.synchronize()
                records[arm] = {'point': round(float(point.detach()), 6), 'ce': round(float(ce.detach()), 6),
                                'posterior_swaps_per_frame': round(float(nsw.mean()), 4), 'candidates_per_frame': round(float(ncand.mean()), 4),
                                'q_mass_per_frame': round(float(qmass.mean()), 4), 'scale_s': round(float(s_.mean()), 4),
                                'pi': round(pi, 6), 'gradient_norm': round(norm, 4), 'seconds': round(time.perf_counter() - t0, 4)}
                if (update + 1) % 100 == 0 or update == start:
                    with torch.no_grad(), autocast_context(config):
                        belief, ha = w.backbone_full(frames, ac)
                        _, aux = w.complete(frames, ac, belief, ha, noise[0], gumbel[0])
                    records[arm]['swaps_per_frame'] = round(float((aux['partner'][:, :15] != torch.arange(81, device='cuda')).sum(-1).float().mean() / 2), 4)
                continue
            with autocast_context(config):
                belief, ha = w.backbone_full(frames, ac)
                outs, moves, logits = [], [], []
                for k in range(2):
                    o, aux = w.complete(frames, ac, belief, ha, noise[k], gumbel[k])
                    outs.append(o[:, :15])
                    if 'logits' in aux:
                        logits.append(aux['logits'])
                    if 'partner' in aux:
                        moves.append(float((aux['partner'][:, :15] != torch.arange(81, device='cuda')).sum(-1).float().mean() / 2))
            values = DM.losses(torch.stack(outs), frames[:, 1:])
            objective = values['energy']
            if arm == TM.COST_ARM:
                values['transition_cost'] = torch.stack([TM.expected_changes(lg[:, :15]) for lg in logits]).mean()
                objective = objective + a.cost * values['transition_cost']
            norm = float(optimizer_step(optimizers[arm], objective, list(w.parameters()), learning_rate=_phase_lr(config, update),
                                        grad_clip=config.agent.grad_clip, strict=True, zero_grad=True))
            torch.cuda.synchronize()
            records[arm] = {k: round(float(v.detach()), 6) for k, v in values.items()}
            records[arm].update(gradient_norm=round(norm, 4), seconds=round(time.perf_counter() - t0, 4))
            if moves:
                records[arm]['swaps_per_frame'] = round(sum(moves) / len(moves), 4)
        if (update + 1) % 100 == 0 or update == start:
            history.append({'update': update + 1, 'arms': records, 'ledger_chain': chain, 'elapsed': round(time.perf_counter() - began, 1),
                            'peak_allocated_gib': round(torch.cuda.max_memory_allocated() / 2**30, 3)})
            print(json.dumps(history[-1]), flush=True)
        snap = update + 1 in (500, 1500, a.updates)
        if (update + 1) % 25 == 0 or snap:
            save(update + 1, snapshot=snap)
        if a.stop_after and update + 1 >= a.stop_after:
            save(update + 1)
            return
    L.atomic({'contract_id': cid, 'update': a.updates, 'history': history, 'ledger_chain': chain, 'completed': True},
             d / 'training.json', True)
    print(json.dumps({'stage': 'done', 'update': a.updates}), flush=True)


if __name__ == '__main__':
    main()
