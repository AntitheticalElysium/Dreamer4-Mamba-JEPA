"""Stage 3 matched training: free synthesis vs joint reassignment (transport_model). TRANSPORT_PREREG.md governs.

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

import torch

import direct_model as DM
import direct_train as BASE
import loss_intervention as L
import transport_model as TM
from d4mj.train import autocast_context, optimizer_step, phase_optimizer, _phase_lr

OUT = L.OUT / 'transport'
PREREG = Path(__file__).with_name('TRANSPORT_PREREG.md')
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
    a = ap.parse_args()
    assert PREREG.exists()
    config = BASE.settings()
    manifest, windows, actions, tokens = BASE.load_data()
    parent = torch.load(PARENT, map_location='cpu', weights_only=False)
    assert parent['update'] == 0 and parent['arm'] == 'paired'
    worlds, shared = TM.initialized_worlds(parent['world'], 'cuda')
    optimizers = {arm: phase_optimizer([w], config) for arm, w in worlds.items()}
    gs = {arm: groups(worlds[arm], optimizers[arm]) for arm in TM.ARMS}
    shared_params = {n for n, _ in worlds['free'].named_parameters()} & {n for n, _ in worlds['operator'].named_parameters()}
    for wd in (0.01, 0.0):
        f = {n for g in gs['free'] if g['weight_decay'] == wd for n in g['names']} & shared_params
        o = {n for g in gs['operator'] if g['weight_decay'] == wd for n in g['names']} & shared_params
        assert f == o, f'decay-group mismatch on shared parameters at wd={wd}: {sorted(f ^ o)[:5]}'
    source = [Path(__file__), Path(TM.__file__), Path(DM.__file__), Path(BASE.__file__), Path(L.__file__), Path(L.TW.__file__),
              L.ROOT / 'd4mj/train.py', L.ROOT / 'd4mj/mamba_recurrence.py', PREREG, PARENT, L.TW.S.CHECKPOINT,
              BASE.D.OUT / 'manifest.json', BASE.D.OUT / 'windows.pt']
    contract = {'format': 'transport_two_arm_v1', 'sources_inputs': {str(p.relative_to(L.ROOT)): L.sha(p) for p in source},
                'data_contract_id': manifest['contract_id'], 'updates': a.updates, 'batch': a.batch, 'arms': list(TM.ARMS),
                'parameters': {arm: {'total': sum(p.numel() for p in w.parameters()),
                                     'completion': sum(p.numel() for p in w.completion.parameters())} for arm, w in worlds.items()},
                'decay_groups': {arm: [{'weight_decay': g['weight_decay'], 'count': len(g['names'])} for g in gs[arm]] for arm in TM.ARMS},
                'seeds': {'initialization': 7, 'windows': 20261014, 'noise': 2026101401}, 'beta': TM.BETA,
                'loss': 'direct_model.losses energy; 35 scored tokens', 'training_semantic_or_simulator_labels': False}
    contract = json.loads(json.dumps(contract))
    cid = hashlib.sha256(json.dumps(contract, sort_keys=True).encode()).hexdigest()
    d = a.directory; d.mkdir(parents=True, exist_ok=True)
    if (d / 'contract.json').exists():
        assert json.loads((d / 'contract.json').read_text()) == contract, 'source/input drift'
    else:
        L.atomic(contract, d / 'contract.json', True)
    order = torch.Generator().manual_seed(20261014)
    noise_rng = torch.Generator().manual_seed(2026101401)
    start, chain, history = 0, 'initial', []
    state = d / 'state.pt'
    if state.exists():
        saved = torch.load(state, map_location='cpu', weights_only=False)
        assert saved['contract_id'] == cid
        for arm in TM.ARMS:
            worlds[arm].load_state_dict(saved['worlds'][arm]); optimizers[arm].load_state_dict(saved['optimizers'][arm])
        order.set_state(saved['order']); noise_rng.set_state(saved['noise_rng'])
        torch.set_rng_state(saved['torch_rng']); torch.cuda.set_rng_state_all(saved['cuda_rng'])
        start, chain, history = saved['update'], saved['ledger_chain'], saved['history']
        print(json.dumps({'stage': 'resume', 'update': start}), flush=True)

    def save(update, snapshot=False):
        states = {arm: L.cpu_state(w) for arm, w in worlds.items()}
        L.atomic({'contract_id': cid, 'update': update, 'worlds': states,
                  'optimizers': {arm: copy.deepcopy(o.state_dict()) for arm, o in optimizers.items()},
                  'order': order.get_state(), 'noise_rng': noise_rng.get_state(), 'torch_rng': torch.get_rng_state(),
                  'cuda_rng': torch.cuda.get_rng_state_all(), 'ledger_chain': chain, 'history': history}, state)
        if snapshot:
            for arm in TM.ARMS:
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
        for arm in TM.ARMS:
            torch.set_rng_state(cpu_rng); torch.cuda.set_rng_state_all(cuda_rng)
            w = worlds[arm]
            torch.cuda.synchronize(); t0 = time.perf_counter()
            with autocast_context(config):
                belief, ha = w.backbone_full(frames, ac)
                outs, moves = [], []
                for k in range(2):
                    o, aux = w.complete(frames, ac, belief, ha, noise[k], gumbel[k])
                    outs.append(o[:, :15])
                    if 'partner' in aux:
                        moves.append(float((aux['partner'][:, :15] != torch.arange(81, device='cuda')).sum(-1).float().mean() / 2))
            values = DM.losses(torch.stack(outs), frames[:, 1:])
            norm = float(optimizer_step(optimizers[arm], values['energy'], list(w.parameters()), learning_rate=_phase_lr(config, update),
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
