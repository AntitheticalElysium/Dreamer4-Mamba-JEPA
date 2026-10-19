"""Chance-forced self-fed continuation (TRANSPORT_SF_PREREG.md). Arms ic / icsf from the IC v3 3,000-update state of one seed.

ic     the unchanged IC objective (transport_ic_train: D loss on LN(T_hat b) + soft canonical CE), teacher-forced
icsf   the same + a self-fed term: frames 0..11 real, frames 12..15 generated one at a time from the model's own context, each
       scored by the D loss and CE against the REAL frame with the chance outcome inferred by the IC posterior, then fed
       back detached. Gradients of the 4 self-fed steps are accumulated (one backbone graph at a time).
Streams continue from the saved IC state (windows, Gaussian draws). Snapshots at 3,500 / 4,500. Resumable every 25 updates.
Usage: transport_sf_train.py --seed-set 1|2 [--updates 1500] [--stop-after N] [--directory DIR]
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
from torch.nn import functional as F

import direct_model as DM
import direct_train as BASE
import loss_intervention as L
import transport_model as TM
import transport_ic_train as IC
from d4mj.train import autocast_context, optimizer_step, phase_optimizer, _phase_lr

OUT = L.OUT / 'transport'
PREREG = Path(__file__).with_name('TRANSPORT_SF_PREREG.md')
ARMS = ('ic', 'icsf')
K = 4


def ic_terms(w, frames, ac, noise, config, t0=0, t1=15):
    """IC loss on target frames t0+1..t1 given the (possibly self-fed) context `frames`; returns loss, generated frames, stats"""
    with autocast_context(config):
        belief, ha = w.backbone_full(frames, ac)
        b = w.reference(frames, ac, belief, ha).float()[:, t0:t1]
        logits = w.completion.propose(w.completion.trunk(belief, noise)).float()[:, t0:t1]
    logits = logits.masked_fill(~w.completion.valid, -torch.inf)
    z = REAL[:, t0 + 1:t1 + 1]
    with torch.no_grad():
        partner, target, nsw, s_, ncand, qmass = IC.infer(F.layer_norm(b.detach(), (192,)), z, IC.pair_prior_logit(logits.detach()))
    out = F.layer_norm(b.gather(2, partner[..., None].expand(-1, -1, -1, 192)), (192,))
    point = L.objective(L.components(out, z), 'D')
    logp = logits.log_softmax(-1)
    ce = -torch.where(target > 0, target * logp, torch.zeros_like(logp)).sum(-1)[..., IC.MAPTILES].mean()
    return point + ce, out, {'point': float(point.detach()), 'ce': float(ce.detach()), 'posterior': float(nsw.mean())}


REAL = None


def main():
    global REAL
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--seed-set', type=int, required=True, choices=(1, 2))
    ap.add_argument('--updates', type=int, default=1500)
    ap.add_argument('--stop-after', type=int)
    ap.add_argument('--directory', type=Path)
    a = ap.parse_args()
    assert PREREG.exists()
    config = BASE.settings()
    _, windows, actions, tokens = BASE.load_data()
    src = OUT / f'ic_seed{a.seed_set}'
    init = torch.load(src / 'state.pt', map_location='cpu', weights_only=False)
    assert init['update'] == 3000
    d = a.directory or OUT / f'sf_seed{a.seed_set}'; d.mkdir(parents=True, exist_ok=True)
    worlds, opts = {}, {}
    for arm in ARMS:
        w = TM.TransportWorld('operator').cuda().train(); w.load_state_dict(init['worlds']['ic'])
        o = phase_optimizer([w], config); o.load_state_dict(init['optimizers']['ic'])
        worlds[arm], opts[arm] = w, o
    source = [Path(__file__), Path(IC.__file__), Path(TM.__file__), Path(DM.__file__), PREREG, src / 'state.pt']
    contract = {'format': 'transport_sf_v1', 'sources_inputs': {str(p.relative_to(L.ROOT)): L.sha(p) for p in source},
                'seed_set': a.seed_set, 'start_update': 3000, 'updates': a.updates, 'K': K, 'arms': list(ARMS)}
    cid = hashlib.sha256(json.dumps(contract, sort_keys=True).encode()).hexdigest()
    if (d / 'contract.json').exists():
        assert json.loads((d / 'contract.json').read_text()) == contract, 'source/input drift'
    else:
        L.atomic(contract, d / 'contract.json', True)
    order = torch.Generator(); order.set_state(init['order'])
    noise_rng = torch.Generator(); noise_rng.set_state(init['noise_rng'])
    torch.set_rng_state(init['torch_rng']); torch.cuda.set_rng_state_all(init['cuda_rng'])
    start, history = 3000, []
    if (d / 'state.pt').exists():
        st = torch.load(d / 'state.pt', map_location='cpu', weights_only=False)
        assert st['contract_id'] == cid
        for arm in ARMS:
            worlds[arm].load_state_dict(st['worlds'][arm]); opts[arm].load_state_dict(st['optimizers'][arm])
        order.set_state(st['order']); noise_rng.set_state(st['noise_rng'])
        torch.set_rng_state(st['torch_rng']); torch.cuda.set_rng_state_all(st['cuda_rng'])
        start, history = st['update'], st['history']
        print(json.dumps({'stage': 'resume', 'update': start}), flush=True)

    def save(update, snap=False):
        states = {arm: L.cpu_state(w) for arm, w in worlds.items()}
        L.atomic({'contract_id': cid, 'update': update, 'worlds': states, 'optimizers': {arm: copy.deepcopy(o.state_dict()) for arm, o in opts.items()},
                  'order': order.get_state(), 'noise_rng': noise_rng.get_state(), 'torch_rng': torch.get_rng_state(),
                  'cuda_rng': torch.cuda.get_rng_state_all(), 'history': history}, d / 'state.pt')
        if snap:
            for arm in ARMS:
                L.atomic({'contract_id': cid, 'update': update, 'arm': arm, 'world': states[arm]}, d / f'{arm}_u{update:04d}.pt')
    began = time.perf_counter()
    end = 3000 + a.updates
    for update in range(start, end):
        idx = torch.randint(len(windows), (16,), generator=order)
        frames, ac, _ = BASE.batch(tokens, actions, windows, idx)
        noise = DM.draw_noise(frames.shape[:2], noise_rng).cuda()
        _ = TM.draw_gumbel(frames.shape[:2], noise_rng)                    # keep the IC stream format
        REAL = frames.float()
        cpu_rng, cuda_rng = torch.get_rng_state(), torch.cuda.get_rng_state_all()
        rec = {}
        for arm in ARMS:
            torch.set_rng_state(cpu_rng); torch.cuda.set_rng_state_all(cuda_rng)
            w, o = worlds[arm], opts[arm]
            t0 = time.perf_counter()
            o.zero_grad()
            sf_stats = []
            if arm == 'icsf':
                ctx = frames.clone()
                for j in range(K):
                    t = 16 - K + j                                           # generate frame t from context frames < t
                    loss_j, out_j, st_j = ic_terms(w, ctx, ac, noise[0], config, t - 1, t)
                    (loss_j / K).backward()
                    ctx = ctx.clone(); ctx[:, t] = out_j.detach()[:, 0]
                    sf_stats.append(st_j)
            loss, _, st = ic_terms(w, frames, ac, noise[0], config)
            norm = float(optimizer_step(o, loss, list(w.parameters()), learning_rate=_phase_lr(config, update),
                                        grad_clip=config.agent.grad_clip, strict=True, zero_grad=False))
            rec[arm] = {**{k: round(v, 5) for k, v in st.items()}, 'gradient_norm': round(norm, 4), 'seconds': round(time.perf_counter() - t0, 3)}
            if sf_stats:
                rec[arm]['selffed_point_by_step'] = [round(x['point'], 5) for x in sf_stats]
                rec[arm]['selffed_posterior_by_step'] = [round(x['posterior'], 4) for x in sf_stats]
        if (update + 1) % 100 == 0 or update == start:
            history.append({'update': update + 1, 'arms': rec, 'elapsed': round(time.perf_counter() - began, 1),
                            'peak_allocated_gib': round(torch.cuda.max_memory_allocated() / 2**30, 3)})
            print(json.dumps(history[-1]), flush=True)
        snap = update + 1 in (3500, end)
        if (update + 1) % 25 == 0 or snap:
            save(update + 1, snap)
        if a.stop_after and update + 1 >= a.stop_after:
            save(update + 1)
            return
    L.atomic({'contract_id': cid, 'update': end, 'history': history, 'completed': True}, d / 'training.json', True)
    print(json.dumps({'stage': 'done', 'update': end}), flush=True)


if __name__ == '__main__':
    main()
