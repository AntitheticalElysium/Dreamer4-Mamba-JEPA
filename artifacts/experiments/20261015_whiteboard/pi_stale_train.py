"""Stale-context augmentation of PI's backbone input (PI_STALE_PREREG.md governs; 2026-10-15 night).

Both arms in one process, identical batches, from PI u3000 (weights + optimizer state of pi_seed<s>/state.pt), updates
3,000 -> 4,000 with the PI objective (pi_train.losses, reproduced here with one input switch):
  piplain   the backbone reads the fresh window (PI as is)
  pistale   per window m ~ U{0..14}: context frames 15-m..14 are replaced, as BACKBONE INPUT only, by the idealized stale
            reconstruction of stale_context.py built from the arm's own current label-free inference on the fresh window
            (scroll estimate, IC posterior, edit posterior; real tokens at changed cells, exact aged copies elsewhere; no
            gradient). Copies, posterior, targets and losses use the fresh frames.
Usage: pi_stale_train.py --seed-set 1|2 [--stop-after N] -> transport/pi_seed<s>/{piplain,pistale}_u4000.pt
"""
import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

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
import transport_ic_train as IC
import transport_model as TM
import pi_model as PI
from d4mj.train import autocast_context, optimizer_step, phase_optimizer, _phase_lr

OUT = L.OUT / 'transport'
PREREG = HERE / 'PI_STALE_PREREG.md'
START, END = 3000, 4000


@torch.no_grad()
def stale_inputs(w, frames, ac, noise, config, scroll, m):
    """idealized stale reconstruction of context frames 15-m..14 (m [B]) from the model's own inference on the fresh window"""
    est = scroll.estimate(frames[:, :-1].float(), frames[:, 1:].float())
    with autocast_context(config):
        belief, ha = w.backbone_full(frames, ac)
        copy, gen, glogit, enter, _ = w.afterstate_parts(frames, ac, belief, ha, scroll=F.pad(est, (0, 1)))
        logits = w.completion.propose(w.completion.trunk(belief, noise)).float()
    copy, gen, glogit, enter, logits = (x[:, :15].float() for x in (copy, gen, glogit, enter, logits))
    enter = enter.bool()
    logits = logits.masked_fill(~w.completion.valid, -torch.inf)
    z = frames[:, 1:].float()
    a0 = torch.where(((glogit > 0) | enter)[..., None], gen, copy)
    partner, _, _, _, _, _ = IC.infer(F.layer_norm(a0, (192,)), z, IC.pair_prior_logit(logits))
    z_pre = z.gather(2, partner[..., None].expand(-1, -1, -1, 192))
    res_copy = (F.layer_norm(copy, (192,)) - z_pre).norm(dim=-1)
    gain = res_copy - (F.layer_norm(gen, (192,)) - z_pre).norm(dim=-1)
    s = res_copy[..., :PI.MAP].mean(-1, keepdim=True).clamp_min(1e-6)
    edit = (torch.sigmoid(gain / s + glogit) > .5) | enter
    idx = torch.zeros(*est.shape, 81, dtype=torch.long, device=frames.device)
    idx[..., :PI.MAP] = est[..., None]
    stale = frames.clone().float()
    for j in range(1, 15):
        cand = w.candidates(stale[:, j - 1:j])[:, 0]
        cp = cand.gather(-2, idx[:, j - 1][..., None, None].expand(-1, -1, 1, 192))[..., 0, :]
        new = torch.where(edit[:, j - 1][..., None], z_pre[:, j - 1], cp).gather(1, partner[:, j - 1][..., None].expand(-1, -1, 192))
        stale[:, j] = torch.where((j >= 15 - m)[:, None, None], new, frames[:, j].float())
    return stale


def losses(w, frames, inputs, ac, noise, config, scroll):
    """pi_train.losses with the backbone reading `inputs` (fresh or stale); everything else from the fresh frames"""
    est = scroll.estimate(frames[:, :-1].float(), frames[:, 1:].float())
    est16 = F.pad(est, (0, 1))
    with autocast_context(config):
        belief, ha = w.backbone_full(inputs, ac)
        copy, gen, glogit, enter, sl = w.afterstate_parts(frames, ac, belief, ha, scroll=est16)
        logits = w.completion.propose(w.completion.trunk(belief, noise)).float()
    copy, gen, glogit, enter, sl, logits = (x[:, :15] for x in (copy, gen, glogit, enter, sl, logits))
    logits = logits.masked_fill(~w.completion.valid, -torch.inf)
    z = frames[:, 1:].float()
    with torch.no_grad():
        a0 = torch.where(((glogit > 0) | enter)[..., None], gen, copy)
        partner, target, nsw, _, _, _ = IC.infer(F.layer_norm(a0, (192,)), z, IC.pair_prior_logit(logits))
        z_pre = z.gather(2, partner[..., None].expand(-1, -1, -1, 192))
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
    possible = sl.gather(-1, est[..., None])[..., 0] > -10
    scr = F.cross_entropy(sl[possible], est[possible]) if bool(possible.any()) else sl.sum() * 0
    stats = {'point': float(point.detach()), 'chance_ce': float(ce.detach()), 'gate_bce': float(gate.detach()),
             'scroll_ce': float(scr.detach()), 'edits_per_frame_nonentering': float((edit & ~enter).float().sum(-1).mean()),
             'scroll_acc': float((sl.argmax(-1) == est).float().mean())}
    return point + ce + gate + scr, stats


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--seed-set', type=int, required=True, choices=(1, 2))
    ap.add_argument('--stop-after', type=int)
    a = ap.parse_args()
    assert PREREG.exists()
    config = BASE.settings()
    manifest, windows, actions, tokens = BASE.load_data()
    d = OUT / f'pi_seed{a.seed_set}'
    st = torch.load(d / 'state.pt', map_location='cpu', weights_only=False)
    assert st['update'] == START
    arms = {}
    for arm in ('piplain', 'pistale'):
        w = PI.PIWorld().cuda().train(); w.load_state_dict(st['world'])
        opt = phase_optimizer([w], config); opt.load_state_dict(st['optimizer'])
        arms[arm] = (w, opt)
    scroll = A5.load_scroll()
    order = torch.Generator().manual_seed(20261026 + a.seed_set)
    noise_rng = torch.Generator().manual_seed(2026102600 + a.seed_set)
    m_rng = torch.Generator().manual_seed(20261027 + a.seed_set)
    source = [Path(__file__), Path(PI.__file__), Path(IC.__file__), Path(TM.__file__), PREREG, d / 'state.pt']
    contract = {'format': 'pi_stale_v1', 'sources_inputs': {str(p.relative_to(L.ROOT)): L.sha(p) for p in source},
                'data_contract_id': manifest['contract_id'], 'seed_set': a.seed_set, 'updates': [START, END],
                'training_semantic_or_simulator_labels': False}
    cid = hashlib.sha256(json.dumps(contract, sort_keys=True).encode()).hexdigest()
    history, began = [], time.perf_counter()
    for update in range(START, END):
        idx = torch.randint(len(windows), (16,), generator=order)
        frames, ac, _ = BASE.batch(tokens, actions, windows, idx)
        noise = DM.draw_noise(frames.shape[:2], noise_rng).cuda()
        m = torch.randint(0, 15, (16,), generator=m_rng).cuda()
        row = {'update': update + 1}
        for arm, (w, opt) in arms.items():
            inputs = stale_inputs(w, frames, ac, noise[0], config, scroll, m) if arm == 'pistale' else frames
            loss, stats = losses(w, frames, inputs, ac, noise[0], config, scroll)
            norm = float(optimizer_step(opt, loss, list(w.parameters()), learning_rate=_phase_lr(config, update),
                                        grad_clip=config.agent.grad_clip, strict=True, zero_grad=True))
            row[arm] = {**{k: round(v, 5) for k, v in stats.items()}, 'gradient_norm': round(norm, 4)}
        if (update + 1) % 100 == 0 or update == START:
            row['elapsed'] = round(time.perf_counter() - began, 1)
            row['peak_allocated_gib'] = round(torch.cuda.max_memory_allocated() / 2**30, 3)
            history.append(row); print(json.dumps(row), flush=True)
        if a.stop_after and update + 1 >= START + a.stop_after:
            return
    for arm, (w, _) in arms.items():
        L.atomic({'contract_id': cid, 'contract': contract, 'update': END, 'arm': arm, 'world': L.cpu_state(w), 'history': history},
                 d / f'{arm}_u{END:04d}.pt')
    print(json.dumps({'stage': 'done', 'update': END}), flush=True)


if __name__ == '__main__':
    main()
