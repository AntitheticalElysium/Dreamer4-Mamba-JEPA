"""How much of the training signal reaches the health token? (2026-10-08, health diagnosis 5.)
A frozen world on ITS OWN training batches: tworld.train's rawlong sampler replayed exactly (batch order generator seed 11,
40 windows of L frames, TERMINAL_SHARE end-aligned deaths), teacher-forced, the training objective err.mean() over
[B, L-1, 81, 192]. Health per true frame from teval's Probes.hud reader; a transition is an ordinary hit if health drops by
> 1.5 and the next health is > 0.5, a death if the next health is < 0.5 (from > 0.5), unchanged if |dh| < 0.5.
Additive components of the objective (each normalized like the objective, so they sum to it):
  tok63_hit / tok63_death / tok63_unchanged / tok63_all = the error at token 63 on those transitions; all = the objective.
Per component: its share of the objective and its gradient on the BACKBONE parameters (everything except the head's proj,
choose, frame, target_gate): norm ratio to the full objective's gradient and cosine with it (summed over batches, i.e. the
update direction an optimizer would see before Adam's normalization). Also, at token 63 on hits: the corrt mixture weights
(the generator receives the error scaled by its weight) and the per-dim error of copy vs the world's output.
E21 worlds (2026-10-08): `all` is the arm's whole objective (teacher L1 + generator loss + event loss, as trained) and the
extra terms are components too: gen_term, event_term, event_tok63_hit (the event loss at token 63 on hits). The event term is
recomputed here as tworld.rollout_losses does it and checked against it on the first chunk.
Usage: health_gradient.py <world.pt> [batches]  -> artifacts/eda/health_chain_v1/gradient_<world>.json
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
ROOT = HERE.parents[2]
OUT = ROOT / 'artifacts/eda/health_chain_v1'
HEAD = ('proj.', 'choose.', 'frame.', 'target_gate.', 'event_head.')


def main(path, batches):
    import teval as T
    import tworld as TW
    from d4mj.config import config_from_dict
    from d4mj.train import autocast_context
    import spatial as S
    device = torch.device('cuda')
    config = config_from_dict(torch.load(S.CHECKPOINT, map_location='cpu', weights_only=False)['config'])
    world, st = T.load_world(Path(path), device)
    L = world.time.shape[0]
    meta, train_roots, train_seeds = T.split()
    P = T.Probes(T.build_cache('raw', torch.device('cpu')), meta, train_roots, train_seeds)
    pool = torch.load(TW.POOLS['rawlong'] / 'labels.pt', weights_only=False, mmap=True)
    tokens = np.memmap(TW.POOLS['rawlong'] / 'tokens.f16', dtype=np.float16, mode='r', shape=(len(pool['terminal']), 64, 81, 192))
    main_rows, term_rows = torch.where(~pool['terminal'])[0], torch.where(pool['terminal'])[0]
    order = torch.Generator().manual_seed(11)
    named = [(n, p) for n, p in world.named_parameters() if not n.startswith(HEAD)]
    params = [p for _, p in named]
    gl, ev = st['args'].get('gen_loss') == 'True', st['args'].get('event') == 'True'
    comps = ('all', 'teacher', 'tok63_all', 'tok63_hit', 'tok63_death', 'tok63_unchanged') + (('gen_term',) if gl else ()) \
        + (('event_term', 'event_tok63_hit') if ev else ())
    checked = False
    grad = {c: [torch.zeros_like(p) for p in params] for c in comps}
    loss = {c: 0.0 for c in comps}
    counts = {'transitions': 0, 'hit': 0, 'death': 0, 'unchanged': 0}
    wsum = {'hit': torch.zeros(6), 'unchanged': torch.zeros(6)}
    err_hit = {'world': 0.0, 'copy': 0.0}
    for u in range(batches):
        W = TW.BATCH
        term = torch.rand(W, generator=order) < TW.TERMINAL_SHARE
        r = torch.where(term, term_rows[torch.randint(len(term_rows), (W,), generator=order)],
                        main_rows[torch.randint(len(main_rows), (W,), generator=order)])
        t0 = torch.where(term, 64 - L, torch.randint(0, 65 - L, (W,), generator=order))
        s_all = torch.stack([torch.from_numpy(np.array(tokens[int(i), int(j):int(j) + L])) for i, j in zip(r, t0)]).float()
        a_all = torch.stack([pool['actions'][int(i), int(j):int(j) + L - 1] for i, j in zip(r, t0)])
        norm = W * (L - 1) * 81 * 192                                     # err.mean() over the whole batch of 40
        for c0 in range(0, W, 8):
            s, a = s_all[c0:c0 + 8].to(device), F.pad(a_all[c0:c0 + 8], (0, 1)).to(device)
            health = (P.hud(s[:, :, 63:81].flatten(-2).flatten(0, 1).cpu())[:, 0] * 9).float().view(len(s), L)   # [b, L]
            dh = health[:, 1:] - health[:, :-1]
            hit = (dh < -1.5) & (health[:, 1:] > 0.5)
            death = (health[:, 1:] < 0.5) & (health[:, :-1] > 0.5)
            unch = dh.abs() < 0.5
            with autocast_context(config):
                pred, h, gen = world(s, a)
            err = (pred[:, :L - 1].float() - s[:, 1:]).abs()                                   # [b, L-1, 81, 192]
            e63 = err[:, :, 63].sum(-1)                                                         # [b, L-1]
            m = {'tok63_hit': hit, 'tok63_death': death, 'tok63_unchanged': unch}
            parts = {'teacher': err.sum() / norm, 'tok63_all': e63.sum() / norm}
            parts.update({k: (e63 * v.to(device)).sum() / norm for k, v in m.items()})
            parts['all'] = parts['teacher']
            if gl:                                                      # tworld.rollout_losses' gen_loss term
                parts['gen_term'] = (F.layer_norm(gen[:, :L - 1].float(), (192,)) - s[:, 1:]).abs().sum() / norm
                parts['all'] = parts['all'] + parts['gen_term']
            if ev:                                                      # tworld.rollout_losses' event term, per element
                y = ((s[:, 1:] - s[:, :-1]).norm(dim=-1) > TW.EVENT_TAU).float()
                logit = world.event_head(h[:, :L - 1].float())[..., 0]
                p = torch.sigmoid(logit)
                p_t, a_t = p * y + (1 - p) * (1 - y), 0.15 * y + 0.85 * (1 - y)
                focal = a_t * (1 - p_t) ** 4 * F.binary_cross_entropy_with_logits(logit, y, reduction='none')
                ges = lambda q: 1 / torch.log(0.1 + q + torch.sqrt(1 + q * q))
                share = lambda part, balance: (part.mean(-1, keepdim=True) / balance).clamp(2e-4, 1).expand_as(part)
                el = focal * torch.cat([ges(share(y[..., :63], 0.25)), ges(share(y[..., 63:], 1.0))], -1) * 0.1
                parts['event_term'] = el.sum() / (W * (L - 1) * 81)
                parts['event_tok63_hit'] = (el[:, :, 63] * hit.to(device)).sum() / (W * (L - 1) * 81)
                parts['all'] = parts['all'] + parts['event_term']
                if not checked:                                         # same value as the training code's term
                    with torch.no_grad(), autocast_context(config):
                        ref = TW.rollout_losses(world, s, a[:, :L - 1], 'teacher', False, None, None, True) \
                            - TW.rollout_losses(world, s, a[:, :L - 1], 'teacher')
                    assert abs(float(ref) - float(el.mean())) < 1e-3 * max(1.0, abs(float(ref))), (float(ref), float(el.mean()))
                    checked = True
            for k, v in parts.items():
                g = torch.autograd.grad(v, params, retain_graph=True, allow_unused=True)
                for acc, gi in zip(grad[k], g):
                    if gi is not None:
                        acc += gi.float()
                loss[k] += float(v)
            counts['transitions'] += hit.numel(); counts['hit'] += int(hit.sum()); counts['death'] += int(death.sum())
            counts['unchanged'] += int(unch.sum())
            wl = world.last_weights[:, :L - 1, 63].float().cpu()                               # [b, L-1, 6]
            wsum['hit'] += wl[hit].sum(0); wsum['unchanged'] += wl[unch].sum(0)
            with torch.no_grad():
                hd = hit.to(device)
                err_hit['world'] += float(err[:, :, 63].mean(-1)[hd].sum())
                err_hit['copy'] += float((s[:, :-1, 63] - s[:, 1:, 63]).abs().mean(-1)[hd].sum())
        print(json.dumps({'batch': u + 1, **counts}), flush=True)
    flat = {k: torch.cat([g.flatten() for g in v]) for k, v in grad.items()}
    res = {'world': st['name'], 'frames': L, 'batches': batches, 'counts': counts,
           'objective': loss['all'] / batches,
           'share_of_objective': {k: round(loss[k] / loss['all'], 6) for k in comps},
           'backbone_grad_norm_ratio': {k: round(float(flat[k].norm() / flat['all'].norm()), 6) for k in comps},
           'backbone_grad_cosine_with_all': {k: round(float(F.cosine_similarity(flat[k], flat['all'], 0)), 4) for k in comps},
           'token63_mixture_weights_self_up_down_left_right_gen': {k: [round(float(x), 4) for x in v / max(counts[k], 1)] for k, v in wsum.items()},
           'token63_mean_L1_on_hits': {k: round(v / max(counts['hit'], 1), 4) for k, v in err_hit.items()}}
    (OUT / f"gradient_{st['name']}.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == '__main__':
    sys.path.insert(0, str(ROOT / 'artifacts/experiments/20260926_diagnosis'))
    sys.path.insert(0, str(ROOT / 'artifacts/experiments/20260921_readout_ladder'))
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 20)
