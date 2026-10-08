"""E21 pre-flight (2026-10-08, after aborting E21 v1): can the event head learn token-63 events at all, and does its loss push
on h63 where B1 needs it? Frozen M16 s7.
  1. The head alone is trained on M16's own training batches (tworld.train's rawlong sampler: seed-11 order, 40 windows of 16
     frames; 60 batches cached in CPU memory, 600 updates, AdamW lr 1e-3) with exactly the E21 event term (tworld.rollout_losses: focal,
     GES, modality weight). Then on the health_chain subset (held episodes, window 15, output position) its token-63 logit:
     AUC hit vs unchanged, fresh hit vs unchanged. Heads: v1 (one 4-unit head shared by every slot, as launched at 16:44) and
     tworld.SlotEvent (per-slot weights, the fix).
  2. With the trained SlotEvent, on 10 other training batches (order seed 12): the gradient that the event term and the teacher
     L1, both at token 63 on hit transitions (reader drop > 1.5, living), send into the backbone output h at slot 63 (taken
     through a forward hook on the backbone's final norm, so both terms flow through the real head code). Ratio of norms; both
     terms carry the objective's own normalization (event elements over B(T-1)81, L1 over B(T-1)81x192), so the ratio is what
     --event adds relative to what the world already gets.
Usage: event_preflight.py  -> artifacts/eda/health_chain_v1/event_preflight.json
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
ROOT = HERE.parents[2]
OUT = ROOT / 'artifacts/eda/health_chain_v1'
M16 = ROOT / 'artifacts/eda/levers_tworlds_v1/corrt_rawlong_teacher_s7_fmamba_L16b40_from36000.pt'


def event_elements(head, h, s, tau):
    """tworld.rollout_losses' event term per element [B, T-1, 81] (its .mean() is the term), for a given head."""
    y = ((s[:, 1:].float() - s[:, :-1].float()).norm(dim=-1) > tau).float()
    logit = head(h.float())[:, :s.shape[1] - 1]
    p = torch.sigmoid(logit)
    p_t, a_t = p * y + (1 - p) * (1 - y), 0.15 * y + 0.85 * (1 - y)
    focal = a_t * (1 - p_t) ** 4 * F.binary_cross_entropy_with_logits(logit, y, reduction='none')
    ges = lambda q: 1 / torch.log(0.1 + q + torch.sqrt(1 + q * q))
    share = lambda part, balance: (part.mean(-1, keepdim=True) / balance).clamp(2e-4, 1).expand_as(part)
    return focal * torch.cat([ges(share(y[..., :63], 0.25)), ges(share(y[..., 63:], 1.0))], -1) * 0.1


class Shared(nn.Module):
    """E21 v1's event head: one Linear(256, 4) -> LayerNorm(4) -> SiLU -> Linear(4, 1) for every slot."""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(256, 4), nn.LayerNorm(4), nn.SiLU(), nn.Linear(4, 1))

    def forward(self, h):
        return self.net(h)[..., 0]


def main():
    import teval as T
    import tworld as TW
    import health_evidence as HE
    from d4mj.config import config_from_dict
    from d4mj.train import autocast_context
    import spatial as S
    dev = torch.device('cuda')
    config = config_from_dict(torch.load(S.CHECKPOINT, map_location='cpu', weights_only=False)['config'])
    world, _ = T.load_world(M16, dev)
    for p in world.parameters():
        p.requires_grad_(False)
    meta, train_roots, train_seeds = T.split()
    P = T.Probes(T.build_cache('raw', torch.device('cpu')), meta, train_roots, train_seeds)
    pool = torch.load(TW.POOLS['rawlong'] / 'labels.pt', weights_only=False, mmap=True)
    tokens = np.memmap(TW.POOLS['rawlong'] / 'tokens.f16', dtype=np.float16, mode='r', shape=(len(pool['terminal']), 64, 81, 192))
    main_rows, term_rows = torch.where(~pool['terminal'])[0], torch.where(pool['terminal'])[0]

    def batches(n, seed):
        order = torch.Generator().manual_seed(seed)
        for _ in range(n):
            term = torch.rand(40, generator=order) < TW.TERMINAL_SHARE
            r = torch.where(term, term_rows[torch.randint(len(term_rows), (40,), generator=order)],
                            main_rows[torch.randint(len(main_rows), (40,), generator=order)])
            t0 = torch.where(term, 64 - 16, torch.randint(0, 49, (40,), generator=order))
            s = torch.stack([torch.from_numpy(np.array(tokens[int(i), int(j):int(j) + 16])) for i, j in zip(r, t0)]).float()
            a = torch.stack([pool['actions'][int(i), int(j):int(j) + 15] for i, j in zip(r, t0)])
            yield s, F.pad(a, (0, 1))

    cache = []
    with torch.no_grad():
        for s, a in batches(60, 11):
            hs = []
            for c in range(0, 40, 8):
                with autocast_context(config):
                    hs.append(world(s[c:c + 8].to(dev), a[c:c + 8].to(dev))[1].half().cpu())
            cache.append((torch.cat(hs), s.half()))
    sub = torch.load(OUT / 'subset.pt', weights_only=False)
    mm = np.memmap(OUT / 'subset.f16', dtype=np.float16, mode='r', shape=tuple(sub['shape']))
    N = len(sub['classes']); hit, unch = sub['classes'] == 1, sub['classes'] == 2; held = ~sub['fit']
    with torch.no_grad():
        hsub = []
        for i in range(0, N, 8):
            with autocast_context(config):
                h = world(torch.from_numpy(np.array(mm[i:i + 8, 0:15])).float().to(dev), sub['actions'][i:i + 8, 0:15].to(dev))[1]
            hsub.append(h[:, -2:].half().cpu())                          # positions 13, 14 (the head reads t and t-1)
        hsub = torch.cat(hsub)
    res = {}
    for name, make in (('v1_shared', Shared), ('slot_event', TW.SlotEvent)):
        torch.manual_seed(0)
        head = make().to(dev)
        opt = torch.optim.AdamW(head.parameters(), lr=1e-3)
        for step in range(600):
            h, s = (t.to(dev) for t in cache[step % len(cache)])
            loss = event_elements(head, h, s, TW.EVENT_TAU).mean()
            opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            e = torch.cat([head(hsub[i:i + 512].to(dev).float())[:, -1, 63].float().cpu() for i in range(0, N, 512)])
        keep, fk = held & (hit | unch), held & ((hit & sub['fresh']) | unch)
        res[name] = {'final_event_loss': round(float(loss), 6), 'token63_auc_hit_vs_unchanged': round(HE.auc(e[keep], hit[keep]), 4),
                     'token63_auc_fresh_vs_unchanged': round(HE.auc(e[fk], hit[fk]), 4)}
        print(json.dumps({name: res[name]}), flush=True)
    # 2. gradient into h at slot 63 on hits: event term (trained SlotEvent) vs teacher L1
    saved = {}

    def grab(module, inputs, output):                                   # the backbone's final norm: [b, T, 82, D]
        saved['x'] = output.detach().float().requires_grad_(True)
        return saved['x']
    hook = world.norm.register_forward_hook(grab)
    g_ev = g_l1 = 0.0
    n_hits = 0
    for s, a in batches(10, 12):
        for c in range(0, 40, 8):
            sc, ac = s[c:c + 8].to(dev), a[c:c + 8].to(dev)
            hp = (P.hud(sc[:, :, 63:81].flatten(-2).flatten(0, 1).cpu())[:, 0] * 9).float().view(8, 16)
            m = (((hp[:, 1:] - hp[:, :-1]) < -1.5) & (hp[:, 1:] > 0.5)).to(dev)            # [8, 15] hit transitions
            if not m.any():
                continue
            with autocast_context(config):
                out, h, _ = world(sc, ac)
            x = saved['x']
            l1 = ((out[:, :15, 63].float() - sc[:, 1:, 63]).abs().mean(-1) * m).sum()
            gl1 = torch.autograd.grad(l1, x, retain_graph=True)[0][:, :15, 64][m]       # slot 63 = index 64 (action first)
            ev = (event_elements(head, x[:, :, 1:], sc, TW.EVENT_TAU)[..., 63] * m).sum()
            gev = torch.autograd.grad(ev, x)[0][:, :15, 64][m]
            g_ev += float(gev.norm(dim=-1).sum()); g_l1 += float(gl1.norm(dim=-1).sum()); n_hits += int(m.sum())
    hook.remove()
    res['grad_into_h63_on_hits'] = {'hits': n_hits, 'event_mean_norm': g_ev / n_hits, 'teacher_l1_mean_norm': g_l1 / n_hits,
                                    'event_over_teacher': round(g_ev / g_l1, 4)}
    print(json.dumps(res['grad_into_h63_on_hits']), flush=True)
    (OUT / 'event_preflight.json').write_text(json.dumps(res, indent=1))


if __name__ == '__main__':
    sys.path.insert(0, str(ROOT / 'artifacts/experiments/20260926_diagnosis'))
    sys.path.insert(0, str(ROOT / 'artifacts/experiments/20260921_readout_ladder'))
    main()
