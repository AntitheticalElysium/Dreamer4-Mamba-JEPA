"""LB evaluation in PI's protocols (LB_PREREG.md; 2026-10-15 late).

recursion  transport_recursion.rollout's protocol: 200 DEV panel roots, context = panel frames 1..15 (observed by the posterior),
           recorded actions; 16 prior steps (the step into frame 15+k takes the action at frame 14+k, as rollout's window); 8
           samples per root; each imagined frame = the token readout of the sampled latent. Writes recursion_samples_<name>.pt
           and recursion_<name>.json exactly as transport_recursion, so every whiteboard analysis reads them by name.
onestep    transport_eval's exploratory cohort: observe the 15 context frames, one prior step with the last context action, 64
           prior samples -> pred_lb.pt in transport_eval's format (reference = readout of the prior mode), then analyze with
           transport_eval.analyze (both readers) against the seed's free arm.
Usage: lb_eval.py <seed> <update> <name>   e.g. lb_eval.py 1 30000 lb_s1
"""
import json
import os
import sys
from pathlib import Path

os.environ.setdefault('TRITON_F32_DEFAULT', 'ieee')
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / '20261012_oracle'))
sys.path.insert(0, str(HERE))

import torch
from torch.nn import functional as F

import lb_world as LB
import audit_s5_direct as A5
import transport_recursion as TR

FREE = {1: A5.BASE / 'transport/eval_u3000/pred_free.pt', 2: A5.BASE / 'transport/seed2/eval_u3000/pred_free.pt'}


@torch.no_grad()
def recursion(w, name):
    panel = torch.load(A5.BASE / 'panel.pt', map_location='cpu', weights_only=False, mmap=True)
    tok, act = panel['tokens'], panel['actions']
    R, K = len(tok), TR.K
    out = torch.empty(R, K, TR.H, 81, 192, dtype=torch.float16)
    torch.manual_seed(20261015)
    for s0 in range(0, R, 8):
        idx = list(range(s0, min(s0 + 8, R)))
        frames = tok[idx, 1:16].float().cuda().repeat_interleave(K, 0)            # [n,15,81,192]
        acts = act[idx].cuda().repeat_interleave(K, 0)                             # [n,31]
        stoch, deter, _ = w.observe(frames, acts[:, 1:16])
        stoch, deter = stoch[:, -1], deter[:, -1]
        for k in range(1, TR.H + 1):
            stoch, deter = w.rssm.img_step(stoch, deter, F.one_hot(acts[:, 14 + k], LB.N_ACT).float())
            out[s0:s0 + len(idx), :, k - 1] = w.readout(stoch, deter).view(len(idx), K, 81, 192).half().cpu()
    torch.save(out, TR.OUT / f'recursion_samples_{name}.pt')
    res = TR.evaluate(name, out)
    (TR.OUT / f'recursion_{name}.json').write_text(json.dumps(res, indent=1))
    b = res['bright']
    print(json.dumps({'model': name, 'bright_census_d1_d8_d16': [b['census_ratio[zombie,cow,skeleton]_by_depth'][d] for d in (0, 7, 15)],
                      'camera_d16': b['camera_consistent_by_depth'][15], 'tile_seen_d1_d16': [b['tile_error_seen_by_depth'][d] for d in (0, 15)],
                      'phantoms_d1_d16': [b['phantoms_per_frame_by_depth'][d] for d in (0, 15)]}), flush=True)


@torch.no_grad()
def onestep(w, seed, update, d):
    import direct_eval as E
    import transport_eval as TE
    data, _ = E.cohort('exploratory')
    preds, refs = [], []
    torch.manual_seed(2026101402)
    for start in range(0, len(data['roots']), 4):
        ctx = data['context_tokens'][start:start + 4].cuda().float(); ac = data['actions'][start:start + 4].cuda()
        stoch, deter, _ = w.observe(ctx, ac)
        deter1 = w.rssm._deter_net(stoch[:, -1], deter[:, -1], F.one_hot(ac[:, -1], LB.N_ACT).float())
        _, logit = w.rssm.prior(deter1)
        mode = F.one_hot(logit.argmax(-1), logit.shape[-1]).float()
        refs.append(w.readout(mode, deter1).float().cpu())
        out = []
        for _ in range(64):
            s1 = w.rssm.get_dist(logit).rsample()                                  # R2-Dreamer draws with rsample
            out.append(w.readout(s1, deter1).float().cpu())
        preds.append(torch.stack(out, 1))
    d.mkdir(parents=True, exist_ok=True)
    torch.save({'arm': 'lb', 'update': update, 'predictions': torch.cat(preds), 'reference': torch.cat(refs), 'partner': None}, d / 'pred_lb.pt')
    TE.analyze(update, eval_dir=d, arms=['lb'], free_path=FREE[seed])                # writes d / analysis.json
    res = json.loads((d / 'analysis.json').read_text())
    a = res['arms']['lb']
    print(json.dumps({'onestep': {r: {k: a[r][k] for k in ('valid', 'duplicate', 'absent', 'valid_move', 'cow_valid_move', 'scored_joint_ES')} for r in ('real', 'bank')},
                      'static': a['scored_static_corruption']}), flush=True)


def main():
    seed, update, name = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
    d = A5.BASE / f'transport/lb_seed{seed}'
    w = LB.LBWorld('cuda').cuda().eval()
    w.load_state_dict(torch.load(d / f'lb_u{update:05d}.pt', map_location='cpu', weights_only=False)['world'])
    onestep(w, seed, update, d / f'eval_u{update:05d}')
    recursion(w, name)


if __name__ == '__main__':
    main()
