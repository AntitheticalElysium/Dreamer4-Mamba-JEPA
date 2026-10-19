"""Causal test (2026-10-15, inference only): does imagined content at the faced cell cause camera divergence?

camera_diag.py found that when the imagined cell the player is about to enter disagrees with reality, the imagined camera
diverges 37-81% of the time (vs 5.9%). Intervention: the same self-fed H16 rollouts as transport_recursion.py (200 DEV roots,
recorded actions, same noise generator seed), but before each step, on trajectories still camera-consistent, ONE cell of the
last imagined frame is replaced by the real frame's token:
  faced    the cell the next action targets (if it is a move)
  random   a random other interior map cell (control: same count of substitutions)
  none     no substitution (reproduces transport_recursion)
Readings: camera-consistent share by depth, first-divergence rate per step, bright / dark.
Usage: faced_substitution.py <model> -> artifacts/eda/whiteboard/faced_substitution_<model>.json
"""
import json
import sys
from pathlib import Path

import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'artifacts/experiments/20261012_oracle'))
import audit_s5_direct as A5  # noqa: E402
import transport_recursion as TR  # noqa: E402

OUT = ROOT / 'artifacts/eda/whiteboard'
FACE = {1: 30, 2: 32, 3: 22, 4: 40}
INTERIOR = [r * 9 + c for r in range(1, 6) for c in range(1, 8) if r * 9 + c != 31]


@torch.no_grad()
def run(name, mode):
    import direct_model as DM
    import direct_train as BASE
    import paired_model as PM
    import transport_model as TM
    from d4mj.train import autocast_context
    config = BASE.settings()
    ckpt, arm, kind = TR.MODELS[name]
    world = (PM.PairedWorld() if arm == 'paired' else TM.TransportWorld(arm)).cuda().eval()
    world.load_state_dict(torch.load(ckpt, map_location='cpu', weights_only=False)['world'])
    panel = torch.load(A5.BASE / 'panel.pt', map_location='cpu', weights_only=False, mmap=True)
    tok, act = panel['tokens'], panel['actions']
    scroll = A5.load_scroll(); shifts = torch.as_tensor(scroll.SHIFTS)
    real_shift = shifts[scroll.estimate(tok[:, 15:31].float(), tok[:, 16:32].float())]       # [R,16,2]
    R = len(tok); K = TR.K if kind == 'sample' else 1
    g = torch.Generator().manual_seed(20261015)
    pick = torch.Generator().manual_seed(7)
    ok_all = torch.zeros(R, K, 16, dtype=torch.bool)
    for s0 in range(0, R, 2):                                              # 2 roots per batch: fits beside two training runs
        idx = list(range(s0, min(s0 + 2, R))); n = len(idx) * K
        acts = act[idx].repeat_interleave(K, 0)
        real = tok[idx].float().repeat_interleave(K, 0)                    # [n,32,81,192]
        rs = real_shift[idx].repeat_interleave(K, 0)
        hist = [real[:, j] for j in range(1, 16)]
        ok = torch.ones(n, dtype=torch.bool)
        for k in range(1, 17):
            # substitution in the last context frame (frame 15 + k - 1), before predicting frame 15 + k
            if mode != 'none' and k > 1:
                last = hist[-1].clone()
                a_prev = acts[:, 15 + k - 1]
                for b in range(n):
                    if not bool(ok[b]):
                        continue
                    if mode == 'faced':
                        ai = int(a_prev[b])
                        if ai in FACE:
                            last[b, FACE[ai]] = real[b, 15 + k - 1, FACE[ai]]
                    else:
                        if int(a_prev[b]) in FACE:
                            c = INTERIOR[int(torch.randint(len(INTERIOR), (1,), generator=pick))]
                            last[b, c] = real[b, 15 + k - 1, c]
                hist[-1] = last
            ctx = torch.stack(hist[-15:], 1).cuda(); a = acts[:, k:k + 15].cuda()
            fr = torch.cat([ctx, torch.zeros_like(ctx[:, :1])], 1); ac = F.pad(a, (0, 1))
            with autocast_context(config):
                belief, ha = world.backbone_full(fr, ac)
                if kind == 'reference':
                    nxt = F.layer_norm(world.reference(fr, ac, belief, ha)[:, 14].float(), (192,))
                else:
                    noise = DM.draw_noise((n, 16), g)[0].cuda(); gum = TM.draw_gumbel((n, 16), g)[0].cuda()
                    nxt = world.complete(fr, ac, belief, ha, noise, gum)[0][:, 14].float()
            nxt = nxt.cpu()
            est = shifts[scroll.estimate(hist[-1][:, None], nxt[:, None])[:, 0]]
            ok = ok & (est == rs[:, k - 1]).all(-1)
            ok_all.view(R * K, 16)[s0 * K:s0 * K + n, k - 1] = ok
            hist.append(nxt)
        print(json.dumps({'model': name, 'mode': mode, 'roots_done': idx[-1] + 1}), flush=True)
    return ok_all


def main():
    name = sys.argv[1]
    selected = set(json.loads((A5.BASE / 'direct_generator/data/existing_eval_selection.json').read_text())['selected_indices'])
    res = {'model': name}
    for mode in ('none', 'faced', 'random'):
        ok = run(name, mode)
        out = {}
        for part, rows in (('bright', [i for i in range(len(ok)) if i in selected]), ('dark', [i for i in range(len(ok)) if i not in selected])):
            o = ok[rows].float()
            out[part] = {'camera_consistent_by_depth': o.mean((0, 1)).round(decimals=3).tolist()}
        res[mode] = out
        print(json.dumps({'mode': mode, 'bright_d8_d16': [out['bright']['camera_consistent_by_depth'][d] for d in (7, 15)],
                          'dark_d8_d16': [out['dark']['camera_consistent_by_depth'][d] for d in (7, 15)]}), flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f'faced_substitution_{name}.json').write_text(json.dumps(res, indent=1))


if __name__ == '__main__':
    main()
