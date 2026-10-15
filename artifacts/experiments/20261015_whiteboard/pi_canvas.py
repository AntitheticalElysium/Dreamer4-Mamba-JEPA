"""Track B, evaluation only (R17 follow-up): PI + a world-aligned token canvas, the memory E18's fcanvas could not read.

The trained PI world unchanged, recursion as transport_recursion.rollout (same roots, context, actions, H16, 8 samples, same
noise / Gumbel stream), with ONE change: a map cell ENTERING the view whose world position was seen before takes the last token
seen there instead of the generated token. The canvas:
  address  world position = view cell (r, c) + origin; the origin follows the label-free scroll estimate between the REAL context
           frames, then the world's OWN hard scroll decision at each imagined step (the next-view address E18 lacked)
  write    every frame (real context, then each imagined frame after chance) writes its 62 map tokens (not the player cell)
  read     known entering cells copy the stored token exactly; unknown entering cells are generated as in PI; all other cells
           follow PI's own decisions (exact copy / gated edit), then PI's chance operator
Names: pi_canvas_s{1,2} (sampled), pi_canvas_s{1,2}_ref (deterministic afterstate, no chance). Writes the same files as
transport_recursion (recursion_samples_<name>.pt, recursion_<name>.json), so every whiteboard analysis reads them by name.
Usage: pi_canvas.py <name> ...
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / '20261012_oracle'))
sys.path.insert(0, str(HERE))

import torch
from torch.nn import functional as F

import audit_s5_direct as A5
import transport_recursion as TR
import pi_model as PI

NAMES = {'pi_canvas_s1': ('pi_seed1', 'sample'), 'pi_canvas_s1_ref': ('pi_seed1', 'reference'),
         'pi_canvas_s2': ('pi_seed2', 'sample'), 'pi_canvas_s2_ref': ('pi_seed2', 'reference')}
P = TR.W + TR.H                                                           # the origin moves at most one cell per frame
ROWS, COLS = 7 + 2 * P, 9 + 2 * P
MAPCELLS = torch.tensor([x for x in range(PI.MAP) if x != 31])


def cells(origin):
    """[n,63] canvas index of every map cell for views at origin [n,2]"""
    r = torch.arange(7)[:, None] + origin[:, 0, None, None] + P
    c = torch.arange(9)[None] + origin[:, 1, None, None] + P
    return (r * COLS + c).flatten(1)


def write(store, seen, origin, frame):
    """store [n,ROWS*COLS,192], seen [n,ROWS*COLS]; frame [n,81,192]: the 62 non-player map tokens"""
    idx = cells(origin)[:, MAPCELLS]
    store.scatter_(1, idx[..., None].expand(-1, -1, 192), frame[:, MAPCELLS].to(store.dtype))
    seen.scatter_(1, idx, True)


@torch.no_grad()
def rollout(name):
    import direct_model as DM
    import direct_train as BASE
    import transport_model as TM
    from d4mj.train import autocast_context
    config = BASE.settings()
    directory, mode = NAMES[name]
    world = PI.PIWorld().cuda().eval()
    world.load_state_dict(torch.load(TR.OUT / directory / 'pi_u3000.pt', map_location='cpu', weights_only=False)['world'])
    panel = torch.load(A5.BASE / 'panel.pt', map_location='cpu', weights_only=False, mmap=True)
    tok, act = panel['tokens'], panel['actions']
    scroll = A5.load_scroll(); shifts = torch.as_tensor(scroll.SHIFTS)
    R = len(tok)
    k_samples = TR.K if mode == 'sample' else 1
    out = torch.empty(R, k_samples, TR.H, 81, 192, dtype=torch.float16)
    flat = out.view(R * k_samples, TR.H, 81, 192)
    g = torch.Generator().manual_seed(20261015)                            # transport_recursion's stream
    B = 8
    stats = {'entering_cells': 0, 'entering_known': 0}
    for s0 in range(0, R, B):
        idx = list(range(s0, min(s0 + B, R)))
        n = len(idx) * k_samples
        frames = tok[idx, 1:16].float().repeat_interleave(k_samples, 0)
        acts = act[idx].repeat_interleave(k_samples, 0)
        hist = [frames[:, j] for j in range(15)]
        store = torch.zeros(n, ROWS * COLS, 192, dtype=torch.float16); seen = torch.zeros(n, ROWS * COLS, dtype=torch.bool)
        origin = torch.zeros(n, 2, dtype=torch.long)
        est = scroll.estimate(frames[:, :-1], frames[:, 1:])               # [n,14] label-free scrolls of the real context
        write(store, seen, origin, frames[:, 0])
        for j in range(14):
            origin = origin + shifts[est[:, j]]
            write(store, seen, origin, frames[:, j + 1])
        for k in range(1, TR.H + 1):
            ctx = torch.stack(hist[-TR.W:], 1).cuda()
            a = acts[:, k:k + TR.W].cuda()
            fr = torch.cat([ctx, torch.zeros_like(ctx[:, :1])], 1); ac = F.pad(a, (0, 1))
            with autocast_context(config):
                belief, ha = world.backbone_full(fr, ac)
                copy, gen, glogit, enter, sl = world.afterstate_parts(fr, ac, belief, ha)
                after = torch.where(((glogit > 0) | enter)[..., None], gen, copy)
                choice = sl[:, 14].argmax(-1).cpu()
                nxt_origin = origin + shifts[choice]
                addr = cells(nxt_origin)                                   # [n,63]
                known = seen.gather(1, addr) & enter[:, 14, :PI.MAP].cpu()
                remembered = store.gather(1, addr[..., None].expand(-1, -1, 192))
                last = after[:, 14].clone()
                last[:, :PI.MAP] = torch.where(known[..., None].cuda(), remembered.cuda().to(last.dtype), last[:, :PI.MAP])
                after = torch.cat([after[:, :14], last[:, None], after[:, 15:]], 1)
                if mode == 'reference':
                    nxt = F.layer_norm(after[:, 14].float(), (192,))
                else:
                    noise = DM.draw_noise((n, 16), g)[0].cuda(); gum = TM.draw_gumbel((n, 16), g)[0].cuda()
                    mixed, _ = world.completion.operate(after, world.completion.trunk(belief, noise), gum)
                    nxt = F.layer_norm(mixed, (192,))[:, 14].float()
            stats['entering_cells'] += int(enter[:, 14, :PI.MAP].sum()); stats['entering_known'] += int(known.sum())
            nxt = nxt.cpu()
            origin = nxt_origin
            write(store, seen, origin, nxt)
            hist.append(nxt)
            flat[s0 * k_samples:s0 * k_samples + n, k - 1] = nxt.half()
        print(json.dumps({'model': name, 'roots_done': idx[-1] + 1, **stats}), flush=True)
    return out, stats


if __name__ == '__main__':
    for name in sys.argv[1:]:
        path = TR.OUT / f'recursion_samples_{name}.pt'
        imagined, stats = rollout(name)
        torch.save(imagined, path)
        res = TR.evaluate(name, imagined)
        res['canvas'] = stats
        (TR.OUT / f'recursion_{name}.json').write_text(json.dumps(res, indent=1))
        b = res['bright']
        print(json.dumps({'model': name, 'canvas': stats, 'bright_census_d1_d8_d16': [b['census_ratio[zombie,cow,skeleton]_by_depth'][d] for d in (0, 7, 15)],
                          'camera_d16': b['camera_consistent_by_depth'][15], 'tile_seen_d1_d16': [b['tile_error_seen_by_depth'][d] for d in (0, 15)],
                          'phantoms_d1_d16': [b['phantoms_per_frame_by_depth'][d] for d in (0, 15)]}), flush=True)
