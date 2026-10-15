"""Track A structural diagnostic (2026-10-15 evening, inference only): is recursive mob fading caused by SOFT persistence, so
that it disappears when unchanged content is copied exactly, with rendering changes handled separately?

Deterministic self-fed H16 rollouts (no chance step) from the 200 DEV panel roots with recorded actions, for one trained world.
At every step the afterstate is built four ways from the same backbone pass:
  soft          LN(b), b = CorrT's soft mixture of copy candidates and the generated token (the current model)
  hard          LN(b_hard): every cell takes its highest-weight CorrT candidate exactly (an idempotent copy when that is a
                copy candidate)
  hard_render   LN(b_hard + r): plus a label-free global rendering update r = mean over map cells of (b - b_hard), the part
                of the soft prediction shared by all cells
  hard_oracle   LN(b_hard + r*): plus the TRUE global rendering change r* = mean real change over cells whose decoded tile and
                mob content are unchanged between the real frames t-1 and t (privileged; a ceiling for exact persistence)
Readings (bright): cow / zombie census by depth, tile error on seen cells, camera consistency, phantoms
(transport_recursion.evaluate), and the persisting-mob score decay (mob_fading.py on the saved samples).
Usage: persistence_variants.py <checkpoint relative to transport/> <tag> -> artifacts/eda/whiteboard/persistence_<tag>.json
"""
import json
import sys
from pathlib import Path

import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'artifacts/experiments/20261012_oracle'))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import audit_s5_direct as A5  # noqa: E402

OUT = ROOT / 'artifacts/eda/whiteboard'
VARIANTS = ('soft', 'hard', 'hard_render', 'hard_oracle')


@torch.no_grad()
def rollout(world, variant, panel, config, real_render):
    import transport_objective as TO
    from d4mj.train import autocast_context
    tok, act = panel['tokens'], panel['actions']
    R = len(tok)
    out = torch.empty(R, 1, 16, 81, 192, dtype=torch.float16)
    for s0 in range(0, R, 8):
        idx = list(range(s0, min(s0 + 8, R))); n = len(idx)
        hist = [tok[idx, j].float() for j in range(1, 16)]
        for k in range(1, 17):
            ctx = torch.stack(hist[-15:], 1).cuda(); a = act[idx, k:k + 15].cuda()
            fr = torch.cat([ctx, torch.zeros_like(ctx[:, :1])], 1); ac = F.pad(a, (0, 1))
            with autocast_context(config):
                belief, ha = world.backbone_full(fr, ac)
                soft = world.reference(fr, ac, belief, ha)[:, 14].float()
                if variant != 'soft':
                    hard = TO.hard_reference(world, fr, ac, belief, ha)[:, 14].float()
            if variant == 'soft':
                nxt = F.layer_norm(soft, (192,))
            elif variant == 'hard':
                nxt = F.layer_norm(hard, (192,))
            elif variant == 'hard_render':
                nxt = F.layer_norm(hard + (soft - hard)[:, :63].mean(1, keepdim=True), (192,))
            else:
                nxt = F.layer_norm(hard + real_render[idx, k - 1][:, None].cuda(), (192,))
            nxt = nxt.cpu()
            hist.append(nxt)
            out[idx, 0, k - 1] = nxt.half()
    return out


def main():
    import direct_coherence as C
    import direct_train as BASE
    import transport_model as TM
    import transport_recursion as TR
    ckpt, tag = sys.argv[1], sys.argv[2]
    config = BASE.settings()
    world = TM.TransportWorld('operator').cuda().eval()
    world.load_state_dict(torch.load(A5.BASE / 'transport' / ckpt, map_location='cpu', weights_only=False)['world'])
    panel = torch.load(A5.BASE / 'panel.pt', map_location='cpu', weights_only=False, mmap=True)
    bank = torch.load(A5.BASE / 'snap_bank_readers.pt', map_location='cpu', weights_only=False)
    head = torch.load(A5.BASE / 'learned_innovation/eval_u3000_h01/mob_heads.pt', map_location='cpu', weights_only=False)['heads']['real']
    scroll = A5.load_scroll(); shifts = torch.as_tensor(scroll.SHIFTS)
    real = panel['tokens'][:, 15:32].float()
    sh = shifts[scroll.estimate(real[:, :-1], real[:, 1:])]
    dec = C.decode(real, bank, head)
    tiles, mobs = dec['tile'], dec['mob']['real'][..., :3]
    R = len(real)
    real_render = torch.zeros(R, 16, 192)                                   # privileged true global rendering change
    for i in range(R):
        for k in range(16):
            dr, dc = (int(v) for v in sh[i, k])
            d = []
            for r in range(7):
                for c in range(9):
                    if 0 <= r + dr < 7 and 0 <= c + dc < 9 and 31 not in (r * 9 + c, (r + dr) * 9 + c + dc):
                        j, jp = r * 9 + c, (r + dr) * 9 + c + dc
                        if tiles[i, k + 1, j] == tiles[i, k, jp] and bool((mobs[i, k + 1, j] == mobs[i, k, jp]).all()):
                            d.append(real[i, k + 1, j] - real[i, k, jp])
            if d:
                real_render[i, k] = torch.stack(d).mean(0)
    res = {'checkpoint': ckpt}
    for v in VARIANTS:
        imag = rollout(world, v, panel, config, real_render)
        key = f'persist_{tag}_{v}'
        torch.save(imag, A5.BASE / f'transport/recursion_samples_{key}.pt')
        b = TR.evaluate(key, imag)['bright']
        c = b['census_ratio[zombie,cow,skeleton]_by_depth']
        res[v] = {'cow_census_d1_d8_d16': [round(c[d][1], 3) for d in (0, 7, 15)], 'zombie_census_d1_d8_d16': [round(c[d][0], 3) for d in (0, 7, 15)],
                  'tile_seen_d1_d8_d16': [b['tile_error_seen_by_depth'][d] for d in (0, 7, 15)],
                  'camera_d8_d16': [b['camera_consistent_by_depth'][d] for d in (7, 15)],
                  'phantoms_max': max(b['phantoms_per_frame_by_depth'])}
        print(json.dumps({v: res[v]}), flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f'persistence_{tag}.json').write_text(json.dumps(res, indent=1))


if __name__ == '__main__':
    main()
