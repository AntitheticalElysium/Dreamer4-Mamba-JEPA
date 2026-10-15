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
  gated_<p>     Track B: exact copy (the argmax candidate) only where CorrT is confident (its largest mixture weight > p), the
                soft blend elsewhere: does a label-free confidence decision give exact persistence without the argmax's
                wrong copies? p in {0.5, 0.8, 0.95} is a diagnostic sweep (a Pareto curve), not a tuned choice.
  oracle_decide (privileged upper bound) per cell, the exact copy (argmax candidate) or the soft prediction, whichever is
                closer to the REAL next token, while the trajectory is camera-consistent (soft otherwise): if persistence
                decisions were right, does exact-copy-unless-change recursion persist without phantoms?
Usage: persistence_variants.py <checkpoint relative to transport/> <tag> [gated | oracle] -> artifacts/eda/whiteboard/persistence_<tag>[_gated].json
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
GATED = ('gated_0.5', 'gated_0.8', 'gated_0.95')
ORACLE = ('oracle_decide',)


def corrt_weights(world, frames, actions, belief, action_belief):
    """CorrT mixture weights over (self, 4 neighbours, generated), exactly as PairedWorld.reference computes them"""
    logits = world.choose(belief).float()
    moved = world.frame(action_belief).float()[:, :, None, :]
    target = torch.tensor([31, 30, 32, 22, 40], device=actions.device)[actions.clamp(max=4)]
    htarget = belief.gather(2, target[:, :, None, None].expand(-1, -1, 1, belief.shape[-1]))[:, :, 0]
    moved = moved + (((actions >= 1) & (actions <= 4)).float()[:, :, None, None] * world.target_gate(htarget).float()[:, :, None, :])
    return (logits + moved * logits.new_tensor([0, 1, 1, 1, 1, 0])).softmax(-1)


@torch.no_grad()
def rollout(world, variant, panel, config, real_render):
    scroll = A5.load_scroll(); shifts = torch.as_tensor(scroll.SHIFTS)
    real_shift = shifts[scroll.estimate(panel['tokens'][:, 15:31].float(), panel['tokens'][:, 16:32].float())]
    import transport_objective as TO
    from d4mj.train import autocast_context
    tok, act = panel['tokens'], panel['actions']
    R = len(tok)
    out = torch.empty(R, 1, 16, 81, 192, dtype=torch.float16)
    for s0 in range(0, R, 8):
        idx = list(range(s0, min(s0 + 8, R))); n = len(idx)
        hist = [tok[idx, j].float() for j in range(1, 16)]
        ok = torch.ones(n, dtype=torch.bool)
        for k in range(1, 17):
            ctx = torch.stack(hist[-15:], 1).cuda(); a = act[idx, k:k + 15].cuda()
            fr = torch.cat([ctx, torch.zeros_like(ctx[:, :1])], 1); ac = F.pad(a, (0, 1))
            with autocast_context(config):
                belief, ha = world.backbone_full(fr, ac)
                soft = world.reference(fr, ac, belief, ha)[:, 14].float()
                if variant != 'soft':
                    hard = TO.hard_reference(world, fr, ac, belief, ha)[:, 14].float()
                if variant.startswith('gated'):
                    wmax = corrt_weights(world, fr, ac, belief, ha)[:, 14].amax(-1, keepdim=True)
            if variant == 'soft':
                nxt = F.layer_norm(soft, (192,))
            elif variant == 'hard':
                nxt = F.layer_norm(hard, (192,))
            elif variant == 'oracle_decide':
                z = tok[idx, 15 + k].float().cuda()
                use_copy = (hard - z).norm(dim=-1, keepdim=True) <= (F.layer_norm(soft, (192,)) - z).norm(dim=-1, keepdim=True)
                use_copy = use_copy & ok.cuda()[:, None, None]
                nxt = F.layer_norm(torch.where(use_copy, hard, soft), (192,))
            elif variant.startswith('gated'):
                nxt = F.layer_norm(torch.where(wmax > float(variant.split('_')[1]), hard, soft), (192,))
            elif variant == 'hard_render':
                nxt = F.layer_norm(hard + (soft - hard)[:, :63].mean(1, keepdim=True), (192,))
            else:
                nxt = F.layer_norm(hard + real_render[idx, k - 1][:, None].cuda(), (192,))
            nxt = nxt.cpu()
            est = shifts[scroll.estimate(hist[-1][:, None], nxt[:, None])[:, 0]]
            ok = ok & (est == real_shift[idx, k - 1]).all(-1)
            hist.append(nxt)
            out[idx, 0, k - 1] = nxt.half()
    return out


def main():
    import direct_coherence as C
    import direct_train as BASE
    import transport_model as TM
    import transport_recursion as TR
    ckpt, tag = sys.argv[1], sys.argv[2]
    gated = sys.argv[3:] == ['gated']
    oracle = sys.argv[3:] == ['oracle']
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
    for v in (GATED if gated else ORACLE if oracle else VARIANTS):
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
    (OUT / f'persistence_{tag}{"_gated" if gated else "_oracle" if oracle else ""}.json').write_text(json.dumps(res, indent=1))


if __name__ == '__main__':
    main()
