"""Does recursion degrade the deterministic part even when the chance outcome is exactly right? (2026-10-15, inference only)

Self-fed H16 rollouts from the 200 DEV panel roots (recorded actions), three ways, for one operator checkpoint:
  sampled          the model's own sampler (Gumbel proposals + matching): the usual imagination
  posterior        chance supplied: at each step the reassignment is inferred label-free from the model's own afterstate and the
                   REAL next frame (transport_ic_train.infer with the model's own prior), on camera-consistent trajectories
                   (otherwise the sampler); the deterministic afterstate still reads the imagined history
  reference        no chance at all: LN(b)
Readings per depth (bright): mob persistence score (real reader score of mobs that persist), cow census ratio, tile error on
seen cells, camera-consistent share, zombie approach rate. If posterior-forced rollouts still fade / drift / stop chasing,
the deterministic dynamics are the recursion bottleneck independently of chance.
Usage: posterior_forced.py <checkpoint relative to transport/> -> artifacts/eda/whiteboard/posterior_forced_<name>.json
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


@torch.no_grad()
def rollout(world, mode, panel, config, K=4):
    import direct_model as DM
    import transport_model as TM
    import transport_ic_train as IC
    from d4mj.train import autocast_context
    tok, act = panel['tokens'], panel['actions']
    scroll = A5.load_scroll(); shifts = torch.as_tensor(scroll.SHIFTS)
    R = len(tok)
    out = torch.empty(R, K, 16, 81, 192, dtype=torch.float16)
    flat = out.view(R * K, 16, 81, 192)
    g = torch.Generator().manual_seed(20261015)
    real_shift = shifts[scroll.estimate(tok[:, 15:31].float(), tok[:, 16:32].float())]
    for s0 in range(0, R, 4):
        idx = list(range(s0, min(s0 + 4, R))); n = len(idx) * K
        acts = act[idx].repeat_interleave(K, 0); real = tok[idx].float().repeat_interleave(K, 0)
        rs = real_shift[idx].repeat_interleave(K, 0)
        hist = [real[:, j] for j in range(1, 16)]
        ok = torch.ones(n, dtype=torch.bool)
        for k in range(1, 17):
            ctx = torch.stack(hist[-15:], 1).cuda(); a = acts[:, k:k + 15].cuda()
            fr = torch.cat([ctx, torch.zeros_like(ctx[:, :1])], 1); ac = F.pad(a, (0, 1))
            with autocast_context(config):
                belief, ha = world.backbone_full(fr, ac)
                b = world.reference(fr, ac, belief, ha)[:, 14:15].float()
                if mode == 'reference':
                    nxt = F.layer_norm(b[:, 0], (192,))
                else:
                    noise = DM.draw_noise((n, 16), g)[0].cuda(); gum = TM.draw_gumbel((n, 16), g)[0].cuda()
                    samp = world.complete(fr, ac, belief, ha, noise, gum)[0][:, 14].float()
                    if mode == 'sampled':
                        nxt = samp
                    else:
                        logits = world.completion.propose(world.completion.trunk(belief, noise)).float()[:, 14:15]
                        logits = logits.masked_fill(~world.completion.valid, -torch.inf)
                        z = real[:, 15 + k][:, None].cuda()
                        partner, _, _, _, _, _ = IC.infer(F.layer_norm(b, (192,)), z, IC.pair_prior_logit(logits))
                        forced = F.layer_norm(b.gather(2, partner[..., None].expand(-1, -1, -1, 192)), (192,))[:, 0]
                        use = ok.cuda()[:, None, None]
                        nxt = torch.where(use, forced, samp)
            nxt = nxt.float().cpu()
            est = shifts[scroll.estimate(hist[-1][:, None], nxt[:, None])[:, 0]]
            ok = ok & (est == rs[:, k - 1]).all(-1)
            hist.append(nxt)
            flat[s0 * K:s0 * K + n, k - 1] = nxt.half()
    return out


def main():
    import direct_train as BASE
    import transport_model as TM
    import transport_recursion as TR
    import mob_fading as MF
    ckpt = sys.argv[1]
    name = ckpt.replace('/', '_').replace('.pt', '')
    config = BASE.settings()
    world = TM.TransportWorld('operator').cuda().eval()
    world.load_state_dict(torch.load(A5.BASE / 'transport' / ckpt, map_location='cpu', weights_only=False)['world'])
    panel = torch.load(A5.BASE / 'panel.pt', map_location='cpu', weights_only=False, mmap=True)
    res = {'checkpoint': ckpt}
    for mode in ('posterior', 'sampled', 'reference'):
        imag = rollout(world, mode, panel, config)
        key = f'pforced_{name}_{mode}'
        torch.save(imag, A5.BASE / f'transport/recursion_samples_{key}.pt')
        r = TR.evaluate(key, imag)
        b = r['bright']
        res[mode] = {'cow_census_d1_d8_d16': [b['census_ratio[zombie,cow,skeleton]_by_depth'][d][1] for d in (0, 7, 15)],
                     'tile_seen_d1_d8_d16': [b['tile_error_seen_by_depth'][d] for d in (0, 7, 15)],
                     'camera_d8_d16': [b['camera_consistent_by_depth'][d] for d in (7, 15)]}
        print(json.dumps({mode: res[mode]}), flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f'posterior_forced_{name}.json').write_text(json.dumps(res, indent=1))


if __name__ == '__main__':
    main()
