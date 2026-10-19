"""Does literal token transport drag terrain along mob paths? (2026-10-15 night, user question on cross-terrain; CPU inference;
layouts diagnostic only)

A chance swap exchanges whole tokens. A mob's token has its background baked in (R2: the sprite occludes it; cross-terrain
transfer error ~1.1 vs 0.02-0.07 same-terrain), so when a mob moves onto different terrain the vacated cell receives the
destination's old token (wrong background) and the mob carries its old background. Same 400 training windows and true STAY-class
moves as ic_posterior.py pi, under PI u3000's training-time inference (scroll copy, posterior swaps, edit posterior). For every
true mob move CAPTURED by a swap, at the VACATED cell (source):
  cross     the real background revealed at t+1 (bank tile reader on the real z) differs from the destination's terrain at t
            (tile reader on the real frame t at the destination): the move crossed terrain
  swap_ok   PI's swapped-in token (no edit) reads as the revealed background
  fixed     PI's edit posterior regenerates the vacated cell (training-time, sees the real target) / the gate alone (MAP,
            prior: what recursion would do)
  final_ok  the cell PI outputs (edit ? gen : swapped copy) reads as the revealed background
Split cross / same terrain.
Usage: swap_background.py -> artifacts/eda/whiteboard/swap_background.json
"""
import dataclasses
import json
import os
import sys
from pathlib import Path

os.environ.setdefault('TRITON_F32_DEFAULT', 'ieee')
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / '20261012_oracle'))
sys.path.insert(0, str(HERE))

import numpy as np
import torch
from torch.nn import functional as F

import audit_s5_direct as A5
import direct_coherence as C
import direct_model as DM
import direct_train as BASE
import transport_ic_train as IC
import transport_stage2 as S2
import pi_model as PI
from d4mj.mamba_recurrence import FunctionalMamba2

OUT = HERE.parents[1] / 'eda/whiteboard'
INTERIOR = [r * 9 + c for r in range(1, 6) for c in range(1, 8)]


@torch.no_grad()
def main():
    torch.set_num_threads(8)
    _, windows, actions, tokens = BASE.load_data()
    valid = torch.from_numpy(np.load(A5.BASE / 'layouts_valid.npy'))
    windows = windows[valid[windows[:, 0]]]
    sel = torch.randperm(len(windows), generator=torch.Generator().manual_seed(20261016))[:400]
    lay = np.load(A5.BASE / 'layouts.npy', mmap_mode='r')
    bank = torch.load(A5.BASE / 'snap_bank_readers.pt', map_location='cpu', weights_only=False)
    head = torch.load(A5.BASE / 'learned_innovation/eval_u3000_h01/mob_heads.pt', map_location='cpu', weights_only=False)['heads']['real']
    scroll = A5.load_scroll()
    pos = set(INTERIOR)
    res = {}
    for seed in (1, 2):
        world = PI.PIWorld().eval()
        world.load_state_dict(torch.load(A5.BASE / f'transport/pi_seed{seed}/pi_u3000.pt', map_location='cpu', weights_only=False)['world'])
        for m_ in world.modules():
            if isinstance(m_, FunctionalMamba2):
                m_.settings = dataclasses.replace(m_.settings, backend='reference')
        g = torch.Generator().manual_seed(1)
        tally = {k: {'n': 0, 'swap_ok': 0, 'fixed_posterior': 0, 'fixed_gate': 0, 'final_ok': 0} for k in ('cross', 'same')}
        for b0 in range(0, 400, 8):
            frames, ac, choices = BASE.batch(tokens, actions, windows, sel[b0:b0 + 8], device='cpu')
            noise = DM.draw_noise(frames.shape[:2], g)[0]
            est = scroll.estimate(frames[:, :-1], frames[:, 1:])
            belief, ha = world.backbone_full(frames, ac)
            copy, gen, glogit, enter, _ = world.afterstate_parts(frames, ac, belief, ha, scroll=F.pad(est, (0, 1)))
            logits = world.completion.propose(world.completion.trunk(belief, noise)).float()
            copy, gen, glogit, enter, logits = (x[:, :15] for x in (copy, gen, glogit, enter, logits))
            logits = logits.masked_fill(~world.completion.valid, -torch.inf)
            z = frames[:, 1:]
            a0 = torch.where(((glogit > 0) | enter)[..., None], gen, copy)
            partner, _, _, _, _, _ = IC.infer(F.layer_norm(a0, (192,)), z, IC.pair_prior_logit(logits))
            z_pre = z.gather(2, partner[..., None].expand(-1, -1, -1, 192))
            res_copy = (F.layer_norm(copy, (192,)) - z_pre).norm(dim=-1)
            gain = res_copy - (F.layer_norm(gen, (192,)) - z_pre).norm(dim=-1)
            s = res_copy[..., :PI.MAP].mean(-1, keepdim=True).clamp_min(1e-6)
            edit = (torch.sigmoid(gain / s + glogit) > .5) | enter
            out_tok = F.layer_norm(torch.where(edit[..., None], gen, copy).gather(2, partner[..., None].expand(-1, -1, -1, 192)), (192,))
            swapped = F.layer_norm(copy.gather(2, partner[..., None].expand(-1, -1, -1, 192)), (192,))
            t_real = C.decode(z, bank, head)['tile']; t_prev = C.decode(frames[:, :15], bank, head)['tile']
            t_out = C.decode(out_tok, bank, head)['tile']; t_swap = C.decode(swapped, bank, head)['tile']
            dec = C.decode(F.layer_norm(a0, (192,)), bank, head)['mob']['real'][..., :3]
            for i, (row, start) in enumerate(choices.tolist()):
                L = torch.from_numpy(np.array(lay[row, start:start + 16]) > 0)
                for t in range(15):
                    prev = S2.afterstate_layout(L[t], *scroll.SHIFTS[int(est[i, t])])
                    for s_, d_, ty in S2.swaps(prev, L[t + 1]):
                        if ty >= 3 or s_ not in pos or d_ not in pos or 31 in (s_, d_):
                            continue
                        src, dst = (s_, d_) if bool(prev[s_, ty] == 1) else (d_, s_)
                        if not (bool(dec[i, t, src, ty]) and not bool(dec[i, t, dst, ty])) or int(partner[i, t, src]) != dst:
                            continue                                            # captured STAY-class moves only
                        # the destination's terrain at t, in the afterstate frame: the copy source of dst
                        revealed = int(t_real[i, t, src])
                        dest_terrain = int(t_swap[i, t, src])                       # what the swap puts at the vacated cell
                        if revealed < 0 or dest_terrain < 0:
                            continue
                        k = 'cross' if dest_terrain != revealed else 'same'
                        a = tally[k]; a['n'] += 1
                        a['swap_ok'] += int(dest_terrain == revealed)
                        a['fixed_posterior'] += int(bool(edit[i, t, src]))
                        a['fixed_gate'] += int(bool(glogit[i, t, src] > 0))
                        a['final_ok'] += int(int(t_out[i, t, src]) == revealed)
            if b0 % 80 == 0:
                print(json.dumps({'seed': seed, 'windows_done': b0 + 8, 'tally': tally}), flush=True)
        res[f'pi_seed{seed}'] = {k: {**v, **{f'{c}_share': round(v[c] / max(v['n'], 1), 4) for c in ('swap_ok', 'fixed_posterior', 'fixed_gate', 'final_ok')}}
                                 for k, v in tally.items()}
        print(json.dumps({f'pi_seed{seed}': res[f'pi_seed{seed}']}), flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'swap_background.json').write_text(json.dumps(res, indent=1))


if __name__ == '__main__':
    main()
