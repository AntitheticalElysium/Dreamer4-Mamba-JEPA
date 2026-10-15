"""Does PI regenerate the cell the player vacates? (2026-10-16; CPU inference; label-free scroll; diagnostic)

After a move, the cell whose copy source is the player cell 31 reveals terrain the player sprite occluded; an exact copy plants a
ghost player there. On 320 seeded training windows (as stale_context.py), at every transition with a scroll: PI u3000's
training-time edit posterior (q > 0.5, as R34's stale construction used) and its prior gate (sigmoid(gate) and MAP gate > 0, as
recursion uses) at the vacated cell, vs at other non-entering map cells.
Usage: vacated_cell.py -> artifacts/eda/whiteboard/vacated_cell.json
"""
import dataclasses, json, os, sys
from pathlib import Path
os.environ.setdefault('TRITON_F32_DEFAULT', 'ieee')
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / '20261012_oracle')); sys.path.insert(0, str(HERE))
import torch
from torch.nn import functional as F
import audit_s5_direct as A5, direct_model as DM, direct_train as BASE, transport_ic_train as IC, pi_model as PI
from d4mj.mamba_recurrence import FunctionalMamba2

@torch.no_grad()
def main():
    torch.set_num_threads(16)
    _, windows, actions, tokens = BASE.load_data()
    sel = torch.randperm(len(windows), generator=torch.Generator().manual_seed(20261023))[:320]
    scroll = A5.load_scroll()
    vac = {s: (3 - dr) * 9 + 4 - dc for s, (dr, dc) in enumerate(PI.SCROLL_OFFSETS) if s}
    res = {}
    for seed in (1, 2):
        w = PI.PIWorld().eval()
        w.load_state_dict(torch.load(A5.BASE / f'transport/pi_seed{seed}/pi_u3000.pt', map_location='cpu', weights_only=False)['world'])
        for m_ in w.modules():
            if isinstance(m_, FunctionalMamba2):
                m_.settings = dataclasses.replace(m_.settings, backend='reference')
        g = torch.Generator().manual_seed(3)
        t = {'vac_n': 0, 'vac_post': 0, 'vac_map': 0, 'vac_prob': 0., 'oth_n': 0, 'oth_map': 0, 'oth_prob': 0.}
        for b0 in range(0, 320, 8):
            frames, ac, _ = BASE.batch(tokens, actions, windows, sel[b0:b0 + 8], device='cpu')
            noise = DM.draw_noise(frames.shape[:2], g)[0]
            est = scroll.estimate(frames[:, :-1], frames[:, 1:])
            belief, ha = w.backbone_full(frames, ac)
            copy, gen, glogit, enter, sl = w.afterstate_parts(frames, ac, belief, ha, scroll=F.pad(est, (0, 1)))
            logits = w.completion.propose(w.completion.trunk(belief, noise)).float()
            copy, gen, glogit, enter, logits = (x[:, :15] for x in (copy, gen, glogit, enter, logits))
            logits = logits.masked_fill(~w.completion.valid, -torch.inf)
            z = frames[:, 1:]
            a0 = torch.where(((glogit > 0) | enter)[..., None], gen, copy)
            partner, _, _, _, _, _ = IC.infer(F.layer_norm(a0, (192,)), z, IC.pair_prior_logit(logits))
            z_pre = z.gather(2, partner[..., None].expand(-1, -1, -1, 192))
            res_copy = (F.layer_norm(copy, (192,)) - z_pre).norm(dim=-1)
            gain = res_copy - (F.layer_norm(gen, (192,)) - z_pre).norm(dim=-1)
            s_ = res_copy[..., :PI.MAP].mean(-1, keepdim=True).clamp_min(1e-6)
            post = torch.sigmoid(gain / s_ + glogit) > .5
            # own scroll decision in recursion: argmax; use est here (teacher-forced) for the cell identity
            own = w.scroll_logits(ac, belief, ha)[:, :15].argmax(-1)
            for i in range(len(frames)):
                for j in range(15):
                    s = int(est[i, j])
                    if s == 0:
                        continue
                    x = vac[s]
                    t['vac_n'] += 1; t['vac_post'] += int(post[i, j, x]); t['vac_map'] += int(glogit[i, j, x] > 0); t['vac_prob'] += float(torch.sigmoid(glogit[i, j, x]))
                    oth = [y for y in range(PI.MAP) if y not in (31, x) and not bool(enter[i, j, y])]
                    t['oth_n'] += len(oth); t['oth_map'] += int((glogit[i, j, oth] > 0).sum()); t['oth_prob'] += float(torch.sigmoid(glogit[i, j, oth]).sum())
        res[f'pi_seed{seed}'] = {'scroll_transitions': t['vac_n'], 'vacated_posterior_edit': round(t['vac_post'] / t['vac_n'], 4),
                                 'vacated_map_gate_fires': round(t['vac_map'] / t['vac_n'], 4), 'vacated_gate_prob': round(t['vac_prob'] / t['vac_n'], 4),
                                 'other_map_gate_fires': round(t['oth_map'] / t['oth_n'], 4), 'other_gate_prob': round(t['oth_prob'] / t['oth_n'], 4)}
        print(json.dumps({f'pi_seed{seed}': res[f'pi_seed{seed}']}), flush=True)
    (HERE.parents[1] / 'eda/whiteboard/vacated_cell.json').write_text(json.dumps(res, indent=1))

if __name__ == '__main__':
    main()
