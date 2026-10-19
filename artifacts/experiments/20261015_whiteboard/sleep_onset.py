"""Does PI carry mobs through the sleep grey-out in ONE teacher-forced step? (2026-10-16; CPU inference; label-free model path,
frozen readers as labels)  Training-ledger windows; transitions t -> t+1 whose action at t is SLEEP (6) vs other actions. At each,
PI u3000's sampled one-step output (teacher-forced context, own scroll / gate / chance) and its deterministic afterstate are read by
the frozen real mob reader; recall of the REAL next frame's mobs (zombie / cow), share of map cells PI edits, and the mean
real-vs-predicted token error. Usage: sleep_onset.py -> prints"""
import dataclasses, json, os, sys
from pathlib import Path
os.environ.setdefault('TRITON_F32_DEFAULT', 'ieee')
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / '20261012_oracle')); sys.path.insert(0, str(HERE))
import torch
from torch.nn import functional as F
import audit_s5_direct as A5, direct_model as DM, direct_train as BASE, direct_coherence as C, transport_model as TM, pi_model as PI
from d4mj.mamba_recurrence import FunctionalMamba2

@torch.no_grad()
def main():
    torch.set_num_threads(16)
    _, windows, actions, tokens = BASE.load_data()
    a_all = torch.as_tensor(actions)
    w_all = torch.as_tensor(windows)
    has = torch.stack([(a_all[int(r), int(s):int(s) + 15] == 6).any() for r, s in w_all])
    cand = torch.where(has)[0]
    sel = cand[torch.randperm(len(cand), generator=torch.Generator().manual_seed(20261033))[:240]]
    bank = torch.load(A5.BASE / 'snap_bank_readers.pt', map_location='cpu', weights_only=False)
    head = torch.load(A5.BASE / 'learned_innovation/eval_u3000_h01/mob_heads.pt', map_location='cpu', weights_only=False)['heads']['real']
    w = PI.PIWorld().eval()
    w.load_state_dict(torch.load(A5.BASE / 'transport/pi_seed1/pi_u3000.pt', map_location='cpu', weights_only=False)['world'])
    for m_ in w.modules():
        if isinstance(m_, FunctionalMamba2):
            m_.settings = dataclasses.replace(m_.settings, backend='reference')
    g = torch.Generator().manual_seed(4)
    acc = {k: {'z_real': 0, 'z_hit': 0, 'c_real': 0, 'c_hit': 0, 'edits': 0., 'err': 0., 'n': 0} for k in ('sleep_onset', 'sleeping', 'other')}
    for b0 in range(0, len(sel), 8):
        frames, ac, _ = BASE.batch(tokens, actions, windows, sel[b0:b0 + 8], device='cpu')
        belief, ha = w.backbone_full(frames, ac)
        noise = DM.draw_noise(frames.shape[:2], g)[0]; gum = TM.draw_gumbel(frames.shape[:2], g)[0]
        out, aux = w.complete(frames, ac, belief, ha, noise, gum)
        real = frames[:, 1:].float(); pred = out[:, :15].float()
        dr = C.decode(real, bank, head)['mob']['real']; dp = C.decode(pred, bank, head)['mob']['real']
        edits = aux['edits'][:, :15, :63].float().mean(-1)
        for i in range(len(frames)):
            for t in range(15):
                prev_sleep = t > 0 and int(ac[i, t - 1]) == 6
                k = 'sleep_onset' if int(ac[i, t]) == 6 and not prev_sleep else ('sleeping' if int(ac[i, t]) == 6 else 'other')
                A = acc[k]; A['n'] += 1
                A['z_real'] += int(dr[i, t, :, 0].sum()); A['z_hit'] += int((dr[i, t, :, 0] & dp[i, t, :, 0]).sum())
                A['c_real'] += int(dr[i, t, :, 1].sum()); A['c_hit'] += int((dr[i, t, :, 1] & dp[i, t, :, 1]).sum())
                A['edits'] += float(edits[i, t]); A['err'] += float((pred[i, t] - real[i, t]).norm(dim=-1).mean())
    res = {k: {'transitions': A['n'], 'zombie_recall': round(A['z_hit'] / max(A['z_real'], 1), 4), 'zombies': A['z_real'],
               'cow_recall': round(A['c_hit'] / max(A['c_real'], 1), 4), 'cows': A['c_real'], 'map_edit_share': round(A['edits'] / A['n'], 4),
               'token_error': round(A['err'] / A['n'], 3)} for k, A in acc.items()}
    print(json.dumps(res, indent=1))

if __name__ == '__main__':
    main()
