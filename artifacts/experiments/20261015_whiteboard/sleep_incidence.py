"""How often does the TRAINING data show a sleeper being attacked? (2026-10-16; R48 companion; readers as labels, evaluation only)

The training pool stores tokens, actions, rewards and alive flags, no state. Labels here come from frozen readers:
  asleep    a logistic probe on the mean map token, fitted on deeppanel DEV frames labelled by the pixel rule
            (sleep_transform.asleep: grey map with the source's blue offset); held-out accuracy reported
  health    the bank HUD reader (decision_proxies.hud_health)
  zombie adjacent   the real mob reader at the 4 cells next to the player
Over every frame of the pool (19,789 trajectories x 64 frames), transitions t -> t+1 with both frames alive:
  counts of asleep / asleep & zombie adjacent; P(health drop >= 0.5 at t+1 | asleep, zombie adjacent) vs awake; and the share
  of the 16-frame training windows (direct_train windows) containing at least one asleep-and-attacked transition.
Usage: sleep_incidence.py -> artifacts/eda/whiteboard/sleep_incidence.json (+ per-frame masks sleep_incidence_masks.pt)
"""
import json
import os
import sys
from pathlib import Path

os.environ.setdefault('TRITON_F32_DEFAULT', 'ieee')
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / '20261012_oracle'))
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / '20260927_levers'))

import numpy as np
import torch

import dv16 as DV
import sleep_transform as ST
from decision_proxies import hud_health

ADJ = [22, 30, 32, 40]


@torch.no_grad()
def main():
    import direct_train as BASE
    import direct_coherence as C
    import audit_s5_direct as A5
    torch.set_num_threads(16)
    bank = torch.load(A5.BASE / 'snap_bank_readers.pt', map_location='cpu', weights_only=False)
    head = torch.load(A5.BASE / 'learned_innovation/eval_u3000_h01/mob_heads.pt', map_location='cpu', weights_only=False)['heads']['real']
    enc = DV.RS.Encoder()
    feats, labels, groups = [], [], []
    rows = [r for f in sorted(DV.DEV.glob('seed-*.pt')) for r in torch.load(f, weights_only=False)]
    pick = torch.randperm(len(rows), generator=torch.Generator().manual_seed(20261041))[:400].tolist()
    for gi, i in enumerate(pick):
        r = rows[i]
        frames = torch.cat([r['frames'][-1:], r['depth_frames'][0], r['depth_frames'][6]])        # root, noop d1-16, sleep d1-16
        tok = enc.recontext(enc.content(frames)).float().cpu()
        feats.append(tok[:, :63].mean(1)); labels += [ST.asleep(fr) for fr in frames]; groups += [gi] * len(frames)
    X = torch.cat(feats); y = torch.tensor(labels, dtype=torch.float32); grp = torch.tensor(groups)
    fit = grp < 270
    w = torch.zeros(193, requires_grad=True)
    Xa = torch.cat([X, torch.ones(len(X), 1)], 1)
    opt = torch.optim.LBFGS([w], max_iter=500, line_search_fn='strong_wolfe')
    def closure():
        opt.zero_grad()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(Xa[fit] @ w, y[fit]) + 1e-4 * w[:192].pow(2).sum()
        loss.backward()
        return loss
    with torch.enable_grad():
        opt.step(closure)
    w = w.detach()
    pred = (Xa[~fit] @ w) > 0
    yt = y[~fit].bool()
    probe = {'frames_fit': int(fit.sum()), 'frames_test': int((~fit).sum()), 'asleep_share_test': round(float(yt.float().mean()), 4),
             'test_accuracy': round(float((pred == yt).float().mean()), 4),
             'test_recall_asleep': round(float(pred[yt].float().mean()), 4), 'test_false_positive_rate': round(float(pred[~yt].float().mean()), 4)}
    print(json.dumps({'probe': probe}), flush=True)
    del enc; torch.cuda.empty_cache()
    _, windows, actions, tokens = BASE.load_data()
    pool = torch.load(BASE.D.POOL / 'labels.pt', map_location='cpu', weights_only=False, mmap=True)
    alive = pool['alive']
    N = tokens.shape[0]
    asleep = torch.zeros(N, 64, dtype=torch.bool); health = torch.zeros(N, 64); zadj = torch.zeros(N, 64, dtype=torch.bool)
    for b0 in range(0, N, 64):
        t = torch.from_numpy(np.array(tokens[b0:b0 + 64])).float()                                  # [b,64,81,192]
        b = len(t)
        asleep[b0:b0 + b] = (torch.cat([t[:, :, :63].mean(2), torch.ones(b, 64, 1)], -1) @ w) > 0
        health[b0:b0 + b] = hud_health(t, bank['heads']['hud'])
        zadj[b0:b0 + b] = C.decode(t, bank, head)['mob']['real'][..., ADJ, 0].any(-1)
        if b0 % 3200 == 0:
            print(json.dumps({'trajectories_done': b0 + b}), flush=True)
    torch.save({'asleep': asleep, 'health': health.half(), 'zombie_adjacent': zadj, 'probe_w': w}, DV.OUT / 'sleep_incidence_masks.pt')
    ok = alive[:, :-1] & alive[:, 1:]
    drop = health[:, 1:] <= health[:, :-1] - .5
    s, z = asleep[:, :-1], zadj[:, :-1]
    cnt = lambda m: int((m & ok).sum())
    rate = lambda m: round(float((drop & m & ok).sum()) / max(cnt(m), 1), 4)
    attacked = s & z & drop & ok
    wins = torch.as_tensor(windows)
    win_has = torch.stack([attacked[int(r), int(st):int(st) + 15].any() for r, st in wins])
    win_sleep = torch.stack([(s & ok)[int(r), int(st):int(st) + 15].any() for r, st in wins])
    res = {'probe': probe, 'transitions': cnt(torch.ones_like(s)), 'asleep': cnt(s), 'asleep_zombie_adjacent': cnt(s & z),
           'awake_zombie_adjacent': cnt(~s & z), 'P_drop_asleep_zombie_adjacent': rate(s & z), 'P_drop_awake_zombie_adjacent': rate(~s & z),
           'P_drop_asleep_no_zombie': rate(s & ~z), 'P_drop_awake_no_zombie': rate(~s & ~z),
           'asleep_attacked_transitions': int(attacked.sum()), 'all_drops': int((drop & ok).sum()),
           'windows': len(wins), 'windows_with_asleep': int(win_sleep.sum()), 'windows_with_asleep_attacked': int(win_has.sum())}
    print(json.dumps(res, indent=1))
    (DV.OUT / 'sleep_incidence.json').write_text(json.dumps(res, indent=1))


if __name__ == '__main__':
    main()
