"""R49 recursion check: is the token appearance operator exact at the identity, and stable when applied repeatedly? (2026-10-16)

R21: transformations that are not idempotent fade content under recursion. If g(t, identity) != t, applying the operator at every
imagined step erodes copies, so it would have to be gated to appearance events. On R47's 200 test roots (awake root frames):
apply g(., theta_identity) k = 1, 2, 4, 8, 16 times; token drift from the original and the frozen readers' agreement with the
original's tile / zombie / cow reading (map cells). Also: sleep theta once, then identity k more times (stability of a transformed
state).
Usage: appearance_identity.py <depth> <seed> [tag, e.g. _id0.2] -> artifacts/eda/whiteboard/appearance_identity_d<depth>_s<seed>.json
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

import appearance_op as AO
import recontext_stale as RS
import sleep_transform as ST

KS = (1, 2, 4, 8, 16)


@torch.no_grad()
def main():
    import direct_coherence as C
    import audit_s5_direct as A5
    depth, seed = int(sys.argv[1]), int(sys.argv[2])
    enc = RS.Encoder()
    op = AO.Operator(depth).cuda().eval()
    tag = sys.argv[3] if len(sys.argv) > 3 else ''
    op.load_state_dict(torch.load(AO.OUT / f'appearance_op_d{depth}_s{seed}{tag}.pt', map_location='cuda', weights_only=False))
    bank = torch.load(A5.BASE / 'snap_bank_readers.pt', map_location='cpu', weights_only=False)
    head = torch.load(A5.BASE / 'learned_innovation/eval_u3000_h01/mob_heads.pt', map_location='cpu', weights_only=False)['heads']['real']
    pairs = []
    for f in sorted(ST.DEV.glob('seed-*.pt')):
        for r in torch.load(f, weights_only=False):
            root, nxt = r['frames'][-1], r['depth_frames'][6][0]
            if not ST.asleep(root) and ST.asleep(nxt):
                pairs.append((root, nxt))
    pick = torch.randperm(len(pairs), generator=torch.Generator().manual_seed(20261034))[:600].tolist()
    test = [pairs[i] for i in pick][400:]
    roots = torch.stack([a for a, _ in test])
    t0 = torch.cat([enc.recontext(enc.content_px(roots[i:i + 50].float() / 255.0)).float() for i in range(0, 200, 50)])
    th_id = torch.cat([torch.eye(3).flatten(), torch.zeros(3)]).cuda()[None].expand(len(t0), -1)
    th_sl = AO.sleep_theta().cuda()[None].expand(len(t0), -1)

    def agree(ref, tok):
        dr, dp = C.decode(ref.cpu(), bank, head), C.decode(tok.cpu(), bank, head)
        v = dr['tile'][:, :63] >= 0
        out = {'token_drift': round(float((tok[:, :63] - ref[:, :63]).norm(dim=-1).mean()), 3),
               'tile_agreement': round(float((dp['tile'][:, :63][v] == dr['tile'][:, :63][v]).float().mean()), 4)}
        for ch, k in ((0, 'zombie'), (1, 'cow')):
            rm, pm = dr['mob']['real'][:, :63, ch], dp['mob']['real'][:, :63, ch]
            out[f'{k}_recall'] = round(float((rm & pm).sum() / rm.sum().clamp_min(1)), 4)
            out[f'{k}_precision'] = round(float((rm & pm).sum() / pm.sum().clamp_min(1)), 4)
        return out

    res = {'roots': len(t0), 'identity_repeated': {}, 'sleep_then_identity': {}}
    x = t0.clone()
    for k in range(1, max(KS) + 1):
        x = torch.cat([op(x[i:i + 50], th_id[i:i + 50]) for i in range(0, 200, 50)])
        if k in KS:
            res['identity_repeated'][k] = agree(t0, x)
    s1 = torch.cat([op(t0[i:i + 50], th_sl[i:i + 50]) for i in range(0, 200, 50)])
    x = s1.clone()
    for k in range(1, max(KS) + 1):
        x = torch.cat([op(x[i:i + 50], th_id[i:i + 50]) for i in range(0, 200, 50)])
        if k in KS:
            res['sleep_then_identity'][k] = agree(s1, x)
    print(json.dumps(res, indent=1))
    (AO.OUT / f'appearance_identity_d{depth}_s{seed}{tag}.json').write_text(json.dumps(res, indent=1))


if __name__ == '__main__':
    main()
