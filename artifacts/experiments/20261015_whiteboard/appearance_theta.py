"""R49 last link: can the sleep onset's appearance parameters be INFERRED from real token pairs through the frozen token operator?
(2026-10-16; GPU; label-free, reconstruction-free)

The token operator g (appearance_op_d<depth>_s<seed>.pt) was trained on synthetic recolourings only and was given the source's sleep
theta. A world must infer theta. Here one theta (12 numbers, initialized at identity) is fitted by Adam (lr 0.01, 400 steps) to
minimize the MSE between g(awake root tokens, theta) and the REAL asleep tokens at the 63 map cells of R47's 400 fit roots (real
pairs: mobs that moved included, no labels, tokens only). Then on R47's 200 test roots, g(root, fitted theta) vs the oracle (as in
appearance_op.py): tile agreement, zombie / cow recall. Also the fitted theta vs the source's.
Usage: appearance_theta.py <depth> <seed> -> artifacts/eda/whiteboard/appearance_theta_d<depth>_s<seed>.json
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
from torch.nn import functional as F

import appearance_op as AO
import recontext_stale as RS
import sleep_transform as ST


def main():
    import direct_coherence as C
    import audit_s5_direct as A5
    depth, seed = int(sys.argv[1]), int(sys.argv[2])
    enc = RS.Encoder()
    op = AO.Operator(depth).cuda().eval()
    op.load_state_dict(torch.load(AO.OUT / f'appearance_op_d{depth}_s{seed}.pt', map_location='cuda', weights_only=False))
    for p in op.parameters():
        p.requires_grad_(False)
    pairs = []
    for f in sorted(ST.DEV.glob('seed-*.pt')):
        for r in torch.load(f, weights_only=False):
            root, nxt = r['frames'][-1], r['depth_frames'][6][0]
            if not ST.asleep(root) and ST.asleep(nxt):
                pairs.append((root, nxt))
    pick = torch.randperm(len(pairs), generator=torch.Generator().manual_seed(20261034))[:600].tolist()
    pairs = [pairs[i] for i in pick]
    fit, test = pairs[:400], pairs[400:]
    with torch.no_grad():
        src = torch.cat([enc.recontext(enc.content_px(torch.stack([a for a, _ in fit[i:i + 100]]).float() / 255.0)).float() for i in range(0, 400, 100)])
        tgt = torch.cat([enc.recontext(enc.content_px(torch.stack([b for _, b in fit[i:i + 100]]).float() / 255.0)).float() for i in range(0, 400, 100)])
    theta = torch.cat([torch.eye(3).flatten(), torch.zeros(3)]).cuda().requires_grad_(True)
    opt = torch.optim.Adam([theta], lr=0.01)
    for step in range(400):
        loss = 0.
        for i in range(0, 400, 100):
            out = op(src[i:i + 100], theta[None].expand(len(src[i:i + 100]), -1))
            loss = loss + F.mse_loss(out[:, :63], tgt[i:i + 100, :63]) / 4
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 100 == 0 or step == 399:
            print(json.dumps({'step': step, 'loss': round(float(loss), 5)}), flush=True)
    th = theta.detach()
    res = {'depth': depth, 'seed': seed, 'fitted_theta': [round(float(v), 4) for v in th.cpu()],
           'source_theta': [round(float(v), 4) for v in AO.sleep_theta()]}
    bank = torch.load(A5.BASE / 'snap_bank_readers.pt', map_location='cpu', weights_only=False)
    head = torch.load(A5.BASE / 'learned_innovation/eval_u3000_h01/mob_heads.pt', map_location='cpu', weights_only=False)['heads']['real']
    a = dict.fromkeys(('tile_ok', 'tile_n', 'z_hit', 'z_ref', 'c_hit', 'c_ref'), 0)
    with torch.no_grad():
        for root, nxt in test:
            s = enc.recontext(enc.content_px(root[None].float() / 255.0)).float()
            c_or = enc.content_px((ST.sleep_pixels(root) / 255.0)[None])[0]
            c_or[63:] = enc.content_px(nxt[None].float() / 255.0)[0, 63:]
            oracle = enc.recontext(c_or[None]).float()
            pred = op(s, th[None]); pred[:, 63:] = oracle[:, 63:]
            dr, dp = C.decode(oracle.cpu(), bank, head), C.decode(pred.cpu(), bank, head)
            v = dr['tile'][0] >= 0
            a['tile_ok'] += int((dp['tile'][0][v] == dr['tile'][0][v]).sum()); a['tile_n'] += int(v.sum())
            for ch, k in ((0, 'z'), (1, 'c')):
                rm, pm = dr['mob']['real'][0, :63, ch], dp['mob']['real'][0, :63, ch]
                a[f'{k}_hit'] += int((rm & pm).sum()); a[f'{k}_ref'] += int(rm.sum())
    res['sleep_vs_oracle_with_fitted_theta'] = {'tile_agreement': round(a['tile_ok'] / a['tile_n'], 4),
                                                'zombie_recall': round(a['z_hit'] / max(a['z_ref'], 1), 4), 'zombies': a['z_ref'],
                                                'cow_recall': round(a['c_hit'] / max(a['c_ref'], 1), 4), 'cows': a['c_ref']}
    print(json.dumps(res, indent=1))
    (AO.OUT / f'appearance_theta_d{depth}_s{seed}.json').write_text(json.dumps(res, indent=1))


if __name__ == '__main__':
    main()
