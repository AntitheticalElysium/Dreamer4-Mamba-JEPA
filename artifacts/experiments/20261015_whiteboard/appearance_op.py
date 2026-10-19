"""R49: a reconstruction-free appearance operator? IWM-style token-space operator trained on synthetic recolourings (2026-10-16; GPU)

R47: the sleep change is a 12-parameter colour map on the pixel-linear content code, but no shared affine map on Raw tokens keeps
content, and operators fitted on REAL pairs absorb the step's content changes. Image World Models (Garrido et al. 2024, arXiv
2403.00504) learn photometric operators in latent space from SYNTHETIC, parameter-conditioned pairs (feature conditioning; MRR 0.79-
0.85 with an 18-layer predictor, 0.09-0.25 with 12 layers on weak / mixed augmentation, 0 unconditioned). Here, with the frozen Raw
encoder:
  data      pixels of diagnosis_futures_v1 frames (seeds 900000-900143; no overlap with deeppanel DEV 15000-16500); 90% of files
            fit, 10% held out. A random per-pixel colour map theta (12 numbers) is applied to the map rows (HUD unchanged, as the
            game renders), rounded to 8 bits: generic family M = b * (s I + (1 - s) L) (L: luminance projection), bias t; s ~ U(0,
            1.2) with grayscale (s = 0) at p 0.3, b ~ U(0.4, 1.3), t ~ U(-0.1, 0.15)^3. The source's sleep map (s 0, b 0.5, t (0, 0,
            8/255)) and its daylight blends lie inside the family; nothing is fitted to them.
  operator  g(tokens [81,192], theta) -> tokens: theta appended to every token (feature conditioning), linear to width 256,
            a pre-LN transformer of depth D (4 heads), linear back, residual on the input token, LN. MSE on all 81 cells.
            AdamW 3e-4, cosine, 6,000 updates, batch 64.
Evaluation:
  synthetic held-out pairs: token error of g and of copy
  sleep: R47's 200 test roots (sleep_transform's seeded DEV pairs), g(awake root tokens, source sleep theta) vs the ORACLE
         (the source's sleep render of the root, recontextualized): tile agreement, zombie / cow recall and precision (frozen
         readers); the colour map (0.996 / 0.974 / 0.976) and the free token operator (0.966 / 0.64 / 0.82) are R47's references
Env APPOP_IDENTITY=p: a share p of draws is the exact identity (target = source tokens), output tag _id<p> (R49 recursion check).
Declared before running: feasible if zombie AND cow recall vs oracle >= 0.90 and tile agreement >= 0.98.
Usage: appearance_op.py <depth> <seed> -> artifacts/eda/whiteboard/appearance_op_d<depth>_s<seed>.json
"""
import json
import math
import os
import sys
from pathlib import Path

os.environ.setdefault('TRITON_F32_DEFAULT', 'ieee')
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / '20261012_oracle'))
sys.path.insert(0, str(HERE))

import torch
from torch import nn
from torch.nn import functional as F

import recontext_stale as RS
import sleep_transform as ST

ROOT = HERE.parents[2]
DIAG = ROOT / 'artifacts/eda/diagnosis_futures_v1'
OUT = ROOT / 'artifacts/eda/whiteboard'
LUM = torch.tensor([0.299, 0.587, 0.114])
UPDATES, BATCH, WIDTH = 6000, 64, 256
P_IDENTITY = float(os.environ.get('APPOP_IDENTITY', 0))
TAG = f"_id{os.environ['APPOP_IDENTITY']}" if os.environ.get('APPOP_IDENTITY') else ''


def sample_theta(n, g):
    """[n,12]: 3x3 colour matrix (row-major) + bias, in [0,1] pixel units"""
    s = torch.rand(n, generator=g) * 1.2
    s = torch.where(torch.rand(n, generator=g) < 0.3, torch.zeros_like(s), s)
    b = 0.4 + 0.9 * torch.rand(n, generator=g)
    t = -0.1 + 0.25 * torch.rand(n, 3, generator=g)
    L = LUM[None, :].expand(3, 3)
    M = b[:, None, None] * (s[:, None, None] * torch.eye(3) + (1 - s)[:, None, None] * L)
    theta = torch.cat([M.flatten(1), t], 1)
    ident = torch.rand(n, generator=g) < P_IDENTITY
    return torch.where(ident[:, None], torch.cat([torch.eye(3).flatten(), torch.zeros(3)])[None], theta)


def sleep_theta():
    M = 0.5 * LUM[None, :].expand(3, 3)
    return torch.cat([M.flatten(), torch.tensor([0., 0., 8 / 255])])


def recolour(frames, theta):
    """uint8 [n,63,63,3], theta [n,12] -> float pixels in [0,1], map rows recoloured and 8-bit rounded"""
    x = frames.float() / 255.0
    M, t = theta[:, :9].view(-1, 3, 3), theta[:, 9:]
    m = torch.einsum('nhwc,ndc->nhwd', x[:, :ST.MAP_PX], M) + t[:, None, None, :]
    x[:, :ST.MAP_PX] = (m * 255).round().clamp(0, 255) / 255.0
    return x


class Operator(nn.Module):
    def __init__(self, depth):
        super().__init__()
        self.inp = nn.Linear(192 + 12, WIDTH)
        self.pos = nn.Parameter(torch.zeros(81, WIDTH))
        layer = nn.TransformerEncoderLayer(WIDTH, 4, 4 * WIDTH, dropout=0.0, batch_first=True, norm_first=True)
        self.body = nn.TransformerEncoder(layer, depth)
        self.out = nn.Linear(WIDTH, 192)
        nn.init.zeros_(self.out.weight); nn.init.zeros_(self.out.bias)

    def forward(self, tok, theta):
        x = self.inp(torch.cat([tok, theta[:, None, :].expand(-1, 81, -1)], -1)) + self.pos
        return F.layer_norm(tok + self.out(self.body(x)), (192,))


def load_frames():
    files = sorted(DIAG.glob('seed-*.pt'))
    held = set(files[::10])
    fit, val = [], []
    for f in files:
        for r in torch.load(f, weights_only=False):
            fr = torch.cat([r['context'], r['future_frames'].flatten(0, 1)])
            (val if f in held else fit).append(fr)
    return torch.cat(fit), torch.cat(val)


@torch.no_grad()
def encode(enc, px):
    return torch.cat([enc.recontext(enc.content_px(px[i:i + 128])).float() for i in range(0, len(px), 128)])


def main():
    depth, seed = int(sys.argv[1]), int(sys.argv[2])
    torch.manual_seed(seed)
    enc = RS.Encoder()
    fit, val = load_frames()
    print(json.dumps({'fit_frames': len(fit), 'val_frames': len(val)}), flush=True)
    op = Operator(depth).cuda()
    opt = torch.optim.AdamW(op.parameters(), lr=3e-4, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda u: min(1.0, (u + 1) / 300) * 0.5 * (1 + math.cos(math.pi * u / UPDATES)))
    g = torch.Generator().manual_seed(seed)
    for u in range(UPDATES):
        idx = torch.randint(len(fit), (BATCH,), generator=g)
        fr = fit[idx]; th = sample_theta(BATCH, g)
        with torch.no_grad():
            src = enc.recontext(enc.content_px(fr.float() / 255.0)).float()
            tgt = enc.recontext(enc.content_px(recolour(fr, th))).float()
        loss = F.mse_loss(op(src, th.cuda()), tgt)
        opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(op.parameters(), 1.0); opt.step(); sched.step()
        if u % 500 == 0 or u == UPDATES - 1:
            print(json.dumps({'update': u, 'loss': round(float(loss), 5)}), flush=True)
    op.eval()
    res = {'depth': depth, 'seed': seed, 'updates': UPDATES}
    with torch.no_grad():
        gv = torch.Generator().manual_seed(7)
        idx = torch.randperm(len(val), generator=gv)[:1024]
        th = sample_theta(len(idx), gv)
        src = encode(enc, val[idx].float() / 255.0); tgt = encode(enc, recolour(val[idx], th))
        pred = torch.cat([op(src[i:i + 128], th[i:i + 128].cuda()) for i in range(0, len(src), 128)])
        res['synthetic_heldout'] = {'pairs': len(idx), 'copy_error': round(float((src - tgt).norm(dim=-1).mean()), 3),
                                    'operator_error': round(float((pred - tgt).norm(dim=-1).mean()), 3)}
        import direct_coherence as C
        import audit_s5_direct as A5
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
        ths = sleep_theta()[None].cuda()
        a = dict.fromkeys(('tile_ok', 'tile_n', 'z_hit', 'z_ref', 'z_pred', 'c_hit', 'c_ref', 'c_pred', 'err', 'err_copy'), 0.)
        for root, nxt in test:
            src = enc.recontext(enc.content_px(root[None].float() / 255.0)).float()
            c_or = enc.content_px((ST.sleep_pixels(root) / 255.0)[None])[0]
            c_or[63:] = enc.content_px(nxt[None].float() / 255.0)[0, 63:]                 # the oracle of R47 (real asleep HUD)
            oracle = enc.recontext(c_or[None]).float()
            pred = op(src, ths)
            pred[:, 63:] = oracle[:, 63:]                                                    # map cells are judged, as in R47
            a['err'] += float((pred[0, :63] - oracle[0, :63]).norm(dim=-1).mean()); a['err_copy'] += float((src[0, :63] - oracle[0, :63]).norm(dim=-1).mean())
            dr, dp = C.decode(oracle.cpu(), bank, head), C.decode(pred.cpu(), bank, head)
            v = dr['tile'][0] >= 0
            a['tile_ok'] += int((dp['tile'][0][v] == dr['tile'][0][v]).sum()); a['tile_n'] += int(v.sum())
            for ch, k in ((0, 'z'), (1, 'c')):
                rm, pm = dr['mob']['real'][0, :63, ch], dp['mob']['real'][0, :63, ch]
                a[f'{k}_hit'] += int((rm & pm).sum()); a[f'{k}_ref'] += int(rm.sum()); a[f'{k}_pred'] += int(pm.sum())
        n = len(test)
        res['sleep_vs_oracle'] = {'roots': n, 'token_error': round(a['err'] / n, 3), 'copy_token_error': round(a['err_copy'] / n, 3),
                                  'tile_agreement': round(a['tile_ok'] / a['tile_n'], 4),
                                  'zombie_recall': round(a['z_hit'] / max(a['z_ref'], 1), 4), 'zombie_precision': round(a['z_hit'] / max(a['z_pred'], 1), 4),
                                  'zombies': int(a['z_ref']), 'cow_recall': round(a['c_hit'] / max(a['c_ref'], 1), 4),
                                  'cow_precision': round(a['c_hit'] / max(a['c_pred'], 1), 4), 'cows': int(a['c_ref'])}
        r = res['sleep_vs_oracle']
        res['feasible'] = bool(r['zombie_recall'] >= 0.9 and r['cow_recall'] >= 0.9 and r['tile_agreement'] >= 0.98)
    print(json.dumps(res, indent=1))
    (OUT / f'appearance_op_d{depth}_s{seed}{TAG}.json').write_text(json.dumps(res, indent=1))
    torch.save(op.state_dict(), OUT / f'appearance_op_d{depth}_s{seed}{TAG}.pt')


if __name__ == '__main__':
    main()
