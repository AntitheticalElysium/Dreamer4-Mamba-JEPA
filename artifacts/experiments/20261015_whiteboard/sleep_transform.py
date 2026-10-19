"""Can a global APPEARANCE transformation preserve content? The sleep case (2026-10-16; R46 follow-up; GPU inference)

Craftax-Classic source (renderer.py:440-450): asleep, map pixels -> 0.5 * luminance(pixels) + (0, 0, 8): a fixed per-pixel map of
the content. The frozen encoder's content code (position-free patch embedding) is linear in pixels, so in content-code space the
sleep transform is linear up to 8-bit rounding, and the encoder's upper layers recompute context (recontext_stale.Encoder).
Data: deeppanel DEV roots (seeded 600; fit on the first 400, test on the last 200) whose key-0 frame one step after first action
SLEEP is rendered asleep (near-grey map) and whose root is awake. Pairs: root frame (awake) -> depth-1 frame (asleep); no scroll
while sleeping. Variants for the asleep frame's tokens:
  copy        the awake root tokens (PI's copy)
  oracle      the source's sleep function applied to the ROOT pixels (map rows), embedded and recontextualized (HUD from the real
              asleep frame): persist content, transform appearance exactly, recompute context
  lin_<l>     an affine map on content codes (192 x 192 + bias), least squares with ridge l * N on fit-root cell pairs (all 63
              map cells, mobs that moved included; no labels), applied to the root's content codes, recontextualized: a learned
              appearance operator; lin_exact_targets: the same fit with the source-rendered targets (checks the theory below); lin_trimmed:
              label-free trimmed least squares on the real pairs (3 rounds keeping the 80% best-fitting cells: an EM-like split of
              persistent cells from cells whose content changed, e.g. mobs that moved); colour_map: one 3x3 + bias colour map shared by every
              pixel (12 parameters), plain least squares on all fit cells in the pixel coordinates that the frozen patch embedding
              inverts exactly (pinv; no decoder, no reconstruction loss). Theory: the patch embedding is affine in pixels (conv 7x7x3 -> 192) and the sleep render is
              affine per pixel, so if the embedding is injective (rank 147) the exact operator is affine in content-code space.
  tok_film / tok_free   operators on PI's own Raw tokens instead (no pixel-equivalent coordinates): a per-dimension affine shared
              by every cell (FiLM, 384 parameters) / a full 192 x 192 affine shared by every cell; least squares on the fit roots'
              map-cell token pairs, LN after; the reconstruction-free alternative
Scored on test roots vs the REAL asleep frame (token error, frozen readers: tile agreement, zombie / cow recall and precision) and,
for content preservation without the confound of mobs moving during the step, vs the ORACLE's decoded content.
Usage: sleep_transform.py -> artifacts/eda/whiteboard/sleep_transform.json
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

import recontext_stale as RS
import audit_s5_direct as A5
import direct_coherence as C

DEV = HERE.parents[2] / 'artifacts/eda/deeppanel_dev_v1'
MAP_PX = 49
RIDGE = (0.0, 1e-6, 1e-4, 1e-3)
TRIM_ROUNDS, TRIM_KEEP = 3, 0.8


def sleep_pixels(frame):
    """uint8 [63,63,3] -> float pixels in [0,255] with the map rows rendered asleep (renderer.py:441-450), HUD unchanged"""
    x = frame.float().clone()
    m = x[:MAP_PX]
    lum = 0.299 * m[..., 0] + 0.587 * m[..., 1] + 0.114 * m[..., 2]
    x[:MAP_PX] = (0.5 * lum)[..., None] + 0.5 * torch.tensor([0., 0., 16.])
    return x.round().clamp(0, 255)


def asleep(frame):
    m = frame[:MAP_PX].float()
    return bool(((m[..., 0] - m[..., 1]).abs().max() <= 2) and ((m[..., 2] - m[..., 0]).mean() > 4))


@torch.no_grad()
def main():
    enc = RS.Encoder()
    bank = torch.load(A5.BASE / 'snap_bank_readers.pt', map_location='cpu', weights_only=False)
    head = torch.load(A5.BASE / 'learned_innovation/eval_u3000_h01/mob_heads.pt', map_location='cpu', weights_only=False)['heads']['real']
    pairs = []
    for f in sorted(DEV.glob('seed-*.pt')):
        for r in torch.load(f, weights_only=False):
            root, nxt = r['frames'][-1], r['depth_frames'][6][0]
            if not asleep(root) and asleep(nxt):
                pairs.append((root, nxt))
    pick = torch.randperm(len(pairs), generator=torch.Generator().manual_seed(20261034))[:600].tolist()
    n_eligible = len(pairs)
    pairs = [pairs[i] for i in pick]
    fit, test = pairs[:400], pairs[400:]
    Wpe = enc.bb.embeddings.patch_embeddings.projection.weight.detach().float().flatten(1)       # [192, 147]
    sv = torch.linalg.svdvals(Wpe.double().cpu())
    meta = {'eligible_pairs': n_eligible, 'fit': len(fit), 'test': len(test), 'patch_embed_shape': list(Wpe.shape),
            'patch_embed_rank': int((sv > sv[0] * 1e-6).sum()), 'patch_embed_cond': float(sv[0] / sv[-1])}
    print(json.dumps(meta), flush=True)
    to_px = lambda fr: fr.float() / 255.0
    X, Y, Yx, T0, T1 = [], [], [], [], []
    for root, nxt in fit:
        X.append(enc.content_px(to_px(root)[None])[0, :63].float().cpu()); Y.append(enc.content_px(to_px(nxt)[None])[0, :63].float().cpu())
        T0.append(enc.recontext(enc.content_px(to_px(root)[None]))[0, :63].float().cpu())
        T1.append(enc.recontext(enc.content_px(to_px(nxt)[None]))[0, :63].float().cpu())
        Yx.append(enc.content_px((sleep_pixels(root) / 255.0)[None])[0, :63].float().cpu())
    X = torch.cat(X).double(); Y = torch.cat(Y).double(); Yx = torch.cat(Yx).double()
    Xa = torch.cat([X, torch.ones(len(X), 1, dtype=torch.float64)], 1)
    ops = {'lin_exact_targets': (torch.linalg.pinv(Xa, rtol=1e-6) @ Yx).float().cuda()}  # validates the affine theory; uses the source's render
    keep = torch.ones(len(Xa), dtype=torch.bool)                                  # label-free trimmed fit: drop the worst-fitting cells, refit
    for it in range(TRIM_ROUNDS):
        Wt = torch.linalg.pinv(Xa[keep], rtol=1e-6) @ Y[keep]
        resid = (Xa @ Wt - Y).norm(dim=-1)
        keep = resid <= resid.quantile(TRIM_KEEP)
    ops['lin_trimmed'] = (torch.linalg.pinv(Xa[keep], rtol=1e-6) @ Y[keep]).float().cuda()
    pe = enc.bb.embeddings.patch_embeddings.projection                             # per-pixel colour map in the pixel coordinates the
    Wf, bpe = pe.weight.detach().double().cpu().flatten(1), pe.bias.detach().double().cpu()   # frozen first layer inverts exactly
    Wpinv = torch.linalg.pinv(Wf)
    mu, sd = enc.enc.pixel_mean.flatten().double().cpu(), enc.enc.pixel_std.flatten().double().cpu()
    to_rgb = lambda c: (((c.double().cpu() - bpe) @ Wpinv.T).view(-1, 3, 49) * sd[None, :, None] + mu[None, :, None]).permute(0, 2, 1).reshape(-1, 3)
    from_rgb = lambda q: ((q.view(-1, 49, 3).permute(0, 2, 1) - mu[None, :, None]) / sd[None, :, None]).reshape(-1, 147) @ Wf.T + bpe
    Pa = torch.cat([to_rgb(X), torch.ones(len(X) * 49, 1, dtype=torch.float64)], 1)
    Mc = torch.linalg.lstsq(Pa, to_rgb(Y)).solution                              # [4, 3]: 12 parameters shared by every pixel
    meta['colour_map_fit'] = [[round(float(v), 4) for v in row] for row in Mc.T]
    meta['colour_map_source'] = [[0.5 * 0.299, 0.5 * 0.587, 0.5 * 0.114, 0.0], [0.5 * 0.299, 0.5 * 0.587, 0.5 * 0.114, 0.0],
                                 [0.5 * 0.299, 0.5 * 0.587, 0.5 * 0.114, round(8 / 255, 4)]]
    ops['colour_map'] = lambda c: from_rgb(torch.cat([to_rgb(c), torch.ones(len(c) * 49, 1, dtype=torch.float64)], 1) @ Mc).float().cuda()
    moved = (Y - Yx).norm(dim=-1) > (Y - Yx).norm(dim=-1).median() * 4                        # cells whose real next content is not the
    meta['fit_cells'] = len(Xa); meta['fit_cells_changed_share'] = round(float(moved.float().mean()), 4)  # exact transform of the root's
    meta['trim_drops_changed_recall'] = round(float((~keep & moved).sum() / moved.sum().clamp_min(1)), 4)
    meta['trim_drops_unchanged_share'] = round(float((~keep & ~moved).sum() / (~moved).sum()), 4)
    for lam in RIDGE:
        if lam == 0:                                                              # content codes span <= 148 affine dims: min-norm solution
            ops['lin_0'] = (torch.linalg.pinv(Xa, rtol=1e-10) @ Y).float().cuda()
        else:
            ops[f'lin_{lam:g}'] = torch.linalg.solve(Xa.T @ Xa + lam * len(Xa) * torch.eye(193, dtype=torch.float64), Xa.T @ Y).float().cuda()
    meta['content_code_rank_fit'] = int((torch.linalg.svdvals(Xa) > torch.linalg.svdvals(Xa)[0] * 1e-8).sum())
    T0 = torch.cat(T0).double(); T1 = torch.cat(T1).double()                        # token-space operators on PI's own (Raw) tokens
    film_a = ((T0 - T0.mean(0)) * (T1 - T1.mean(0))).mean(0) / (T0 - T0.mean(0)).pow(2).mean(0)
    film_b = T1.mean(0) - film_a * T0.mean(0)
    Wt = torch.linalg.pinv(torch.cat([T0, torch.ones(len(T0), 1, dtype=torch.float64)], 1), rtol=1e-6) @ T1
    tok_ops = {'tok_film': lambda t: t * film_a.float().cuda() + film_b.float().cuda(),
               'tok_free': lambda t: torch.cat([t, torch.ones(len(t), 1, device=t.device)], 1) @ Wt.float().cuda()}
    names = ['copy', 'oracle'] + list(ops) + list(tok_ops)
    keys = ('err', 'n', 'tile_ok', 'tile_n', 'z_hit', 'z_real', 'z_pred', 'c_hit', 'c_real', 'c_pred')
    tally = {ref: {v: dict.fromkeys(keys, 0) for v in names} for ref in ('real', 'oracle')}
    for root, nxt in test:
        c_root = enc.content_px(to_px(root)[None])[0].float()
        c_next = enc.content_px(to_px(nxt)[None])[0].float()
        c_or = enc.content_px((sleep_pixels(root) / 255.0)[None])[0].float()
        c_or[63:] = c_next[63:]
        toks = {'copy': enc.recontext(c_root[None])[0].float(), 'oracle': enc.recontext(c_or[None])[0].float()}
        for name, op in ops.items():
            c_li = c_next.clone()
            c_li[:63] = op(c_root[:63]) if callable(op) else torch.cat([c_root[:63], torch.ones(63, 1, device=c_root.device)], 1) @ op
            toks[name] = enc.recontext(c_li[None])[0].float()
        refs = {'real': enc.recontext(c_next[None])[0].float(), 'oracle': toks['oracle']}
        for name, op in tok_ops.items():
            t = refs['real'].clone(); t[:63] = F.layer_norm(op(toks['copy'][:63]), (192,)); toks[name] = t
        dec = {name: C.decode(tok[None].cpu(), bank, head) for name, tok in toks.items()}
        dec_real = C.decode(refs['real'][None].cpu(), bank, head)
        for ref, rtok in refs.items():
            dr = dec_real if ref == 'real' else dec['oracle']
            for name in names:
                a, dp = tally[ref][name], dec[name]
                a['err'] += float((toks[name][:63] - rtok[:63]).norm(dim=-1).mean()); a['n'] += 1
                v = dr['tile'][0] >= 0
                a['tile_ok'] += int((dp['tile'][0][v] == dr['tile'][0][v]).sum()); a['tile_n'] += int(v.sum())
                for ch, k in ((0, 'z'), (1, 'c')):
                    rm, pm = dr['mob']['real'][0, :63, ch], dp['mob']['real'][0, :63, ch]
                    a[f'{k}_hit'] += int((rm & pm).sum()); a[f'{k}_real'] += int(rm.sum()); a[f'{k}_pred'] += int(pm.sum())
    res = {'meta': meta}
    for ref, tal in tally.items():
        res[f'vs_{ref}'] = {}
        for name, a in tal.items():
            res[f'vs_{ref}'][name] = {'token_error': round(a['err'] / a['n'], 3), 'tile_agreement': round(a['tile_ok'] / max(a['tile_n'], 1), 4),
                                      'zombie_recall': round(a['z_hit'] / max(a['z_real'], 1), 4), 'zombie_precision': round(a['z_hit'] / max(a['z_pred'], 1), 4),
                                      'zombies': a['z_real'], 'cow_recall': round(a['c_hit'] / max(a['c_real'], 1), 4),
                                      'cow_precision': round(a['c_hit'] / max(a['c_pred'], 1), 4), 'cows': a['c_real']}
    print(json.dumps(res, indent=1))
    (HERE.parents[1] / 'eda/whiteboard/sleep_transform.json').write_text(json.dumps(res, indent=1))


if __name__ == '__main__':
    main()
