"""Persist content, recompute context: R34's causal test with candidate representations (2026-10-15 late, user programme
"persist information, not representation"; GPU inference, no training; exact simulator state used ONLY to define the true
transformations and the evaluation labels).

The frozen Raw encoder factors exactly (max |diff| 0.001) as  token = LN(Upper(PatchEmbed(pixels) + PosEmbed)), where
PatchEmbed is a local, position-free linear map of the tile's own 7x7 pixels (its CONTENT CODE) and Upper = the 12 attention
layers + final LN (the learned CONTEXT). Data: diagnosis_futures_v1 (pixels + exact visible state); window = root frame + 15
future frames of one future, the policy's own actions. True transformations per step from the exact state: camera shift (exact
tile maps), mob moves (exact layouts, transport_stage2.swaps), and CHANGED cells (entering, terrain edits, mob changes not
explained by a swap, arrows, the player cell when facing changes, HUD when any HUD / inventory value changes) which take the
fresh real content. The last m context frames (15-m..14) are rebuilt with the true transformations from the persisted
representation:
  T    Raw tokens copied forward (R34's stale tokens)
  C    content codes copied forward, then the whole frame RECONTEXTUALIZED by the encoder's upper layers at the cells' current
       positions and neighbours (contextual recomputation; content and its rendering stay as last seen)
  P    pixels copied forward (as C, done in pixel space; content codes are linear in pixels), embedded and recontextualized
  G    P with each stale frame's copied cells re-lit by ONE global affine colour transform (3x3 + offset, 12 numbers per frame)
       fitted from the copied pixels to the real frame's pixels over the copied map cells: an oracle for a global lighting
       STATE only (content and positions untouched), then embedded and recontextualized
  fresh  the real frames
(env RECONTEXT_RELIGHT=1 adds P and G and writes recontext_stale_relight.json)
PI (u3000, both seeds) reads each context; its belief-driven outputs at index 14 (the 14 -> 15 transition) are compared with
fresh on the same windows: chase toward-probability of exact zombies (distance <= 4), gate AUC for exact non-swap map changes,
gen error at exact terrain edits (target: the real next token), health-drop AUC of the gate at the health token, scroll
accuracy, belief cosine by cell class.
Splits: scrolls in the stale span (0 / >= 1) x rendering change in the stale span (STABLE: < 5% of persisting, unchanged cells
change pixels in every transition; else CHANGED), so position / context staleness and rendering staleness separate.
Usage: recontext_stale.py -> artifacts/eda/whiteboard/recontext_stale.json   (env RECONTEXT_ROOTS, default 240)
"""
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
import direct_model as DM
import direct_train as BASE
import transport_ic_train as IC
import transport_stage2 as S2
import pi_model as PI
import rep_layers as RL
from d4mj.train import autocast_context

OUT = HERE.parents[1] / 'eda/whiteboard'
T = A5.BASE / 'transport'
RELIGHT = os.environ.get('RECONTEXT_RELIGHT') == '1'
REPS = ('T', 'C', 'P', 'G') if RELIGHT else ('T', 'C')
DEPTHS = (4, 8, 14)
ADJ = (22, 30, 32, 40)
INTERIOR = set(r * 9 + c for r in range(1, 6) for c in range(1, 8))


def dist(x):
    r, c = divmod(x, 9)
    return abs(r - 3) + abs(c - 4)


class Encoder:
    """the frozen Raw encoder split into content codes and recontextualization"""
    def __init__(self):
        # the bundle's execution identity was recorded under PyTorch's default numerics; importing the training stack changes
        # them, so load under the defaults and restore the caller's settings afterwards
        saved = (torch.get_float32_matmul_precision(), torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32,
                 torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic, torch.are_deterministic_algorithms_enabled())
        torch.set_float32_matmul_precision('highest'); torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = True; torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = False
        torch.use_deterministic_algorithms(False)
        self.enc = RL.encoder().cuda().eval()
        torch.set_float32_matmul_precision(saved[0]); torch.backends.cuda.matmul.allow_tf32 = saved[1]
        torch.backends.cudnn.allow_tf32 = saved[2]; torch.backends.cudnn.benchmark = saved[3]; torch.backends.cudnn.deterministic = saved[4]
        torch.use_deterministic_algorithms(saved[5])
        self.bb = self.enc.backbone

    @torch.no_grad()
    def content(self, frames):                                                    # uint8 [n,63,63,3] -> [n,81,192]
        px = frames.cuda().permute(0, 3, 1, 2).float() / 255.0
        px = (px - self.enc.pixel_mean.cuda()) / self.enc.pixel_std.cuda()
        return self.bb.embeddings.patch_embeddings(px)

    @torch.no_grad()
    def content_px(self, px):                                                     # float pixels [n,63,63,3] in [0,1] -> [n,81,192]
        p = px.cuda().permute(0, 3, 1, 2)
        p = (p - self.enc.pixel_mean.cuda()) / self.enc.pixel_std.cuda()
        return self.bb.embeddings.patch_embeddings(p)

    @torch.no_grad()
    def recontext(self, c):                                                       # [n,81,192] -> LN tokens [n,81,192]
        emb = self.bb.embeddings
        x = torch.cat([emb.cls_token.expand(len(c), -1, -1), c], 1)
        x = emb.dropout(x + emb.interpolate_pos_encoding(x, 63, 63))
        for layer in self.bb.layers:
            x = layer(x, None)
        return F.layer_norm(self.bb.layernorm(x)[:, 1:].float(), (192,))


def facts(vis):
    v = vis.float()
    tiles = v[..., :1071].reshape(*v.shape[:-1], 63, 17).argmax(-1)
    mobs = v[..., 1071:1512].reshape(*v.shape[:-1], 63, 7) > 0
    return tiles, mobs, v[..., 1512:1534], v[..., 1516:1520]


def transitions(tiles, mobs, hud, facing, frames):
    """exact transformation of every step j-1 -> j: scroll index, swaps [(src,dst)], changed mask [81], render change share"""
    out = []
    lay = torch.cat([mobs[..., :3], mobs[..., 3:].any(-1, keepdim=True)], -1).long()       # zombie, cow, skeleton, arrows
    for j in range(1, len(tiles)):
        dr, dc = RL.camera(tiles[j - 1], tiles[j])
        s = RL.SHIFTS.index((dr, dc))
        prev = S2.afterstate_layout(lay[j - 1], dr, dc)
        swaps = [(a, b) for a, b, ty in S2.swaps(prev, lay[j]) if ty < 3]
        enter = PI.entering_mask()[s]
        changed = enter.clone()
        prev_tile = torch.full((63,), -1, dtype=torch.long)
        for x in range(63):
            r, c = divmod(x, 9)
            if 0 <= r + dr < 7 and 0 <= c + dc < 9:
                prev_tile[x] = tiles[j - 1][(r + dr) * 9 + c + dc]
        moved = prev.clone().clamp_min(0)
        for a, b in swaps:
            moved[[a, b]] = moved[[b, a]]
        for x in range(63):
            if bool(enter[x]):
                continue
            if int(prev_tile[x]) != int(tiles[j][x]) or bool((moved[x] != lay[j][x]).any()) or bool(lay[j][x, 3]) or bool(prev[x, 3] == 1):
                changed[x] = True
        if bool((facing[j] != facing[j - 1]).any()):
            changed[31] = True
        if bool((hud[j] != hud[j - 1]).any()):
            changed[63:] = True
        # rendering change: persisting, unchanged, camera-aligned map cells whose pixels differ
        px_now = frames[j].reshape(9, 7, 9, 7, 3).permute(0, 2, 1, 3, 4).reshape(81, -1).int()
        px_prev = frames[j - 1].reshape(9, 7, 9, 7, 3).permute(0, 2, 1, 3, 4).reshape(81, -1).int()
        keep = [x for x in range(63) if not bool(changed[x]) and x != 31 and not lay[j][x].any() and (moved[x] == 0).all()]
        src = [((x // 9) + dr) * 9 + (x % 9) + dc for x in keep]
        diff = (px_now[keep] - px_prev[src]).abs().amax(-1) > 2 if keep else torch.zeros(0, dtype=torch.bool)
        out.append({'scroll': s, 'swaps': swaps, 'changed': changed, 'render': float(diff.float().mean()) if len(keep) else 0.,
                    'lay_prev': prev, 'lay': lay[j]})
    return out


def to_patches(frames):                                                           # uint8 [n,63,63,3] -> float [n,81,147]
    return frames.reshape(len(frames), 9, 7, 9, 7, 3).permute(0, 1, 3, 2, 4, 5).reshape(len(frames), 81, 147).float()


def to_frames(patches):                                                           # [n,81,147] -> float pixels [n,63,63,3] in [0,1]
    return patches.reshape(len(patches), 9, 9, 7, 7, 3).permute(0, 1, 3, 2, 4, 5).reshape(len(patches), 63, 63, 3) / 255.0


def shift_any(x, s, idx_all):
    """[81,D] -> [81,D] exact copy under scroll choice s (map cells; HUD copies itself)"""
    d = x.shape[-1]
    grid = x.reshape(9, 9, d); pad = F.pad(grid, (0, 0, 1, 1, 1, 1))
    cand = torch.stack([grid.reshape(81, d)] + [pad[1 + dr:10 + dr, 1 + dc:10 + dc].reshape(81, d) for dr, dc in PI.SCROLL_OFFSETS[1:]], 1)
    return cand.gather(1, idx_all[s][:, None, None].expand(-1, 1, d))[:, 0]


def rebuild_px(fresh_px, trans, m, idx_all, relight):
    """pixel patches [16,81,147]; frames 15-m..14 rebuilt; relight: fit one global affine colour map per stale frame"""
    s_ = fresh_px.clone()
    for j in range(15 - m, 15):
        tr = trans[j - 1]
        cp = shift_any(s_[j - 1], tr['scroll'], idx_all)
        sw = cp.clone()
        for a, b in tr['swaps']:
            sw[a], sw[b] = cp[b], cp[a]
        s_[j] = torch.where(tr['changed'].cuda()[:, None], fresh_px[j], sw)
    if relight:
        out = s_.clone()
        for j in range(15 - m, 15):
            copied = torch.ones(81, dtype=torch.bool, device=s_.device); copied[63:] = False
            for jj in range(15 - m, j + 1):
                copied &= ~trans[jj - 1]['changed'].cuda()                          # never refreshed in the span: a pure copy
            if int(copied.sum()) < 4:
                continue
            src = s_[j][copied].reshape(-1, 3); dst = fresh_px[j][copied].reshape(-1, 3)
            X = torch.cat([src, torch.ones(len(src), 1, device=src.device)], 1).double()
            eye = torch.zeros(4, 3, dtype=torch.float64, device=src.device); eye[:3] = torch.eye(3, dtype=torch.float64, device=src.device)
            lam = 1e-3 * float((X * X).sum())                                          # ridge toward the identity map: few colours
            A = torch.linalg.solve(X.T @ X + lam * torch.eye(4, dtype=torch.float64, device=src.device), X.T @ dst.double() + lam * eye).float()
            cells = torch.where(copied)[0]
            out[j, cells] = (torch.cat([s_[j, cells].reshape(-1, 3), torch.ones(len(cells) * 49, 1, device=src.device)], 1) @ A).reshape(len(cells), 147).clamp(0, 255)
        s_ = out
    return s_


def rebuild(fresh, trans, m, idx_all):
    """fresh [16,81,192] representation; frames 15-m..14 rebuilt from frame 14-m with the true transformations"""
    s = fresh.clone()
    for j in range(15 - m, 15):
        tr = trans[j - 1]
        cand = PI.PIWorld.candidates(None, s[j - 1][None, None])[0, 0]              # [81,5,192]
        cp = cand.gather(1, idx_all[tr['scroll']][:, None, None].expand(-1, 1, 192))[:, 0]
        sw = cp.clone()
        for a, b in tr['swaps']:
            sw[a], sw[b] = cp[b], cp[a]
        s[j] = torch.where(tr['changed'].cuda()[:, None], fresh[j], sw)
    return s


@torch.no_grad()
def main():
    import rollouts as Rr
    config = BASE.settings()
    enc = Encoder()
    files = sorted(Rr.STORE.glob('seed-*.pt'))
    roots = [r for f in files for r in torch.load(f, weights_only=False)]
    pick = torch.randperm(len(roots), generator=torch.Generator().manual_seed(20261028))[:int(os.environ.get('RECONTEXT_ROOTS', 240))].tolist()
    idx_all = torch.zeros(5, 81, dtype=torch.long)
    for s in range(5):
        idx_all[s, :PI.MAP] = s
    idx_all = idx_all.cuda()
    pair_index = {p: i for i, p in enumerate(IC.PAIRS)}
    windows = []
    for ri in pick:
        r = roots[ri]
        for k in range(5):
            if bool(r['future_dead'][k][:15].any()):
                continue
            frames = torch.cat([r['context'][-1:], r['future_frames'][k][:15]])
            vis = torch.cat([r['root_visible'][None], r['future_visible'][k][:15]])
            ac = F.pad(r['future_actions'][:15], (0, 1))
            windows.append((frames, vis, ac))
    print(json.dumps({'windows': len(windows)}), flush=True)
    prepared = []
    for frames, vis, ac in windows:
        tiles, mobs, hud, facing = facts(vis)
        trans = transitions(tiles, mobs, hud, facing, frames)
        c = enc.content(frames)                                                    # [16,81,192]
        fresh = enc.recontext(c)
        prepared.append({'ac': ac, 'trans': trans, 'c': c.half(), 'fresh': fresh.half(), 'mobs': mobs, 'tiles': tiles, 'hud': hud,
                         'px': to_patches(frames).half() if RELIGHT else None})
    res = {}
    for seed in (1, 2):
        w = PI.PIWorld().cuda().eval()
        w.load_state_dict(torch.load(T / f'pi_seed{seed}/pi_u3000.pt', map_location='cpu', weights_only=False)['world'])
        g = torch.Generator().manual_seed(5)
        acc = {}
        for b0 in range(0, len(prepared), 8):
            batch = prepared[b0:b0 + 8]
            ac = torch.stack([p['ac'] for p in batch]).cuda()
            noise = DM.draw_noise((len(batch), 16), g)[0].cuda()
            contexts = {'fresh': torch.stack([p['fresh'].float() for p in batch])}
            for m in DEPTHS:
                contexts[f'T{m}'] = torch.stack([rebuild(p['fresh'].float(), p['trans'], m, idx_all) for p in batch])
                cs = [rebuild(p['c'].float(), p['trans'], m, idx_all) for p in batch]
                contexts[f'C{m}'] = torch.stack([torch.cat([p['fresh'][:15 - m].float(), enc.recontext(x[15 - m:15]), p['fresh'][15:].float()])
                                                 for p, x in zip(batch, cs)])
                if RELIGHT:
                    for tag, rl in (('P', False), ('G', True)):
                        rows = []
                        for p in batch:
                            px = rebuild_px(p['px'].float().cuda(), p['trans'], m, idx_all, rl)
                            emb = enc.content_px(to_frames(px[15 - m:15]))
                            rows.append(torch.cat([p['fresh'][:15 - m].float(), enc.recontext(emb), p['fresh'][15:].float()]))
                        contexts[f'{tag}{m}'] = torch.stack(rows)
            outs = {}
            for name, fr in contexts.items():
                with autocast_context(config):
                    belief, ha = w.backbone_full(fr, ac)
                    _, gen, glogit, _, sl = w.afterstate_parts(fr, ac, belief, ha)
                    logits = w.completion.propose(w.completion.trunk(belief, noise)).float()
                logits = logits.masked_fill(~w.completion.valid, -torch.inf)
                outs[name] = (belief[:, 14].float(), gen[:, 14].float(), glogit[:, 14].float(), sl[:, 14].float(),
                              torch.sigmoid(IC.pair_prior_logit(logits[:, 14:15]))[:, 0])
            for i, p in enumerate(batch):
                tr14 = p['trans'][14]
                mob14 = p['mobs'][14]
                target = p['fresh'][15].float()
                edits = [x for x in range(63) if bool(tr14['changed'][x]) and not bool(PI.entering_mask()[tr14['scroll']][x])
                         and x != 31 and not p['mobs'][15][x].any() and not tr14['lay_prev'][x].clamp_min(0).any()]
                health_drop = float(p['hud'][15][0] - p['hud'][14][0]) < 0
                gate_lab = tr14['changed'][:63].clone(); gate_lab[31] = False
                for x in range(63):
                    if bool(PI.entering_mask()[tr14['scroll']][x]):
                        gate_lab[x] = False
                for m in DEPTHS:
                    span = p['trans'][14 - m:14]
                    sc = 'scrolls0' if all(t['scroll'] == 0 for t in span) else 'scrolls1+'
                    rd = 'render_stable' if all(t['render'] < .05 for t in span) else 'render_changed'
                    for rep in REPS:
                        key = f'm{m}_{sc}_{rd}'
                        a = acc.setdefault(key, {})
                        for nm in ('fresh', rep):
                            b = a.setdefault(nm, {'n': 0, 'toward': [], 'gate_s': [], 'gate_y': [], 'gen_err': [], 'health_s': [], 'health_y': [],
                                                  'scroll_ok': 0, 'cos': {'zombie': [], 'adjacent': [], 'map_other': [], 'hud': []}})
                            if nm == 'fresh' and rep != REPS[0]:
                                continue                                       # fresh tallied once per key
                            bel, gen, gl, sl, prob = (o[i] for o in outs['fresh' if nm == 'fresh' else f'{rep}{m}'])
                            b['n'] += 1
                            b['scroll_ok'] += int(int(sl.argmax()) == tr14['scroll'])
                            b['gate_s'].append(torch.sigmoid(gl[:63]).cpu()); b['gate_y'].append(gate_lab)
                            b['health_s'].append(float(torch.sigmoid(gl[63]))); b['health_y'].append(health_drop)
                            if edits:
                                b['gen_err'].append((F.layer_norm(gen[edits], (192,)) - target[edits]).norm(dim=-1).cpu())
                            prev = tr14['lay_prev']
                            for x in range(63):
                                if x not in INTERIOR or x == 31 or int(prev[x, 0]) != 1 or dist(x) > 4:
                                    continue
                                tw = 0.
                                for yy in (x - 9, x + 9, x - 1, x + 1):
                                    pr = (min(x, yy), max(x, yy))
                                    if pr in pair_index and dist(yy) < dist(x):
                                        tw += float(prob[pair_index[pr]])
                                b['toward'].append(tw)
                            if nm != 'fresh':
                                cs_ = F.cosine_similarity(bel, outs['fresh'][0][i], dim=-1).cpu()
                                zm = torch.zeros(81, dtype=torch.bool); zm[:63] = mob14[:, 0]
                                am = torch.zeros(81, dtype=torch.bool); am[list(ADJ)] = True
                                hm = torch.zeros(81, dtype=torch.bool); hm[63:] = True
                                om = ~(zm | am | hm); om[31] = False
                                for k_, mk in (('zombie', zm), ('adjacent', am), ('map_other', om), ('hud', hm)):
                                    if mk.any():
                                        b['cos'][k_].append(cs_[mk])
            if b0 % 160 == 0:
                print(json.dumps({'seed': seed, 'windows_done': b0 + 8}), flush=True)
        out = {}
        for key, a in sorted(acc.items()):
            row = {}
            for nm, b in a.items():
                gs, gy = torch.cat(b['gate_s']), torch.cat(b['gate_y'])
                hs, hy = torch.tensor(b['health_s']), torch.tensor(b['health_y'])
                row[nm] = {'windows': b['n'], 'chase_toward': round(float(np.mean(b['toward'])), 4) if b['toward'] else None, 'zombies': len(b['toward']),
                           'gate_auc': None if RL.auc(gs, gy) is None else round(RL.auc(gs, gy), 4),
                           'health_auc': None if RL.auc(hs, hy) is None else round(RL.auc(hs, hy), 4), 'health_drops': int(hy.sum()),
                           'gen_err_edits': round(float(torch.cat(b['gen_err']).mean()), 3) if b['gen_err'] else None,
                           'edit_cells': int(sum(len(e) for e in b['gen_err'])), 'scroll_acc': round(b['scroll_ok'] / max(b['n'], 1), 4),
                           'belief_cos': {k_: round(float(torch.cat(v).mean()), 4) for k_, v in b['cos'].items() if v}}
            out[key] = row
        res[f'seed{seed}'] = out
        print(json.dumps({f'seed{seed}': out}), flush=True)
        del w; torch.cuda.empty_cache()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f'recontext_stale{"_relight" if RELIGHT else ""}.json').write_text(json.dumps(res, indent=1))


if __name__ == '__main__':
    main()
