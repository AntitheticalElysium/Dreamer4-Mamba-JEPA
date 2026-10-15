"""Does one-step coherence carry into recursive imagination? (2026-10-15, evaluation only, no training.)

Self-fed H16 rollouts from the 200 DEV panel roots (context = panel frames 1..15, recorded actions 15..30): each step the
model's own sample is appended to the context. Stochastic models (free / operator arms): 8 samples per root, fresh noise per
step. Deterministic controls: the reference alone, LN(b) (the D parent's own output, and each operator's own afterstate).
Readers: the frozen real-fitted mob reader and the frozen bank tile reader, applied identically to real and imagined frames.
Truth: the single recorded future (panel frames 16..31).
Per depth k = 1..16, split bright (the 104 evaluation roots) / dark (the other 96):
  census      decoded count per mob type, imagined vs real (summed over roots), and mean |imagined - real| per frame
  camera      share of trajectories whose every scroll so far equals the real scroll (parameter-free estimator)
  terrain     on camera-consistent trajectories: decoded tile disagreement with the real frame at cells with no decoded mob in
              either, split into cells SEEN at the root (world position inside the root view) and newly ENTERING cells
  mobs_fp     decoded mobs at cells farther than k (Manhattan, world frame) from every real root mob of that type: phantoms
Usage: transport_recursion.py <name> ... (names in MODELS) -> artifacts/eda/oracle_v1/transport/recursion_<name>.json
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import audit_s5_direct as A5  # noqa: E402

OUT = A5.BASE / 'transport'
T = OUT
MODELS = {  # name: (checkpoint, arm, mode)
    'D_parent_ref': (A5.BASE / 'paired_event/training/paired_u0000.pt', 'paired', 'reference'),
    'free_s1': (T / 'training/free_u3000.pt', 'free', 'sample'),
    'op_s1': (T / 'training/operator_u3000.pt', 'operator', 'sample'),
    'op_s1_ref': (T / 'training/operator_u3000.pt', 'operator', 'reference'),
    'free_s2': (T / 'seed2/free_u3000.pt', 'free', 'sample'),
    'op_s2': (T / 'seed2/operator_u3000.pt', 'operator', 'sample'),
    'opcost_s2': (T / 'seed2/operator_cost_u3000.pt', 'operator', 'sample'),
    # TRANSPORT_IC_PREREG secondary readings (R1-R3): IC arms and their in-process ES replicates
    'ic_s1': (T / 'ic_seed1/ic_u3000.pt', 'operator', 'sample'),
    'esrep_s1': (T / 'ic_seed1/operator_u3000.pt', 'operator', 'sample'),
    'ic_s1_ref': (T / 'ic_seed1/ic_u3000.pt', 'operator', 'reference'),
    'ic_s2': (T / 'ic_seed2/ic_u3000.pt', 'operator', 'sample'),
    'esrep_s2': (T / 'ic_seed2/operator_u3000.pt', 'operator', 'sample'),
    # TRANSPORT_SF_PREREG (Track A): continued IC vs chance-forced self-fed IC, 4,500 updates
    'sf_ic_s1': (T / 'sf_seed1/ic_u4500.pt', 'operator', 'sample'),
    'sf_icsf_s1': (T / 'sf_seed1/icsf_u4500.pt', 'operator', 'sample'),
    'sf_ic_s2': (T / 'sf_seed2/ic_u4500.pt', 'operator', 'sample'),
    'sf_icsf_s2': (T / 'sf_seed2/icsf_u4500.pt', 'operator', 'sample'),
    # TRANSPORT_PI_PREREG (Track B): persistent state + inferred transformations
    'pi_s1': (T / 'pi_seed1/pi_u3000.pt', 'pi', 'sample'),
    'pi_s1_ref': (T / 'pi_seed1/pi_u3000.pt', 'pi', 'reference'),
    'pi_s2': (T / 'pi_seed2/pi_u3000.pt', 'pi', 'sample'),
}
H, W, K = 16, 15, 8


@torch.no_grad()
def rollout(name):
    import direct_model as DM
    import direct_train as BASE
    import paired_model as PM
    import transport_model as TM
    from d4mj.train import autocast_context
    config = BASE.settings()
    ckpt, arm, mode = MODELS[name]
    if arm == 'pi':                                                        # TRANSPORT_PI_PREREG (Track B world)
        sys.path.insert(0, str(HERE.parent / '20261015_whiteboard')); import pi_model
        world = pi_model.PIWorld().cuda().eval()
    else:
        world = (PM.PairedWorld() if arm == 'paired' else TM.TransportWorld(arm)).cuda().eval()
    world.load_state_dict(torch.load(ckpt, map_location='cpu', weights_only=False)['world'])
    panel = torch.load(A5.BASE / 'panel.pt', map_location='cpu', weights_only=False, mmap=True)
    tok, act = panel['tokens'], panel['actions']
    R = len(tok)
    k_samples = K if mode == 'sample' else 1
    out = torch.empty(R, k_samples, H, 81, 192, dtype=torch.float16)
    flat = out.view(R * k_samples, H, 81, 192)
    g = torch.Generator().manual_seed(20261015)
    B = 8                                                                  # roots per batch
    for s0 in range(0, R, B):
        idx = list(range(s0, min(s0 + B, R)))
        n = len(idx) * k_samples
        frames = tok[idx, 1:16].float().repeat_interleave(k_samples, 0)    # [n,15,81,192]
        acts = act[idx].repeat_interleave(k_samples, 0)                    # [n,31]
        hist = [frames[:, j] for j in range(15)]
        for k in range(1, H + 1):
            ctx = torch.stack(hist[-W:], 1).cuda()
            a = acts[:, k:k + W].cuda()
            fr = torch.cat([ctx, torch.zeros_like(ctx[:, :1])], 1); ac = F.pad(a, (0, 1))
            with autocast_context(config):
                belief, ha = world.backbone_full(fr, ac)
                if mode == 'reference':
                    nxt = F.layer_norm(world.reference(fr, ac, belief, ha)[:, 14].float(), (192,))
                else:
                    noise = DM.draw_noise((n, 16), g)[0].cuda(); gum = TM.draw_gumbel((n, 16), g)[0].cuda()
                    nxt = world.complete(fr, ac, belief, ha, noise, gum)[0][:, 14].float()
            nxt = nxt.cpu()
            hist.append(nxt)
            flat[s0 * k_samples:s0 * k_samples + n, k - 1] = nxt.half()
        print(json.dumps({'model': name, 'roots_done': idx[-1] + 1}), flush=True)
    return out


def evaluate(name, imagined):
    import direct_coherence as C
    panel = torch.load(A5.BASE / 'panel.pt', map_location='cpu', weights_only=False, mmap=True)
    bank = torch.load(A5.BASE / 'snap_bank_readers.pt', map_location='cpu', weights_only=False)
    head = torch.load(A5.BASE / 'learned_innovation/eval_u3000_h01/mob_heads.pt', map_location='cpu', weights_only=False)['heads']['real']
    selected = set(json.loads((A5.BASE / 'direct_generator/data/existing_eval_selection.json').read_text())['selected_indices'])
    scroll = A5.load_scroll()
    shifts = torch.as_tensor(scroll.SHIFTS)
    R, k_samples = imagined.shape[:2]
    real = panel['tokens'][:, 16:32].float()                               # [R,16,81,192]
    real_prev = panel['tokens'][:, 15:31].float()
    real_dec = C.decode(real, bank, head)
    rmob = real_dec['mob']['real'][..., :3]                                # [R,16,63,3] zombie, cow, skeleton (layout order)
    rtile = real_dec['tile']
    real_shift = shifts[scroll.estimate(real_prev, real)]                  # [R,16,2]
    root_mob = panel['layouts'][:, 15].bool()[..., :3]                     # [R,63,3]
    res = {'model': name, 'samples_per_root': k_samples, 'bright': {}, 'dark': {}}
    acc = {part: {k: [] for k in ('img_count', 'real_count', 'abs_err', 'camera_ok', 'tile_seen', 'tile_enter', 'phantom', 'real_phantom')} for part in ('bright', 'dark')}
    for i in range(R):
        part = 'bright' if i in selected else 'dark'
        img = imagined[i].float()                                          # [K,16,81,192]
        dec = C.decode(img, bank, head)
        imob = dec['mob']['real'][..., :3]; itile = dec['tile']
        prev = torch.cat([panel['tokens'][i, 15].float()[None, None].expand(k_samples, 1, -1, -1), img[:, :-1]], 1)
        img_shift = shifts[scroll.estimate(prev, img)]                     # [K,16,2]
        cam_ok = (img_shift == real_shift[i][None]).all(-1).cumprod(1).bool()     # [K,16]
        # world offset of each depth relative to the root view: cumulative real shift
        cum = real_shift[i].cumsum(0)                                      # [16,2] (dr,dc): successor[r,c] = prev[r+dr,c+dc]
        ic = imob.sum(-2).float()                                          # [K,16,3]
        rc = rmob[i].sum(-2).float()                                       # [16,3]
        acc[part]['img_count'].append(ic.mean(0)); acc[part]['real_count'].append(rc)
        acc[part]['abs_err'].append((ic - rc[None]).abs().mean(0))
        acc[part]['camera_ok'].append(cam_ok.float().mean(0))
        seen_err, enter_err, phantom, real_phantom = torch.zeros(16), torch.zeros(16), torch.zeros(16), torch.zeros(16)
        seen_n, enter_n = torch.zeros(16), torch.zeros(16)
        for k in range(16):
            dr, dc = int(cum[k, 0]), int(cum[k, 1])
            seen = torch.zeros(63, dtype=torch.bool)
            for r in range(7):
                for c in range(9):
                    seen[r * 9 + c] = 0 <= r + dr < 7 and 0 <= c + dc < 9
            seen[31] = False
            ok = cam_ok[:, k]
            if ok.any():
                nomob = ~(imob[ok, k].any(-1) | rmob[i, k].any(-1)[None])  # [k_ok,63]
                bad = (itile[ok, k] != rtile[i, k][None]) & nomob
                m_seen = nomob & seen[None]; m_enter = nomob & ~seen[None]
                m_enter[:, 31] = False
                seen_err[k] += bad[m_seen].float().sum(); seen_n[k] += m_seen.float().sum()
                enter_err[k] += bad[m_enter].float().sum(); enter_n[k] += m_enter.float().sum()
            # phantoms: decoded mobs farther than k+1 (Manhattan, world frame) from every real root mob of that type
            far = torch.ones(63, 3, dtype=torch.bool)
            for t in range(3):
                src = torch.where(root_mob[i, :, t])[0].tolist()
                for x in range(63):
                    r, c = divmod(x, 9); wr, wc = r + dr, c + dc                # world coords in the root frame
                    if any(abs(wr - s // 9) + abs(wc - s % 9) <= k + 1 for s in src):
                        far[x, t] = False
            phantom[k] = (imob[:, k] & far[None]).float().sum((-1, -2)).mean()
            real_phantom[k] = (rmob[i, k] & far).float().sum()
        acc[part]['tile_seen'].append(torch.stack([seen_err, seen_n], -1))
        acc[part]['tile_enter'].append(torch.stack([enter_err, enter_n], -1))
        acc[part]['phantom'].append(phantom); acc[part]['real_phantom'].append(real_phantom)
    for part in ('bright', 'dark'):
        a = acc[part]
        if not a['img_count']:
            continue
        ic, rc = torch.stack(a['img_count']).sum(0), torch.stack(a['real_count']).sum(0)   # [16,3]
        ts, te = torch.stack(a['tile_seen']).sum(0), torch.stack(a['tile_enter']).sum(0)
        res[part] = {'roots': len(a['img_count']),
                     'census_ratio[zombie,cow,skeleton]_by_depth': (ic / rc.clamp_min(1)).round(decimals=3).tolist(),
                     'abs_count_error_per_frame_by_depth': torch.stack(a['abs_err']).mean(0).sum(-1).round(decimals=3).tolist(),
                     'camera_consistent_by_depth': torch.stack(a['camera_ok']).mean(0).round(decimals=3).tolist(),
                     'tile_error_seen_by_depth': (ts[:, 0] / ts[:, 1].clamp_min(1)).round(decimals=4).tolist(),
                     'tile_error_entering_by_depth': (te[:, 0] / te[:, 1].clamp_min(1)).round(decimals=4).tolist(),
                     'phantoms_per_frame_by_depth': torch.stack(a['phantom']).mean(0).round(decimals=3).tolist(),
                     'real_phantoms_per_frame_by_depth': torch.stack(a['real_phantom']).mean(0).round(decimals=3).tolist()}
    return res


if __name__ == '__main__':
    for name in sys.argv[1:]:
        path = OUT / f'recursion_samples_{name}.pt'
        imagined = torch.load(path, weights_only=False) if path.exists() else rollout(name)
        if not path.exists():
            torch.save(imagined, path)
        res = evaluate(name, imagined)
        (OUT / f'recursion_{name}.json').write_text(json.dumps(res, indent=1))
        b = res['bright']
        print(json.dumps({'model': name, 'bright_census_d1_d8_d16': [b['census_ratio[zombie,cow,skeleton]_by_depth'][d] for d in (0, 7, 15)],
                          'camera_d16': b['camera_consistent_by_depth'][15], 'tile_seen_d1_d16': [b['tile_error_seen_by_depth'][d] for d in (0, 15)],
                          'phantoms_d1_d16': [b['phantoms_per_frame_by_depth'][d] for d in (0, 15)]}), flush=True)
