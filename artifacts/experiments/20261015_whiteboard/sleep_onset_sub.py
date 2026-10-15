"""R48: is PI's sleep blind spot the onset erasure alone? Substitute the onset frame, let PI imagine the rest (2026-10-16; GPU inference)

R46: PI regenerates 49% of map cells at sleep onset and erases zombies; R47: a content-preserving appearance operator (12-parameter
colour map, or the source's sleep render) keeps them. Substitution test on the DV16 roots (dv16.load_roots, 200 roots; PI s1 and s2;
M = 4 futures; first action SLEEP; recorded continuation; 16 steps as dv16.imagine_pass; the same noise draws for every variant):
  natural      PI imagines all 16 frames (as DV16)
  appearance   imagined frame 1 := the root frame with its map rendered asleep by the source's function (R47: the learned colour map
               is equivalent, zombie / cow recall 0.97 / 0.98 vs this), root HUD, embedded and recontextualized by the frozen encoder;
               PI imagines frames 2-16 from it
  real_onset   imagined frame 1 := the real depth-1 frame (deeppanel key 0; real mob moves and HUD); PI imagines frames 2-16
  noop_natural control: first action NOOP, PI imagines all 16 frames (an awake player with the same zombies)
  app_player / app_hud / app_player_hud   appearance with the REAL onset frame's content at the player cell (the renderer draws a
               sleeping player texture) / at the 18 HUD cells / both: which cue separates appearance from real_onset
  app_pihud    appearance map cells, PI's OWN sampled HUD at frame 1 (the appearance substitution otherwise fixes the HUD to the
               root's and deletes PI's own onset-step consequence): an emulation of PI with an appearance operator on its copies
  tokop_pihud  as app_pihud, but the map cells come from R49's token-space operator (appearance_op_d4_s1.pt) applied to the
               root's Raw tokens with the source's sleep theta: no pixel-equivalent state
  tokopfit_pihud  the same with theta INFERRED from real token pairs through the frozen operator (appearance_theta_d4_s1.json)
Env: SUB_VARIANTS (comma list), SUB_TAG (output suffix).
Per-future health and zombie adjacency by depth are saved (sleep_onset_sub_<world>.pt) for conditional readouts: P(drop | zombie
adjacent), damage after depth 1 only (real_onset's depth-1 HUD is real).
Readouts (frozen readers on imagined frames, as DV16): ZA (zombie adjacent at some depth), HD (health drop >= 0.5 below the root's by
16), HL (max health loss), mean zombies on the map at depths 1/2/4/8/16. Decision: DV16's saved PI rows with the SLEEP risk (HD)
replaced by each variant's; expected safe and the share of roots choosing sleep (references from dv16.json).
Readings, declared before running:
  a. if appearance's ZA rises from natural's toward the real 0.92 (PI's own noop: 0.83), the onset erasure explains the lost zombies;
  b. if ZA recovers but HD stays far below the real 0.98, PI does not imagine what zombies do to a sleeper: a consequence gap after
     onset, which a better appearance operator cannot fix;
  c. if the decision with the substituted sleep risk stops preferring sleep and recovers to the sleep-excluded level (~0.57), the
     onset is the whole decision failure.
Usage: sleep_onset_sub.py -> artifacts/eda/whiteboard/sleep_onset_sub.json
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

import torch
from torch.nn import functional as F

import dv16 as DV
import sleep_transform as ST
from decision_proxies import hud_health

SLEEP, M, DEPTHS = 6, 4, (1, 2, 4, 8, 16)
VARIANTS = tuple(os.environ.get('SUB_VARIANTS', 'natural,appearance,real_onset,noop_natural').split(','))
TAG = os.environ.get('SUB_TAG', '')


@torch.no_grad()
def rollout(world, ctx, acts, g, config, onset, own_hud=False):
    """dv16.imagine_pass for M rows of one action; onset [81,192] (or None) replaces imagined frame 1 (its map cells only if
    own_hud: the HUD stays PI's own sampled prediction) -> [M,16,81,192] half"""
    import direct_model as DM
    import transport_model as TM
    from d4mj.train import autocast_context
    n = len(acts)
    acts = acts.cuda()
    hist = [ctx[j][None].expand(n, -1, -1).float().cuda() for j in range(len(ctx))]
    out = []
    for k in range(1, DV.H + 1):
        L = len(hist); start = max(0, L - DV.W)
        fr = torch.stack(hist[start:L], 1)
        fr = torch.cat([fr, torch.zeros_like(fr[:, :1])], 1)
        ac = F.pad(acts[:, start:L], (0, 1))
        with autocast_context(config):
            belief, ha = world.backbone_full(fr, ac)
            noise = DM.draw_noise((n, fr.shape[1]), g)[0].cuda(); gum = TM.draw_gumbel((n, fr.shape[1]), g)[0].cuda()
            nxt = world.complete(fr, ac, belief, ha, noise, gum)[0][:, L - 1 - start].float()
        if k == 1 and onset is not None:
            sub = onset[None].expand(n, -1, -1).float().cuda()
            nxt = torch.cat([sub[:, :63], nxt[:, 63:]], 1) if own_hud else sub
        hist.append(nxt); out.append(nxt.half().cpu())
    return torch.stack(out, 1)


def main():
    import direct_train as BASE
    import direct_coherence as C
    import audit_s5_direct as A5
    config = BASE.settings()
    roots, _ = DV.load_roots()
    full = {}
    for f in sorted(DV.DEV.glob('seed-*.pt')):
        for r in torch.load(f, weights_only=False):
            full[(int(r['seed']), int(r['step']))] = r
    enc = DV.RS.Encoder()
    bank = torch.load(A5.BASE / 'snap_bank_readers.pt', map_location='cpu', weights_only=False)
    head = torch.load(A5.BASE / 'learned_innovation/eval_u3000_h01/mob_heads.pt', map_location='cpu', weights_only=False)['heads']['real']
    summary = json.loads((DV.OUT / 'dv16.json').read_text())
    P16 = torch.stack([r['P'] for r in roots])[..., 2]                                        # [200,17]
    top = None
    if any(v.startswith('tokop') for v in VARIANTS):
        import appearance_op as AO
        globals()['AO'] = AO
        top = AO.Operator(4).cuda().eval()
        top.load_state_dict(torch.load(DV.OUT / 'appearance_op_d4_s1.pt', map_location='cuda', weights_only=False))
        th_fit = torch.tensor(json.loads((DV.OUT / 'appearance_theta_d4_s1.json').read_text())['fitted_theta']).cuda()
    onsets = []
    for r in roots:
        root = r['frames'][-1]
        c_root = enc.content(root[None])[0].float()
        c_app = enc.content_px((ST.sleep_pixels(root) / 255.0)[None])[0].float()
        c_app[63:] = c_root[63:]
        real1 = full[(r['seed'], r['step'])]['depth_frames'][SLEEP][0]
        c_real = enc.content(real1[None])[0].float()
        variants = {'appearance': c_app, 'real_onset': c_real}
        for name, cells in (('app_player', [31]), ('app_hud', list(range(63, 81))), ('app_player_hud', [31] + list(range(63, 81)))):
            c = c_app.clone(); c[cells] = c_real[cells]; variants[name] = c
        on = {k: enc.recontext(c[None])[0].half().cpu() for k, c in variants.items()}
        if top is not None:                                                                # R49 token-space operator, source sleep theta
            with torch.no_grad():
                on['tokop'] = top(enc.recontext(c_root[None]).float(), AO.sleep_theta()[None].cuda())[0].half().cpu()
                on['tokopfit'] = top(enc.recontext(c_root[None]).float(), th_fit[None])[0].half().cpu()
        onsets.append(on)
    res = {'roots': len(roots), 'M': M}
    for wname in ('pi_s1', 'pi_s2'):
        kind, ckpt, _ = DV.WORLDS[wname]
        world = DV.load_world(kind, ckpt)
        acc = {v: {'HD': [], 'HL': [], 'ZA': [], 'zmap': []} for v in VARIANTS}
        dump = {v: {'h0': [], 'health': [], 'adjacent': []} for v in VARIANTS}
        for ri, r in enumerate(roots):
            ctx = enc.recontext(enc.content(r['frames'])).half().cpu()
            h0 = float(hud_health(ctx[-1:].float(), bank['heads']['hud'])[0])
            for v in VARIANTS:
                acts = torch.cat([r['led'][1:8], torch.tensor([0 if v == 'noop_natural' else SLEEP]), r['rec']])[None].repeat(M, 1)
                g = torch.Generator().manual_seed(20261040 + ri)                                 # the same draws for every variant
                src = {'app_pihud': 'appearance', 'tokop_pihud': 'tokop', 'tokopfit_pihud': 'tokopfit'}.get(v, v)
                imag = rollout(world, ctx, acts, g, config, onsets[ri].get(src), v in ('app_pihud', 'tokop_pihud', 'tokopfit_pihud'))
                hk = hud_health(imag.float(), bank['heads']['hud'])                               # [M,16]
                mob = C.decode(imag.float(), bank, head)['mob']['real'][..., 0]                   # [M,16,81]
                A = acc[v]
                A['HD'].append(float((hk <= h0 - .5).any(-1).float().mean()))
                A['HL'].append(float((h0 - hk).amax(-1).clamp_min(0).mean()))
                A['ZA'].append(float(mob[..., DV.ADJ].any(-1).any(-1).float().mean()))
                A['zmap'].append(mob[..., :63].sum(-1).float().mean(0)[[d - 1 for d in DEPTHS]].tolist())
                dump[v]['h0'].append(h0); dump[v]['health'].append(hk.float()); dump[v]['adjacent'].append(mob[..., DV.ADJ].any(-1))
            if ri % 50 == 0:
                print(json.dumps({'world': wname, 'roots_done': ri + 1}), flush=True)
        torch.save({v: {'h0': torch.tensor(d['h0']), 'health': torch.stack(d['health']), 'adjacent': torch.stack(d['adjacent'])} for v, d in dump.items()},
                   DV.OUT / f'sleep_onset_sub{TAG}_{wname}.pt')
        rows = torch.load(DV.OUT / f'dv16_{wname}.pt', weights_only=False)
        out = {}
        for v, A in acc.items():
            hd = torch.tensor(A['HD'])
            dec = []
            for i, row in enumerate(rows):
                risk = row['HD'].clone(); risk[SLEEP] = hd[i]
                m = risk == risk.min()
                dec.append((1 - float(P16[i][m].mean()), float(m[SLEEP]) / float(m.sum())))
            out[v] = {'ZA': round(sum(A['ZA']) / len(A['ZA']), 4), 'HD': round(float(hd.mean()), 4), 'HL': round(sum(A['HL']) / len(A['HL']), 4),
                      'zombies_on_map_by_depth': [round(float(torch.tensor([z[k] for z in A['zmap']]).mean()), 3) for k in range(len(DEPTHS))],
                      'decision_expected_safe_HD': round(sum(d[0] for d in dec) / len(dec), 4) if v != 'noop_natural' else None,
                      'sleep_chosen_share': round(sum(d[1] for d in dec) / len(dec), 4) if v != 'noop_natural' else None}
        out['saved_dv16_sleep_HD'] = round(float(torch.stack([row['HD'][SLEEP] for row in rows]).mean()), 4)
        out['saved_dv16_expected_safe_HD'] = summary[wname]['HD']['expected_safe']
        res[wname] = out
        print(json.dumps({wname: out}, indent=1), flush=True)
        del world; torch.cuda.empty_cache()
    res['references'] = {k: summary['references'][k] for k in ('uniform', 'prior', 'one_real_future', 'oracle31')}
    res['real_sleep'] = json.loads((DV.OUT / 'sleep_check.json').read_text())['sleep']
    (DV.OUT / f'sleep_onset_sub{TAG}.json').write_text(json.dumps(res, indent=1))


if __name__ == '__main__':
    main()
