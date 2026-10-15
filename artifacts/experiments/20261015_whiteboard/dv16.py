"""DV16: the H16 decision value of a world's imagination (DV16_PREREG.md governs; 2026-10-16; GPU inference, no training).

Roots: deeppanel DEV store, kept by deepeval's rule, opp16 (real P(dead by 16) varies across the 17 first actions), no sentinel
action in the context; 200 seeded roots. Context: the root's 8 frames (frozen Raw tokens). Per first action a (17): M futures of
16 steps -- action a at the root, then the recorded continuation -- rolled out as transport_recursion (windows of the last <= 15
frames). Action convention (transport worlds: the action AT frame t): context frame j takes led_to_action[j + 1], the root takes a,
imagined frame k takes recorded[k - 1].
Risk (frozen readers on imagined frames): HD share of futures whose health (bank HUD reader) drops >= 0.5 below the root's within
16 steps (primary); HL mean max health loss; ZA share with a zombie adjacent at some step. Decision: argmin risk, uniform over ties;
expected safe = 1 - P16(chosen) (32-key real labels), averaged over roots; references (check_h16_value.arms) on the same roots.
Usage: dv16.py <world> ...   worlds: pi_s1 pi_s2 ic_s1 ic_s2 D_parent_ref   -> artifacts/eda/whiteboard/dv16_<world>.pt + dv16.json
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

import recontext_stale as RS                                                       # Encoder (frozen Raw, exact)
import audit_s5_direct as A5
from decision_proxies import hud_health

OUT = HERE.parents[1] / 'eda/whiteboard'
DEV = HERE.parents[2] / 'artifacts/eda/deeppanel_dev_v1'
N_ACT, H, W, N_ROOTS = 17, 16, 15, int(os.environ.get("DV16_ROOTS", 200))
ADJ = [22, 30, 32, 40]
WORLDS = {'pi_s1': ('pi', 'pi_seed1/pi_u3000.pt', 4), 'pi_s2': ('pi', 'pi_seed2/pi_u3000.pt', 4),
          'ic_s1': ('operator', 'ic_seed1/ic_u3000.pt', 4), 'ic_s2': ('operator', 'ic_seed2/ic_u3000.pt', 4),
          'D_parent_ref': ('paired', None, 1)}


def load_roots():
    rows = []
    for f in sorted(DEV.glob('seed-*.pt')):
        for r in torch.load(f, weights_only=False):
            P = r['p_dead_by']['recorded'][:, [0, 3, 15]].float()
            if not bool((P.amax(0) > P.amin(0)).any()) or not bool(P[:, 2].amax() > P[:, 2].amin()):
                continue
            if bool((r['led_to_action'][1:] >= N_ACT).any()):
                continue
            rows.append({'frames': r['frames'], 'led': r['led_to_action'], 'rec': r['recorded'], 'P': P,
                         'D': r['depth_dead'][:, [0, 3, 15]].float(), 'seed': int(r['seed']), 'step': int(r['step'])})
    pick = torch.randperm(len(rows), generator=torch.Generator().manual_seed(20261030))[:N_ROOTS].tolist()
    return [rows[i] for i in sorted(pick)], len(rows)


def load_world(kind, ckpt):
    import paired_model as PM
    import transport_model as TM
    import pi_model as PI
    if kind == 'pi':
        w = PI.PIWorld()
    elif kind == 'operator':
        w = TM.TransportWorld('operator')
    else:
        w = PM.PairedWorld()
    path = A5.BASE / ('paired_event/training/paired_u0000.pt' if ckpt is None else f'transport/{ckpt}')
    w.load_state_dict(torch.load(path, map_location='cpu', weights_only=False)['world'])
    return w.cuda().eval()


@torch.no_grad()
def imagine(world, kind, ctx, acts_full, M, g, config):
    """ctx [8,81,192]; acts_full [17, 23] long -> imagined [17*M, 16, 81, 192] (row = action * M + sample); one sample of
    all 17 actions per pass (6 GB GPU shared with other runs)"""
    passes = [imagine_pass(world, kind, ctx, acts_full, g, config) for _ in range(M)]          # each [17,16,81,192]
    return torch.stack(passes, 1).flatten(0, 1)


@torch.no_grad()
def imagine_pass(world, kind, ctx, acts_full, g, config):
    import direct_model as DM
    import transport_model as TM
    from d4mj.train import autocast_context
    n = N_ACT
    acts = acts_full.cuda()
    hist = [ctx[j][None].expand(n, -1, -1).float().cuda() for j in range(len(ctx))]
    out = []
    for k in range(1, H + 1):
        L = len(hist); start = max(0, L - W)
        fr = torch.stack(hist[start:L], 1)
        fr = torch.cat([fr, torch.zeros_like(fr[:, :1])], 1)
        ac = F.pad(acts[:, start:L], (0, 1))
        with autocast_context(config):
            belief, ha = world.backbone_full(fr, ac)
            i = L - 1 - start
            if kind == 'paired':
                nxt = F.layer_norm(world.reference(fr, ac, belief, ha)[:, i].float(), (192,))
            else:
                noise = DM.draw_noise((n, fr.shape[1]), g)[0].cuda(); gum = TM.draw_gumbel((n, fr.shape[1]), g)[0].cuda()
                nxt = world.complete(fr, ac, belief, ha, noise, gum)[0][:, i].float()
        hist.append(nxt); out.append(nxt.half().cpu())
    return torch.stack(out, 1)


def risks(imag, root_tok, bank, head, M):
    """imag [17*M,16,81,192] -> HD, HL, ZA per action [17]"""
    import direct_coherence as C
    h0 = float(hud_health(root_tok[None].float(), bank['heads']['hud'])[0])
    hk = hud_health(imag.float(), bank['heads']['hud'])                                  # [n,16]
    za = C.decode(imag.float(), bank, head)['mob']['real'][..., ADJ, 0].any(-1).any(-1)   # [n]
    hd = ((hk <= h0 - .5).any(-1)).float().view(N_ACT, M).mean(1)
    hl = (h0 - hk).amax(-1).clamp_min(0).view(N_ACT, M).mean(1)
    return hd, hl, za.float().view(N_ACT, M).mean(1)


def decide(risk, P16):
    m = risk == risk.min()
    return 1 - float(P16[m].mean()), bool(m.all())


def main():
    import direct_train as BASE
    import check_h16_signal as HS
    import check_h16_value as HV
    config = BASE.settings()
    roots, n_all = load_roots()
    enc = RS.Encoder()
    toks = [enc.recontext(enc.content(r['frames'])).half().cpu() for r in roots]       # [8,81,192] each
    bank = torch.load(A5.BASE / 'snap_bank_readers.pt', map_location='cpu', weights_only=False)
    head = torch.load(A5.BASE / 'learned_innovation/eval_u3000_h01/mob_heads.pt', map_location='cpu', weights_only=False)['heads']['real']
    P = torch.stack([r['P'] for r in roots]); D = torch.stack([r['D'] for r in roots])
    Pall, Dall, Sall = HS.load()
    opp_fit = (Pall[..., 2].amax(1) > Pall[..., 2].amin(1)) & (Sall == 0)
    prior_action = int(Pall[opp_fit][..., 2].mean(0).argmin())
    summary_path = OUT / 'dv16.json'
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
    summary['references'] = HV.arms(P, D, prior_action, torch.Generator().manual_seed(20261003))
    summary['roots'] = {'used': len(roots), 'eligible_dev_opp16': n_all, 'prior_action': prior_action}
    for name in sys.argv[1:]:
        kind, ckpt, M = WORLDS[name]
        world = load_world(kind, ckpt)
        g = torch.Generator().manual_seed(20261031)
        rows = []
        for ri, r in enumerate(roots):
            acts_full = torch.stack([torch.cat([r['led'][1:8], torch.tensor([a]), r['rec']]) for a in range(N_ACT)])
            imag = imagine(world, kind, toks[ri], acts_full, M, g, config)
            hd, hl, za = risks(imag, toks[ri][-1], bank, head, M)
            rows.append({'HD': hd, 'HL': hl, 'ZA': za})
            if ri % 20 == 0:
                print(json.dumps({'world': name, 'roots_done': ri + 1}), flush=True)
        torch.save(rows, OUT / f'dv16_{name}.pt')
        res = {}
        for key in ('HD', 'HL', 'ZA'):
            vals = [decide(row[key], P[i, :, 2]) for i, row in enumerate(rows)]
            res[key] = {'expected_safe': round(sum(v[0] for v in vals) / len(vals), 4), 'all_tied_share': round(sum(v[1] for v in vals) / len(vals), 4)}
        summary[name] = res
        print(json.dumps({name: res, 'references': {k: summary['references'][k] for k in ('uniform', 'prior', 'one_real_future', 'oracle31')}}), flush=True)
        summary_path.write_text(json.dumps(summary, indent=1))
        del world; torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
