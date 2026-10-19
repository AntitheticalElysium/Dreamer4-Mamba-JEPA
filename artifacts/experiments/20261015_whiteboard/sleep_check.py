"""R45 follow-up: after SLEEP, does the world fail to imagine damage, or do the frozen readers fail on grey sleep frames?
(2026-10-16; GPU inference; deeppanel DV16 roots; real future frames are deeppanel's stored key-0 frames at depths 1,2,4,8,16)
Per root, first action sleep (6) and, as a control, noop (0): reader health (bank HUD reader) and zombie-adjacent (real mob reader)
on (a) the REAL future frames at depths 1, 2, 4, 8, 16, with the real death flags; (b) PI s1's imagined frames (M = 4) at the same
depths. Also the mean token norm distance between real and imagined frames (does PI draw the grey sleep frame?).
Usage: sleep_check.py -> artifacts/eda/whiteboard/sleep_check.json"""
import json, os, sys
from pathlib import Path
os.environ.setdefault('TRITON_F32_DEFAULT', 'ieee')
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / '20261012_oracle')); sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent / '20260927_levers'))
import torch
import dv16 as DV
from decision_proxies import hud_health
DEPTHS = (1, 2, 4, 8, 16)

def main():
    import direct_train as BASE, direct_coherence as C, audit_s5_direct as A5
    config = BASE.settings()
    roots, _ = DV.load_roots()
    full = {}
    for f in sorted(DV.DEV.glob('seed-*.pt')):                              # the stored real depth frames of the same roots
        for r in torch.load(f, weights_only=False):
            full[(int(r['seed']), int(r['step']))] = r
    enc = DV.RS.Encoder()
    bank = torch.load(A5.BASE / 'snap_bank_readers.pt', map_location='cpu', weights_only=False)
    head = torch.load(A5.BASE / 'learned_innovation/eval_u3000_h01/mob_heads.pt', map_location='cpu', weights_only=False)['heads']['real']
    world = DV.load_world('pi', 'pi_seed1/pi_u3000.pt')
    g = torch.Generator().manual_seed(7)
    acc = {a: {'real_drop': [], 'imag_drop': [], 'real_dead16': [], 'real_za': [], 'imag_za': [], 'real_h': [], 'imag_h': [], 'dist': []} for a in (0, 6)}
    for r in roots:
        row = full[(r['seed'], r['step'])]
        ctx = enc.recontext(enc.content(r['frames'])).half().cpu()
        h0 = float(hud_health(ctx[-1:].float(), bank['heads']['hud'])[0])
        for a in (0, 6):
            real = enc.recontext(enc.content(row['depth_frames'][a])).float().cpu()                      # [5,81,192]
            rh = hud_health(real, bank['heads']['hud'])
            rz = C.decode(real, bank, head)['mob']['real'][..., [22, 30, 32, 40], 0].any(-1)
            acts = torch.cat([r['led'][1:8], torch.tensor([a]), r['rec']])[None].repeat(DV.N_ACT, 1)    # 17 identical rows
            imag = DV.imagine_pass(world, 'pi', ctx, acts, g, config)[:4]                                 # 4 samples of action a
            idx = [d - 1 for d in DEPTHS]
            ih = hud_health(imag[:, idx].float(), bank['heads']['hud'])                                   # [4,5]
            iz = C.decode(imag[:, idx].float(), bank, head)['mob']['real'][..., [22, 30, 32, 40], 0].any(-1)
            A = acc[a]
            A['real_drop'].append(float((rh <= h0 - .5).any())); A['imag_drop'].append(float((ih <= h0 - .5).any(-1).float().mean()))
            A['real_dead16'].append(float(row['depth_dead'][a, 15])); A['real_za'].append(float(rz.any())); A['imag_za'].append(float(iz.any(-1).float().mean()))
            A['real_h'].append((rh - h0).tolist()); A['imag_h'].append((ih.mean(0) - h0).tolist())
            A['dist'].append(float((imag[:, idx].float().mean(0) - real).norm(dim=-1).mean()))
    res = {}
    for a, A in acc.items():
        nm = {0: 'noop', 6: 'sleep'}[a]
        res[nm] = {'real_health_drop_by16': round(sum(A['real_drop']) / len(A['real_drop']), 4), 'imagined_health_drop_by16': round(sum(A['imag_drop']) / len(A['imag_drop']), 4),
                   'real_dead_by16': round(sum(A['real_dead16']) / len(A['real_dead16']), 4),
                   'real_zombie_adjacent': round(sum(A['real_za']) / len(A['real_za']), 4), 'imagined_zombie_adjacent': round(sum(A['imag_za']) / len(A['imag_za']), 4),
                   'mean_health_change_by_depth_real': [round(float(torch.tensor([h[k] for h in A['real_h']]).mean()), 3) for k in range(5)],
                   'mean_health_change_by_depth_imag': [round(float(torch.tensor([h[k] for h in A['imag_h']]).mean()), 3) for k in range(5)],
                   'imag_vs_real_token_distance': round(sum(A['dist']) / len(A['dist']), 3)}
    print(json.dumps(res, indent=1))
    (DV.OUT / 'sleep_check.json').write_text(json.dumps(res, indent=1))

if __name__ == '__main__':
    main()
