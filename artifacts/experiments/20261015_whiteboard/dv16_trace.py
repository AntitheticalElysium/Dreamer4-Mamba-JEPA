"""R53: where do imagined dynamics lose ACTION-DEPENDENT risk? Trace the escape choice in the scene and in the consequence (2026-10-16;
GPU inference; frozen readers as labels; evaluation only)

DV16's roots (dv16.load_roots, 200), the escape choice: first action noop / left / right / up / down, then the recorded
continuation, 16 steps as dv16.imagine_pass (8-frame context). Per (root, action):
  real       deeppanel's stored key-0 future frames at depths 1, 2, 4, 8, 16 (one real sample) and the 32-key P(dead by 16)
  imagined   each world's frames at depths 1-16 (M futures: 4 sampled, 1 for the deterministic D parent)
Per frame (frozen real mob reader, bank HUD reader): zombie mask on the 63 map cells and health. Saved per world as
dv16_trace_<world>.pt {zombies [R,5,M,16,63] bool, health [R,5,M,16]} and dv16_trace_real.pt {zombies [R,5,5,63], health
[R,5,5], h0 [R]}. The analysis is dv16_trace_analysis.py.
Usage: dv16_trace.py <world> ... (pi_s1 pi_s2 ic_s1 ic_s2 D_parent_ref) | real
"""
import os
import sys
from pathlib import Path

os.environ.setdefault('TRITON_F32_DEFAULT', 'ieee')
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / '20261012_oracle'))
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / '20260927_levers'))

import json

import torch

import dv16 as DV
from decision_proxies import hud_health

MOVES = (0, 1, 2, 3, 4)


def main():
    import direct_train as BASE
    import direct_coherence as C
    import audit_s5_direct as A5
    config = BASE.settings()
    roots, _ = DV.load_roots()
    enc = DV.RS.Encoder()
    bank = torch.load(A5.BASE / 'snap_bank_readers.pt', map_location='cpu', weights_only=False)
    head = torch.load(A5.BASE / 'learned_innovation/eval_u3000_h01/mob_heads.pt', map_location='cpu', weights_only=False)['heads']['real']
    read = lambda tok: (C.decode(tok, bank, head)['mob']['real'][..., :63, 0], hud_health(tok, bank['heads']['hud']))
    if sys.argv[1:] == ['real']:
        full = {}
        for f in sorted(DV.DEV.glob('seed-*.pt')):
            for r in torch.load(f, weights_only=False):
                full[(int(r['seed']), int(r['step']))] = r
        Z, Hh, h0 = [], [], []
        with torch.no_grad():
            for r in roots:
                row = full[(r['seed'], r['step'])]
                tok = enc.recontext(enc.content(row['depth_frames'][list(MOVES)].flatten(0, 1))).float().cpu()      # [25,81,192]
                z, h = read(tok)
                Z.append(z.view(5, 5, 63)); Hh.append(h.view(5, 5))
                h0.append(float(hud_health(enc.recontext(enc.content(r['frames'][-1:])).float().cpu(), bank['heads']['hud'])[0]))
        torch.save({'zombies': torch.stack(Z), 'health': torch.stack(Hh), 'h0': torch.tensor(h0)}, DV.OUT / 'dv16_trace_real.pt')
        print(json.dumps({'real': len(Z)}), flush=True)
        return
    toks = [enc.recontext(enc.content(r['frames'])).half().cpu() for r in roots]
    for name in sys.argv[1:]:
        kind, ckpt, M = DV.WORLDS[name]
        world = DV.load_world(kind, ckpt)
        g = torch.Generator().manual_seed(20261050)
        Z, Hh = [], []
        for ri, r in enumerate(roots):
            acts = torch.stack([torch.cat([r['led'][1:8], torch.tensor([a]), r['rec']]) for a in MOVES])
            passes = [DV.imagine_pass(world, kind, toks[ri], acts, g, config) for _ in range(M)]                  # M x [5,16,81,192]
            imag = torch.stack(passes, 1).float()                                                                # [5,M,16,81,192]
            with torch.no_grad():
                z, h = read(imag.flatten(0, 2))
            Z.append(z.view(5, M, 16, 63)); Hh.append(h.view(5, M, 16))
            if ri % 50 == 0:
                print(json.dumps({'world': name, 'roots_done': ri + 1}), flush=True)
        torch.save({'zombies': torch.stack(Z), 'health': torch.stack(Hh)}, DV.OUT / f'dv16_trace_{name}.pt')
        del world; torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
