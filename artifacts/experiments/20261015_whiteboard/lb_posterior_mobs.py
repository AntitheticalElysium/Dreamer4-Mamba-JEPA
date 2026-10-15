"""Does LB's latent encode mobs at all? (2026-10-16; GPU inference) The token readout of LB's POSTERIOR states (frames it has
observed) on 320 seeded training windows, read by the frozen real mob reader and tile reader, vs the real frames: mob recall /
precision per type and tile agreement. If the posterior readout has no mobs, the global latent does not encode them (a
representation failure, not an imagination one).
Usage: lb_posterior_mobs.py <update> -> prints"""
import json, os, sys
from pathlib import Path
os.environ.setdefault('TRITON_F32_DEFAULT', 'ieee')
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / '20261012_oracle')); sys.path.insert(0, str(HERE))
import torch
import lb_world as LB
import audit_s5_direct as A5, direct_train as BASE, direct_coherence as C

@torch.no_grad()
def main():
    update = int(sys.argv[1])
    _, windows, actions, tokens = BASE.load_data()
    sel = torch.randperm(len(windows), generator=torch.Generator().manual_seed(20261032))[:320]
    bank = torch.load(A5.BASE / 'snap_bank_readers.pt', map_location='cpu', weights_only=False)
    head = torch.load(A5.BASE / 'learned_innovation/eval_u3000_h01/mob_heads.pt', map_location='cpu', weights_only=False)['heads']['real']
    for s in (1, 2):
        w = LB.LBWorld('cuda').cuda().eval()
        w.load_state_dict(torch.load(A5.BASE / f'transport/lb_seed{s}/lb_u{update:05d}.pt', map_location='cpu', weights_only=False)['world'])
        tp = {k: [0, 0, 0] for k in ('zombie', 'cow', 'skeleton')}; tiles = [0, 0]
        for b0 in range(0, 320, 16):
            frames, ac, _ = BASE.batch(tokens, actions, windows, sel[b0:b0 + 16])
            stoch, deter, _ = w.observe(frames, ac)
            rec = w.readout(stoch, deter).float().cpu()
            dr = C.decode(frames.float().cpu(), bank, head); dp = C.decode(rec, bank, head)
            for t, k in enumerate(('zombie', 'cow', 'skeleton')):
                r_, p_ = dr['mob']['real'][..., t], dp['mob']['real'][..., t]
                tp[k][0] += int((r_ & p_).sum()); tp[k][1] += int(r_.sum()); tp[k][2] += int(p_.sum())
            v = dr['tile'] >= 0
            tiles[0] += int((dr['tile'][v] == dp['tile'][v]).sum()); tiles[1] += int(v.sum())
        out = {k: {'real': v[1], 'recall': round(v[0] / max(v[1], 1), 4), 'predicted': v[2], 'precision': round(v[0] / max(v[2], 1), 4)} for k, v in tp.items()}
        out['tile_agreement'] = round(tiles[0] / tiles[1], 4)
        print(json.dumps({f'lb_seed{s}_u{update}': out}), flush=True)

if __name__ == '__main__':
    main()
