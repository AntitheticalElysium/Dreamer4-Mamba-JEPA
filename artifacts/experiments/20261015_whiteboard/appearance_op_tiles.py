"""R49 follow-up: which tiles does the token-space appearance operator get wrong at the sleep onset? (2026-10-16; GPU inference)

On R47's 200 test roots: tile classes read by the frozen bank reader on the ORACLE asleep frame vs on (a) the operator's output
(appearance_op_d<depth>_s<seed>.pt, source sleep theta), (b) the awake copy, (c) the REAL asleep frame, (d) the real AWAKE frame
vs the real asleep frame. The largest confusion pairs, and the reader's own agreement between the oracle and the real asleep frame
(how much of the gap is the reader's ambiguity on grey frames).
Usage: appearance_op_tiles.py <depth> <seed> -> artifacts/eda/whiteboard/appearance_op_tiles_d<depth>_s<seed>.json
"""
import collections
import json
import os
import sys
from pathlib import Path

os.environ.setdefault('TRITON_F32_DEFAULT', 'ieee')
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / '20261012_oracle'))
sys.path.insert(0, str(HERE))

import torch

import appearance_op as AO
import recontext_stale as RS
import sleep_transform as ST


@torch.no_grad()
def main():
    import direct_coherence as C
    import audit_s5_direct as A5
    depth, seed = int(sys.argv[1]), int(sys.argv[2])
    enc = RS.Encoder()
    op = AO.Operator(depth).cuda().eval()
    op.load_state_dict(torch.load(AO.OUT / f'appearance_op_d{depth}_s{seed}.pt', map_location='cuda', weights_only=False))
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
    agree = collections.Counter(); conf = {k: collections.Counter() for k in ('operator', 'copy', 'real_asleep')}
    per_class = collections.Counter(); per_class_bad = collections.Counter()
    for root, nxt in test:
        src = enc.recontext(enc.content_px(root[None].float() / 255.0)).float()
        c_or = enc.content_px((ST.sleep_pixels(root) / 255.0)[None])[0]
        c_or[63:] = enc.content_px(nxt[None].float() / 255.0)[0, 63:]
        oracle = enc.recontext(c_or[None]).float()
        real = enc.recontext(enc.content_px(nxt[None].float() / 255.0)).float()
        pred = op(src, AO.sleep_theta()[None].cuda()); pred[:, 63:] = oracle[:, 63:]
        t_or = C.decode(oracle.cpu(), bank, head)['tile'][0, :63]
        for name, tok in (('operator', pred), ('copy', src), ('real_asleep', real)):
            t = C.decode(tok.cpu(), bank, head)['tile'][0, :63]
            v = t_or >= 0
            agree[name] += int((t[v] == t_or[v]).sum()); agree['n_' + name] += int(v.sum())
            for a, b in zip(t_or[v].tolist(), t[v].tolist()):
                if a != b:
                    conf[name][f'{a}->{b}'] += 1
                if name == 'operator':
                    per_class[a] += 1; per_class_bad[a] += int(a != b)
    res = {'roots': len(test), 'tile_agreement_vs_oracle': {k: round(agree[k] / agree['n_' + k], 4) for k in conf},
           'top_confusions': {k: c.most_common(8) for k, c in conf.items()},
           'operator_error_by_oracle_class': {str(k): [per_class[k], round(per_class_bad[k] / per_class[k], 3)] for k in sorted(per_class)}}
    print(json.dumps(res, indent=1))
    (AO.OUT / f'appearance_op_tiles_d{depth}_s{seed}.json').write_text(json.dumps(res, indent=1))


if __name__ == '__main__':
    main()
