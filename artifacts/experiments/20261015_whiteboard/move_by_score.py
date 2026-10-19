"""(2026-10-15 evening, CPU, saved recursion samples; usage: move_by_score.py <model> ...; prints only) Does an imagined mob's movement depend on how crisp it is? Move rate binned by the mob's reader score at k-1 (bright,
camera-consistent, camera-still steps; mob_fading's exact transition rule). Bins: terciles of REAL decoded-mob scores."""
import sys, json
sys.path.insert(0, '/home/antithetical/EPITA/PERSO/DynamicHorizons-Mamba-JEPA/artifacts/experiments/20261015_whiteboard')
import mob_fading as MF
import torch
A5 = MF.A5
import direct_coherence as C
torch.set_num_threads(8)
panel = torch.load(A5.BASE / 'panel.pt', map_location='cpu', weights_only=False, mmap=True)
bank = torch.load(A5.BASE / 'snap_bank_readers.pt', map_location='cpu', weights_only=False)
head = torch.load(A5.BASE / 'learned_innovation/eval_u3000_h01/mob_heads.pt', map_location='cpu', weights_only=False)['heads']['real']
selected = sorted(json.loads((A5.BASE / 'direct_generator/data/existing_eval_selection.json').read_text())['selected_indices'])
scroll = A5.load_scroll(); shifts = torch.as_tensor(scroll.SHIFTS)
tok = panel['tokens'][selected].float()
real_shift = shifts[scroll.estimate(tok[:, 15:31], tok[:, 16:32])]
rd = C.decode(tok[:, 15:32], bank, head)
rdec, rsc = rd['mob']['real'][..., :3], rd['scores']['real'][..., :3]

def collect(dec, sc, steps):
    rows = []                                                       # (type, score_prev, outcome)
    for (dp, dn, sp, ok) in steps:
        for n in torch.where(ok)[0].tolist():
            for t in (0, 1):
                for x in MF.INTERIOR:
                    if not bool(dp[n, x, t]): continue
                    if bool(dn[n, x, t]): o = 'stay'
                    elif any(bool(dn[n, y, t]) and not bool(dp[n, y, t]) for y in MF.NB[x]): o = 'move'
                    else: o = 'vanish'
                    rows.append((t, float(sp[n, x, t]), o))
    return rows

real_steps = [(rdec[:, k], rdec[:, k + 1], rsc[:, k], (real_shift[:, k] == 0).all(-1)) for k in range(16)]
real_rows = collect(None, None, real_steps)
res = {}
for t, name in ((0, 'zombie'), (1, 'cow')):
    s = torch.tensor([r[1] for r in real_rows if r[0] == t])
    edges = torch.quantile(s, torch.tensor([1 / 3, 2 / 3])).tolist()
    res[name + '_edges'] = [round(e, 3) for e in edges]
def summarize(rows, label):
    out = {}
    for t, name in ((0, 'zombie'), (1, 'cow')):
        e = res[name + '_edges']
        for b, (lo, hi) in enumerate(((-1e9, e[0]), (e[0], e[1]), (e[1], 1e9))):
            sel = [r for r in rows if r[0] == t and lo <= r[1] < hi]
            n = len(sel)
            out[f'{name}_bin{b}'] = (n, round(sum(r[2] == 'move' for r in sel) / max(n, 1), 3), round(sum(r[2] == 'vanish' for r in sel) / max(n, 1), 3))
    print(label, json.dumps(out), flush=True)
    res[label] = out
summarize(real_rows, 'REAL')
for name in sys.argv[1:]:
    imag = torch.load(A5.BASE / f'transport/recursion_samples_{name}.pt', weights_only=False)[selected].float()
    Rb, K = imag.shape[:2]
    seq = torch.cat([tok[:, 15][:, None, None].expand(-1, K, 1, -1, -1), imag], 2)
    d = C.decode(seq, bank, head)
    idec, isc = d['mob']['real'][..., :3], d['scores']['real'][..., :3]
    est = shifts[scroll.estimate(seq[:, :, :-1], seq[:, :, 1:])]
    cam_ok = (est == real_shift[:, None]).all(-1).cumprod(2).bool()
    steps = [(idec[:, :, k].flatten(0, 1), idec[:, :, k + 1].flatten(0, 1), isc[:, :, k].flatten(0, 1),
              (cam_ok[:, :, k] & (real_shift[:, None, k] == 0).all(-1)).flatten()) for k in range(16)]
    summarize(collect(None, None, steps), name)
print(json.dumps({k: v for k, v in res.items() if k.endswith('edges')}))
