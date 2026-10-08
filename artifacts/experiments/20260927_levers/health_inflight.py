"""In-flight check of a training world (2026-10-08, E21): token-63 generator fidelity, the phase (since = 6) in h63, emitted hits /
fresh / false drops (and those inside the cooldown window), the event head's token-63 AUC, on the health_chain subset (held).
Usage: health_inflight.py <world name (its __w15__ext.pt in health_chain_v1)> ..."""
import sys, torch, numpy as np, hashlib
import torch.nn.functional as F
sys.path.insert(0, 'artifacts/experiments/20260927_levers'); sys.path.insert(0, 'artifacts/experiments/20260926_diagnosis'); sys.path.insert(0, 'artifacts/experiments/20260921_readout_ladder')
import health_evidence as HE, teval as T
OUT = HE.OUT
sub = torch.load(OUT / 'subset.pt', weights_only=False); st = torch.load(OUT / 'strata.pt', weights_only=False)
mm = np.memmap(OUT / 'subset.f16', dtype=np.float16, mode='r', shape=tuple(sub['shape']))
t14 = torch.from_numpy(np.array(mm[:, 14, 63])).float(); t15 = torch.from_numpy(np.array(mm[:, 15, 63])).float()
cls = sub['classes']; hit, unch = cls == 1, cls == 2; keep = hit | unch; N = len(cls); allm = torch.ones(N, dtype=torch.bool)
fit = sub['fit']; inner = fit & torch.tensor([int(hashlib.sha256((e + '/inner').encode()).hexdigest(), 16) % 4 == 0 for e, _ in sub['ids']]); held = ~fit
meta, tr, ts = T.split(); P = T.Probes(T.build_cache('raw', torch.device('cpu')), meta, tr, ts)
read = lambda hud: (P.hud(hud.float().flatten(-2))[..., 0] * 9).float()
cur = torch.cat([read(torch.from_numpy(np.array(mm[i:i + 2048, 14, 63:81]))) for i in range(0, N, 2048)])
cl = (t14 - t15).abs().mean(-1); since = st['since']; cool = (since >= 1) & (since <= 5)
for n in sys.argv[1:]:
    r = torch.load(OUT / f'{n}__w15__ext.pt', weights_only=False)
    g = F.layer_norm(r['gen63'].float(), (192,)); gl = (g - t15).abs().mean(-1)
    em = read(r['hud']) < cur - 1.5
    h63 = r['h_ext'][:, 0].float()
    pr = HE.weighted_probe(h63, since == 6, torch.ones(N), allm & fit & ~inner, allm & inner, steps=2000)
    line = (f"{n[-48:]:48s} gen L1 hit {gl[hit & held].mean():.3f} beats copy {float((gl[hit & held] < cl[hit & held]).float().mean()):.3f}"
            f" | gen w hit {float(r['weights63'][hit & held][:, 5].mean()):.3f} | since6 h63 {HE.auc(pr[held], (since == 6)[held]):.3f}"
            f" | emitted hits {int((em & hit & held).sum())} fresh {int((em & hit & sub['fresh'] & held).sum())} false {int((em & unch & held).sum())} (cooldown {int((em & unch & held & cool).sum())})")
    if 'event63' in r:
        e = r['event63'].float(); line += f" | event63 AUC {HE.auc(e[held & keep], hit[held & keep]):.3f}"
    print(line, flush=True)
