"""E14c check (2026-10-02): where does the end-to-end mask1 + skip world lose? (teval s7: interact 0.863 -> 0.652, but blocked
0.301 -> 0.926, idle 0.442 -> 0.882, all +72%, gen_16 +0.160.) Held pool windows (tworld's 2,048), teacher-forced. Per action
class (move & the true frame scrolls = moved; move & no scroll = blocked; DO / place = interact; sleep; everything else = idle)
x token class (check_costwhere's classes): mean L1 of each world, and the absolute increase over the baseline. Per action
class: the drawn scroll (scroll.estimate(input frame, prediction)) vs the true scroll -- share of transitions drawn with a
scroll the truth does not have (false scroll) and missing one it has (missed scroll).
Usage: check_e14c_cost.py <baseline.pt> <world.pt> ...
"""
import sys, json, torch, torch.nn.functional as F
sys.path.insert(0, "artifacts/experiments/20260927_levers")
import check_costwhere as C
from scroll import estimate

ACLS = ("moved", "blocked", "interact", "sleep", "idle")


@torch.no_grad()
def run(world):
    L = torch.zeros(len(ACLS), len(C.CLASSES)); N = torch.zeros(len(ACLS), len(C.CLASSES))
    fs = torch.zeros(len(ACLS)); ms = torch.zeros(len(ACLS)); nt = torch.zeros(len(ACLS))
    for i in range(0, len(C.held), 64):
        r = C.held[i:i + 64]; s = C.pool["tokens"][r].float(); a = C.pool["actions"][r]; alive = C.pool["alive"][r][:, 1:]
        with C.autocast_context(C.config):
            pred = world(s.to(C.dev), F.pad(a, (0, 1)).to(C.dev))[0][:, :5].float().cpu()
        err = (pred - s[:, 1:]).abs().mean(-1)
        k = C.classes(r, s)
        for t in range(5):
            tru = estimate(s[:, t], s[:, t + 1]); drawn = estimate(s[:, t], pred[:, t])
            move = (a[:, t] >= 1) & (a[:, t] <= 4)
            ac = torch.full((len(r),), ACLS.index("idle"))
            ac[move & (tru != 0)] = ACLS.index("moved"); ac[move & (tru == 0)] = ACLS.index("blocked")
            ac[torch.isin(a[:, t], torch.tensor([5, 7, 8, 9, 10]))] = ACLS.index("interact"); ac[a[:, t] == 6] = ACLS.index("sleep")
            for c in range(len(ACLS)):
                m = (ac == c) & alive[:, t]
                if not m.any():
                    continue
                nt[c] += m.sum(); fs[c] += ((drawn != 0) & (tru == 0) & m).sum(); ms[c] += ((drawn == 0) & (tru != 0) & m).sum()
                for kk in range(len(C.CLASSES)):
                    mm = m[:, None] & (k[:, t] == kk)
                    L[c, kk] += err[:, t][mm].sum(); N[c, kk] += mm.sum()
    return L / N.clamp(min=1), N, fs / nt.clamp(min=1), ms / nt.clamp(min=1)


paths = sys.argv[1:]
base, st0 = C.T.load_world(paths[0], C.dev)
Lb, N, fsb, msb = run(base)
out = {"baseline": st0["name"], "tokens": {a: {c: int(N[i, j]) for j, c in enumerate(C.CLASSES)} for i, a in enumerate(ACLS)},
       st0["name"]: {"l1": {a: {c: round(float(Lb[i, j]), 4) for j, c in enumerate(C.CLASSES)} for i, a in enumerate(ACLS)},
                     "false_scroll": {a: round(float(fsb[i]), 4) for i, a in enumerate(ACLS)},
                     "missed_scroll": {a: round(float(msb[i]), 4) for i, a in enumerate(ACLS)}}}
for p in paths[1:]:
    w, st = C.T.load_world(p, C.dev)
    L, _, fs, ms = run(w)
    inc = (L - Lb) * N
    out[st["name"]] = {"l1": {a: {c: round(float(L[i, j]), 4) for j, c in enumerate(C.CLASSES)} for i, a in enumerate(ACLS)},
                       "share_of_total_increase": {a: {c: round(float(inc[i, j] / inc.sum()), 4) for j, c in enumerate(C.CLASSES)} for i, a in enumerate(ACLS)},
                       "false_scroll": {a: round(float(fs[i]), 4) for i, a in enumerate(ACLS)},
                       "missed_scroll": {a: round(float(ms[i]), 4) for i, a in enumerate(ACLS)}}
    print(json.dumps({st["name"]: out[st["name"]]}), flush=True)
print(json.dumps(out))
