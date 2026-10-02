"""Item 4 check (2026-10-02): do the dosed worlds mispredict the agent's FACING? (e14d_cost: the player token's L1 on moved / blocked
steps is +136-142% at dose x31 and worse at x304, more than any other class; the player sprite encodes facing, which also defines
the faced tile the mask doses.) Held pool windows, teacher-forced. Per action class (moved / blocked / interact / sleep / idle, as
check_e14c_cost): accuracy of the facing class read by teval's facing probe on the PREDICTED player token (31) vs on the TRUE next
player token; and, among transitions where the true facing changes, the share predicted as still the old facing (missed turn).
Usage: check_facing.py <world.pt> ...
"""
import sys, json, torch, torch.nn.functional as F
sys.path.insert(0, "artifacts/experiments/20260927_levers")
import check_costwhere as C
import check_e14c_cost as E
from scroll import estimate

probes = C.T.Probes(C.T.build_cache("raw", torch.device("cpu")), *C.T.split())
out = {}
for p in sys.argv[1:]:
    w, st = C.T.load_world(p, C.dev)
    acc = {a: [0, 0] for a in E.ACLS}; turn = {a: [0, 0, 0] for a in E.ACLS}
    with torch.no_grad():
        for i in range(0, len(C.held), 64):
            r = C.held[i:i + 64]; s = C.pool["tokens"][r].float(); a = C.pool["actions"][r]; alive = C.pool["alive"][r][:, 1:]
            with C.autocast_context(C.config):
                pred = w(s.to(C.dev), F.pad(a, (0, 1)).to(C.dev))[0][:, :5].float().cpu()
            for t in range(5):
                f0 = probes.facing(s[:, t, 31]).argmax(-1); f1 = probes.facing(s[:, t + 1, 31]).argmax(-1)
                fp = probes.facing(pred[:, t, 31]).argmax(-1)
                tru = estimate(s[:, t], s[:, t + 1]); move = (a[:, t] >= 1) & (a[:, t] <= 4)
                ac = torch.full((len(r),), E.ACLS.index("idle"))
                ac[move & (tru != 0)] = E.ACLS.index("moved"); ac[move & (tru == 0)] = E.ACLS.index("blocked")
                ac[torch.isin(a[:, t], torch.tensor([5, 7, 8, 9, 10]))] = E.ACLS.index("interact"); ac[a[:, t] == 6] = E.ACLS.index("sleep")
                for c, name in enumerate(E.ACLS):
                    m = (ac == c) & alive[:, t]
                    acc[name][0] += int((fp == f1)[m].sum()); acc[name][1] += int(m.sum())
                    ch = m & (f0 != f1)
                    turn[name][0] += int(ch.sum()); turn[name][1] += int((fp == f0)[ch].sum()); turn[name][2] += int((fp == f1)[ch].sum())
    out[st["name"]] = {a: {"facing_acc": round(acc[a][0] / max(acc[a][1], 1), 4), "n": acc[a][1], "turns": turn[a][0],
                           "turn_missed": round(turn[a][1] / max(turn[a][0], 1), 4), "turn_right": round(turn[a][2] / max(turn[a][0], 1), 4)}
                       for a in E.ACLS}
    print(json.dumps({st["name"]: out[st["name"]]}), flush=True)
print(json.dumps(out))
