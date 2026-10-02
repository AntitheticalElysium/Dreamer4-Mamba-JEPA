"""Item 4 check (2026-10-02): is the generate candidate at the player token still usable in the dosed worlds? (check_facing_h:
h at token 31 encodes the next facing in every world (0.95-0.98), but the dosed heads put ~0 weight on generate there on turns:
mask1 + skip self 0.883 / gen 0.000; baseline 0.351 / 0.540.) Held windows, move transitions where the true facing changes (a
turn): per world, L1 to the true next player token of (i) the generate candidate (layer-normed proj output, the token the head
would draw if it generated), (ii) the copy candidate (the current token), (iii) the head's actual output.
Reading, declared before running: gen_degraded = the dosed world's generate-candidate L1 on turns >= the copy candidate's (then
copying is the rational choice and the turn is lost because the shared generate path no longer draws the player), while the
baseline's generate candidate beats copy.
Usage: check_gencand.py <world.pt> ...
"""
import sys, json, torch, torch.nn.functional as F
sys.path.insert(0, "artifacts/experiments/20260927_levers")
import check_costwhere as C
probes = C.T.Probes(C.T.build_cache("raw", torch.device("cpu")), *C.T.split())
out = {}
for p in sys.argv[1:]:
    w, st = C.T.load_world(p, C.dev)
    G, P, Cp, n = 0.0, 0.0, 0.0, 0
    with torch.no_grad():
        for i in range(0, len(C.held), 64):
            r = C.held[i:i + 64]; s = C.pool["tokens"][r].float(); a = C.pool["actions"][r]; alive = C.pool["alive"][r][:, 1:]
            with C.autocast_context(C.config):
                pred, _, gen = w(s.to(C.dev), F.pad(a, (0, 1)).to(C.dev))
            gen = F.layer_norm(gen[:, :5, 31].float(), (192,)).cpu(); pred = pred[:, :5, 31].float().cpu()
            for t in range(5):
                m = (a[:, t] >= 1) & (a[:, t] <= 4) & alive[:, t]
                m = m & (probes.facing(s[:, t, 31]).argmax(-1) != probes.facing(s[:, t + 1, 31]).argmax(-1))
                if not m.any():
                    continue
                y = s[m, t + 1, 31]
                G += float((gen[m, t] - y).abs().mean(-1).sum()); P += float((pred[m, t] - y).abs().mean(-1).sum())
                Cp += float((s[m, t, 31] - y).abs().mean(-1).sum()); n += int(m.sum())
    out[st["name"]] = {"turns": n, "l1_generate_candidate": round(G / n, 4), "l1_copy": round(Cp / n, 4), "l1_output": round(P / n, 4)}
    print(json.dumps({st["name"]: out[st["name"]]}), flush=True)
b = [k for k in out if k.endswith("teacher_s7_u18000")]
if b:
    out["readings"] = {k: {"gen_degraded": v["l1_generate_candidate"] >= v["l1_copy"] and out[b[0]]["l1_generate_candidate"] < out[b[0]]["l1_copy"]} for k, v in out.items() if k != b[0]}
print(json.dumps(out))
