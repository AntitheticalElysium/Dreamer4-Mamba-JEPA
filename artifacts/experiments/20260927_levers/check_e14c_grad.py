"""E14c check (2026-10-02): does the mask term dominate the gradient? (check_e14c_cost: the end-to-end mask1 + skip world's
player / HUD / static errors on moved and blocked steps sit at the baseline's 6k level, as if the rest were under-trained.)
On 20 seeded training batches of 16 windows (the trainer's rows; 16, not 40: memory beside training), teacher-forced: U = the uniform per-token L1 mean, M = lambda x the
mask-normalized mean over the faced tiles of attempts (lambda 1, tworld's --weight mask1). Gradients of U and of M w.r.t. the
backbone and the head parameters: norms, ratio ||g_M|| / ||g_U||, cosine. For a fresh skip world (init seed 7, the trainer's
init) and the trained E14c world.
Reading, declared before running:
  dominates   ||g_M|| / ||g_U|| >= 3 on the backbone for the trained world (the mask term sets the update direction / Adam's scale)
Result (2026-10-02, batches of 16): backbone ratio 2.26 (fresh skip init, cos 0.38), 1.74 (baseline teacher s7, cos 0.08), 5.38
(E14c s7, cos 0.05): dominates TRUE. Head 1.06 / 2.02 / 9.62.
"""
import sys, json, torch, torch.nn.functional as F
sys.path.insert(0, "artifacts/experiments/20260927_levers")
import check_costwhere as C
import tworld as TW

lab = torch.load(TW.LABELS)
main_rows = torch.where(~C.pool["terminal"])[0]
rows_all = torch.cat([main_rows[~torch.isin(main_rows, C.held)], torch.where(C.pool["terminal"])[0]])
HEAD = ("proj.", "choose.", "frame.", "target_gate.")


def anatomy(world):
    world.train()
    groups = {"backbone": [p for n, p in world.named_parameters() if not n.startswith(HEAD)],
              "head": [p for n, p in world.named_parameters() if n.startswith(HEAD)]}
    gen = torch.Generator().manual_seed(123)
    acc = {g: {"U": [], "M": [], "cos": []} for g in groups}
    for _ in range(20):
        idx = rows_all[torch.randint(len(rows_all), (16,), generator=gen)]
        s = C.pool["tokens"][idx].float().to(C.dev); a = F.pad(C.pool["actions"][idx], (0, 1)).to(C.dev)
        m = torch.zeros(16, 5, 81, dtype=torch.bool)
        m.scatter_(2, lab["faced"][idx][..., None], lab["attempt"][idx][..., None]); m = m.to(C.dev)
        with C.autocast_context(C.config):
            err = (world(s, a)[0][:, :5].float() - s[:, 1:]).abs()
        tok = err.mean(-1)
        U = err.mean(); M = (tok * m).sum() / m.sum().clamp(min=1)
        for g, ps in groups.items():
            gu = torch.cat([x.flatten() for x in torch.autograd.grad(U, ps, retain_graph=True, allow_unused=True) if x is not None]).float()
            gm = torch.cat([x.flatten() for x in torch.autograd.grad(M, ps, retain_graph=True, allow_unused=True) if x is not None]).float()
            acc[g]["U"].append(float(gu.norm())); acc[g]["M"].append(float(gm.norm())); acc[g]["cos"].append(float(F.cosine_similarity(gu, gm, dim=0)))
    mean = lambda v: sum(v) / len(v)
    return {g: {"norm_U": mean(x["U"]), "norm_M": mean(x["M"]), "ratio_M_over_U": mean(x["M"]) / mean(x["U"]), "cos": mean(x["cos"])} for g, x in acc.items()}


out = {}
torch.manual_seed(7); torch.cuda.manual_seed_all(7)
fresh = TW.TWorld("corrt", None, "full", "all", 0, True).to(C.dev)
out["fresh_skip_init_s7"] = anatomy(fresh)
print(json.dumps({"fresh_skip_init_s7": out["fresh_skip_init_s7"]}), flush=True)
for p in sys.argv[1:]:
    w, st = C.T.load_world(p, C.dev)
    out[st["name"]] = anatomy(w)
    print(json.dumps({st["name"]: out[st["name"]]}), flush=True)
trained = [k for k in out if "mask1" in k]
out["readings"] = {"dominates": {k: out[k]["backbone"]["ratio_M_over_U"] >= 3 for k in trained}}
print(json.dumps(out))
