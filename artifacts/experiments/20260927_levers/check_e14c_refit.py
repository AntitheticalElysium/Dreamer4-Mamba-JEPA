"""E14c check (2026-10-02): is the E14c world's damage in its head or in its backbone? (check_e14c_cost s7: idle / sleep static
and interact near-player errors are worse than even the baseline's 6k snapshot -- active damage, beyond under-training.)
Substitution: the E14c world's BACKBONE (every non-head parameter) in a plain corrt TWorld with a fresh LINEAR head (the module's
own init, seed 0), trained head-only with the uniform per-token L1 (headfit's protocol: 3,000 updates, batches of 40 seeded
training windows, AdamW 1e-3, wd 0.01, 100 warmup, clip 1). The same for the baseline backbone (control: E14a showed a fresh
uniform head reproduces it). Then check_e14c_cost's table (held windows: action class x token class L1) and consfit-style
consequence catch (headfit.evaluate) for: baseline, baseline-refit, E14c, E14c-backbone-refit.
Reading, declared before running:
  head_damage       E14c-backbone-refit idle and sleep static L1 within 10% of the baseline-refit's: the backbone is intact,
                    the damage sits in the skip head
  backbone_damage   otherwise (the representation itself was degraded by the end-to-end dose)
Usage: check_e14c_refit.py <baseline.pt> <e14c.pt>
"""
import sys, json, torch, torch.nn.functional as F
sys.path.insert(0, "artifacts/experiments/20260927_levers")
import check_costwhere as C
import check_e14c_cost as E
import headfit as HF
import tworld as TW

HEAD = ("proj.", "choose.", "frame.", "target_gate.")
lab = torch.load(TW.LABELS)
main_rows = torch.where(~C.pool["terminal"])[0]
train_rows = torch.cat([main_rows[~torch.isin(main_rows, C.held)], torch.where(C.pool["terminal"])[0]])


def refit(src):
    torch.manual_seed(0)
    w = TW.TWorld("corrt").to(C.dev)
    sd = w.state_dict()
    sd.update({k: v for k, v in src.state_dict().items() if not k.startswith(HEAD) and k in sd})
    w.load_state_dict(sd)
    for n, p in w.named_parameters():
        p.requires_grad_(n.startswith(HEAD))
    params = [p for n, p in w.named_parameters() if n.startswith(HEAD)]
    opt = torch.optim.AdamW(params, lr=1e-3, weight_decay=0.01)
    gen = torch.Generator().manual_seed(11)
    w.train()
    for u in range(3000):
        for g in opt.param_groups:
            g["lr"] = 1e-3 * min(1.0, (u + 1) / 100)
        rows = train_rows[torch.randint(len(train_rows), (40,), generator=gen)]
        s = C.pool["tokens"][rows].float().to(C.dev); a = C.pool["actions"][rows].to(C.dev)
        with C.autocast_context(C.config):
            loss = HF.per_token(w, s, a, "l1", C.dev).mean()
        opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step()
    return w.eval()


base, _ = C.T.load_world(sys.argv[1], C.dev)
e14c, st = C.T.load_world(sys.argv[2], C.dev)
probes = HF.T.Probes(HF.T.build_cache("raw", torch.device("cpu")), *HF.T.split())
out = {}
for name, w in (("baseline", base), ("baseline_refit", refit(base)), ("e14c", e14c), ("e14c_backbone_refit", refit(e14c))):
    L, _, fs, ms = E.run(w)
    cf = HF.evaluate(w, C.pool, lab, C.held, probes, C.dev, C.config)
    out[name] = {"l1": {a: {c: round(float(L[i, j]), 4) for j, c in enumerate(C.CLASSES)} for i, a in enumerate(E.ACLS)},
                 "consequence": {k: (round(v, 4) if isinstance(v, float) else v) for k, v in cf.items()}}
    print(json.dumps({name: out[name]}), flush=True)
r, b = out["e14c_backbone_refit"]["l1"], out["baseline_refit"]["l1"]
out["readings"] = {"head_damage": all(r[a]["static"] <= 1.1 * b[a]["static"] for a in ("idle", "sleep"))}
out["readings"]["backbone_damage"] = not out["readings"]["head_damage"]
print(json.dumps(out))
