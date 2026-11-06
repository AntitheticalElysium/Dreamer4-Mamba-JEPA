"""T3 (PLAN.md): three fine-tunes of P1-F3 checkpoint 10 that differ ONLY in the physical sigma draw of each training
frame, then their evaluation on the audit's 208 pairs.
  C: the original lognormal(-0.4, 1.2) capped at 20.   H: with probability 0.25 log-uniform in [10, 20] instead.
  F: frames whose state t - 1 is asleep (labels; diagnostic) take the band with probability 0.8, others original.
F3's loop (the F2 trainer: online gaze controller, canonical x-loss from the checkpoint's cfg, harmoniser, noise term,
all-parameter clip), 16 x 256 frames in 4 microbatches, 750 updates, LR linear from the checkpoint's 1e-5 to 5.8e-5 over
100 updates then constant; perception frozen. The original lognormal is drawn in every arm, so the global RNG (data
noise, context noise) is consumed identically; band choices come from a separate generator.
Usage (project venv, GPU): MAMBA_DETERMINISTIC=1 python -m artifacts.experiments.20261106_decision_levels.finetune
  train ARM [--smoke N]  |  evaluate ARM      (ARM: C, H, F; evaluate also takes f3 = the untouched checkpoint)
"""
import copy, importlib, json, os, sys, time
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import numpy as np
import torch

from saccade import data as D, model as M, train as TR
from saccade.eval import s1 as S

FZ = importlib.import_module("artifacts.experiments.20261106_decision_levels.frozen")
F2T = importlib.import_module("artifacts.experiments.20261103_p1f2.trainer")
R, A_, CV, C = FZ.R, FZ.A_, FZ.CV, F2T.C
HERE = Path(__file__).resolve().parent
OUT = HERE / "finetune"
CKPT = TR.ROOT / CV.CKPT
UPDATES, WARMUP, LR, STEPS = 750, 100, 5.8e-5, 256
Q_H, Q_F, BAND, BAND_SEED = 0.25, 0.8, (10.0, 20.0), 611
ORIG = TR.sample_sigma


def asleep_prev(split):
    """[train frames] bool, aligned with D.frame_offsets(split['train']): frame t's state t - 1 is asleep (labels)."""
    L = np.load(CV.LABELS / "s0_train_labels.npz")
    s_off, sleep = np.r_[0, np.cumsum(L["ep_T"] + 1)], L["s_sleep"]
    where = {(int(s), int(o)): e for e, (s, o) in enumerate(zip(L["ep_shard"], L["ep_slot"]))}
    off = D.frame_offsets(split["train"])
    out = np.zeros(int(off[-1]), bool)
    for i, ep in enumerate(split["train"]):
        e = where[(ep.shard, ep.slot)]
        n = int(off[i + 1] - off[i])
        assert n == s_off[e + 1] - s_off[e], "labels and split disagree on an episode's length"
        out[off[i] + 1:off[i + 1]] = sleep[s_off[e]:s_off[e] + n - 1]
    return out


class Sigma:
    """Replaces TR.sample_sigma inside p1_losses: its first call per microbatch draws the physical levels (the arm's
    rule), the second the glimpse levels (original). `target` = the microbatch's global frame indices (-1: none)."""

    def __init__(self, arm, flags):
        self.arm, self.flags, self.gen = arm, flags, torch.Generator(device="cuda").manual_seed(BAND_SEED)
        self.calls, self.target, self.band = 0, None, []

    def __call__(self, shape, cfg, device, gen=None):
        sig = ORIG(shape, cfg, device, gen)
        self.calls += 1
        if self.calls != 1 or self.arm == "C":
            return sig
        u = torch.rand(shape, device=device, generator=self.gen)
        band = BAND[0] * (BAND[1] / BAND[0]) ** torch.rand(shape, device=device, generator=self.gen)
        if self.arm == "H":
            m = u < Q_H
        else:
            asleep = self.flags[np.maximum(self.target, 0)] & (self.target >= 0)
            m = (u < Q_F) & torch.as_tensor(asleep, device=device)
        self.band.append(float(m.float().mean()))
        return torch.where(m, band, sig)


def train(arm, smoke=None):
    assert os.environ.get("MAMBA_DETERMINISTIC") == "1" and arm in ("C", "H", "F")
    dev = torch.device("cuda")
    ck = torch.load(CKPT, weights_only=False, map_location=dev)
    cfg, d, P = ck["cfg"], ck["cfg"]["data"], ck["cfg"]["data"]["context_prefix"]
    out = OUT / (f"smoke_{arm}" if smoke else arm)
    assert not (out / "ft.pt").exists(), "arm already trained"
    out.mkdir(parents=True, exist_ok=True)
    split = D.load_split(cfg)
    offsets = D.frame_offsets(split["train"])
    targets = np.load(TR.ROOT / cfg["p1"]["out_dir"] / "targets/change_targets.npy", mmap_mode="r")
    flags = asleep_prev(split)
    model = M.M0(cfg).to(dev)
    model.load_state_dict(ck["model"])
    model.denoiser.checkpoint = cfg["p1"]["activation_checkpointing"]
    model.perception.requires_grad_(False)                              # tokens and the frozen readers stay valid
    o = cfg["optimizer"]
    opt = torch.optim.AdamW(model.param_groups(), lr=o["lr"], betas=tuple(o["betas"]), eps=o["eps"])
    opt.load_state_dict(ck["opt"])
    streams = D.Streams(split["train"], d["streams"], P, seed=ck["world_seed"], targets_offsets=offsets)
    streams.load_state(ck["streams"])
    controller = C.Controller(model, cfg, C.config(), out, dev)
    controller.load_state(dict(ck["fixation_controller"], trace_rows=0, log_bytes=0))   # the head, optimiser and RNGs;
                                                                         # this arm's own gaze trace / log start empty
    carry, cache = ck["carry"], ck["cache"]
    torch.set_rng_state(ck["rng"]["cpu"].cpu()), torch.cuda.set_rng_state(ck["rng"]["cuda"].cpu())
    lr0 = o["lr"] * TR.lr_factor(ck["state"]["frames"], cfg)
    perception0 = {k: v.clone() for k, v in model.perception.state_dict().items()}
    del ck
    sigma = Sigma(arm, flags)
    TR.sample_sigma = sigma
    ident = {"arm": arm, "checkpoint_sha256": TR.file_sha256(CKPT), "script_sha256": TR.file_sha256(Path(__file__)),
             "plan_sha256": TR.file_sha256(HERE / "PLAN.md"), "updates": UPDATES, "lr": LR, "warmup": WARMUP, "steps": STEPS,
             "q_h": Q_H, "q_f": Q_F, "band": BAND, "lr0": lr0, "perception": "frozen"}
    log = open(out / "log.jsonl", "a")
    log.write(json.dumps({"identity": ident}) + "\n"), log.flush()
    micro, n_up = d["accumulate_at_256"], smoke or UPDATES
    t0, logs, frames = time.time(), [], 0
    for u in range(n_up):
        chunk = streams.next(STEPS)
        controller.prepare(chunk, streams)
        lr = lr0 + (LR - lr0) * min(1.0, u / WARMUP)
        for g in opt.param_groups:
            g["lr"] = lr * g["lr_scale"]
        opt.zero_grad(set_to_none=True)
        for rows in np.array_split(np.arange(d["streams"]), micro):
            sub = D.select(chunk, rows)
            sigma.target, sigma.calls = np.asarray(sub["target"]), 0
            terms, c_new, cache_new = TR.p1_losses(model, sub, M.carry_rows(carry, rows), cache[rows], targets, cfg)
            assert sigma.calls == 2, "expected one physical and one glimpse draw"
            ((model.harmoniser(TR.groups(terms)) + terms.get("noise", 0)) / micro).backward()
            M.carry_put(carry, rows, c_new)
            cache[rows] = cache_new
            logs.append({k: float(v.detach()) for k, v in terms.items()})
        del terms, c_new, cache_new
        gn = float(torch.nn.utils.clip_grad_norm_(model.parameters(), o["grad_clip"]))
        opt.step()
        n = int(chunk["valid"][:, P:].sum())
        controller.learn(n)
        frames += n
        if (u + 1) % 25 == 0 or u == 0 or smoke:
            rec = {"update": u + 1, "frames": frames, "lr": lr, "grad_norm": gn,
                   "band_share": float(np.mean(sigma.band)) if sigma.band else 0.0,
                   **{k: float(np.mean([l[k] for l in logs])) for k in logs[0]},
                   "sec_per_update": (time.time() - t0) / (u + 1), "peak_gib": torch.cuda.max_memory_allocated() / 2 ** 30}
            log.write(json.dumps(rec) + "\n"), log.flush()
            print(json.dumps(rec), flush=True)
            logs, sigma.band = [], []
    TR.sample_sigma = ORIG
    assert all(torch.equal(v, perception0[k]) for k, v in model.perception.state_dict().items()), "perception moved"
    if not smoke:
        torch.save({"model": model.state_dict(), "controller": controller.state(), "identity": ident}, out / "ft.pt")
        mon = TR.monitor(model, cfg, split["monitor"])
        log.write(json.dumps({"monitor": mon}) + "\n"), log.flush()
        print("monitor", json.dumps(mon), flush=True)
    controller.close()


@torch.no_grad()
def evaluate(arm):
    """The audit's 208 pairs with its base-5 and prefix-20 recipes (its physical loop, seeds and readers), events by the
    frozen readers; the pure-noise first evaluation at sigma 20 for both energy members; for f3 the numbers must equal
    the audit's (R0 of this instrument)."""
    assert os.environ.get("MAMBA_DETERMINISTIC") == "1"
    state = None if arm == "f3" else torch.load(OUT / arm / "ft.pt", weights_only=False, map_location="cpu")["model"]
    model, cfg, L = FZ.setup(state)
    enc = S.encode(model, L, cfg)
    rd = FZ.readers()
    den, shapes = model.denoiser, [tuple(enc[0][k].shape[1:]) for k in ("p", "g", "v")]
    lo, K = cfg["edm"]["inference_context_sigma"], A_.K
    bank = torch.load(R.HERE / "readings/bank.pt", map_location="cpu", weights_only=False)
    T = CV.transitions(L)
    rows = []
    for r in A_.select(bank, False):
        if r["kind"] in R.SLOT:
            d = CV.IF.DIRS[L["s_dir"][int(T["g"][r["j"]])]]
            r["face_cell"] = int((3 + d[0]) * 9 + 4 + d[1])
        P2, Ps, b, bp, caches, fx = R.pair_state(model, cfg, enc, r)
        A, B = (A_.vector(Ps.p[m].float(), Ps.v[m].float(), r["kind"]) for m in (0, 1))
        reader = rd[R.SLOT[r["kind"]] if r["kind"] in R.SLOT else "asleep"]
        rec = {"kind": r["kind"], "e": r["e"], "stratum": r["stratum"], "truth": r["truth"], "members": []}
        for m in (0, 1):
            a = torch.full((K, 1), r["actions"][m], device=CV.DEV, dtype=torch.long)
            mem = model.belief.memory(bp[m:m + 1])[:, None].expand(K, -1, -1, -1)
            fix, cache = fx.expand(K, -1)[:, None], R.stack(caches[lo][m], K)
            ev = {}
            for recipe in ("base5", "prefix20"):
                p, v, _ = A_.physical(den, shapes, a, mem, fix, cache, cfg, r["j"], r["kind"], reader, A, B, recipe, record=False)
                q = A_.metric(p[:, 0].float(), v[:, 0].float(), r["kind"], reader)
                ev[recipe] = ((q[:, 1] < 0.5) if r["kind"] not in R.SLOT else (q.argmax(-1) == r["truth"]["count"][1] + 1)).float().mean().item()
            gen = torch.Generator(device=CV.DEV).manual_seed(r["j"])
            x = (20 * torch.randn((K, 1, *shapes[0]), device=CV.DEV, generator=gen), torch.zeros((K, 1, *shapes[1]), device=CV.DEV),
                 20 * torch.randn((K, 1, *shapes[2]), device=CV.DEV, generator=gen))
            D_ = FZ.first_eval(den, x, 20.0, a, mem, fix, cache, cfg)
            ev["pure20_f"] = float((((A_.vector(D_[0][:, 0].float(), D_[2][:, 0].float(), r["kind"]) - A) * (B - A)).sum(-1) / (B - A).square().sum()).mean())
            rec["members"].append(ev)
        rows.append(rec)
    res = summary(rows)
    if arm == "f3":                                                     # R0: this instrument reproduces the audit
        audit = json.loads((A_.HERE / "readings/SUMMARY.json").read_text())["groups"]
        for kind in ("sleep", "energy", "coal", "wood"):
            for recipe in ("base5", "prefix20"):
                for m in (0, 1):
                    mine, theirs = res[kind][recipe][f"member{m}"]["est"], audit[f"{kind}/all"][recipe][str(m)]["mean"]
                    assert abs(mine - theirs) < 1e-9, ("R0: differs from the audit", kind, recipe, m, mine, theirs)
        res["r0"] = "reproduces the audit's 16 event rates exactly"
    if arm != "f3":
        res["monitor"] = [json.loads(l)["monitor"] for l in open(OUT / arm / "log.jsonl") if '"monitor"' in l][-1]
    (HERE / "readings").mkdir(exist_ok=True)
    torch.save(rows, HERE / f"readings/eval_{arm}.pt")
    (HERE / f"readings/eval_{arm}.json").write_text(json.dumps(res, indent=1))
    print(arm, json.dumps(res, indent=1), flush=True)


def summary(rows):
    """Event rates per family and member at both starts, episode-bootstrap 95% (2,000, seed 1106)."""
    out = {}
    fam = {"sleep": "sleep", "energy": "energy", "coal": "coal", "wood": "wood"}
    for kind in fam:
        P_ = [r for r in rows if r["kind"] == kind]
        ep = np.array([r["e"] for r in P_])
        out[kind] = {}
        for recipe in ("base5", "prefix20", "pure20_f"):
            x = np.array([[r["members"][m][recipe] for m in (0, 1)] for r in P_])
            out[kind][recipe] = {"member0": R.boot(x[:, 0], ep), "member1": R.boot(x[:, 1], ep), "member1_minus_0": R.boot(x[:, 1] - x[:, 0], ep)}
    return out


if __name__ == "__main__":
    smoke = int(sys.argv[sys.argv.index("--smoke") + 1]) if "--smoke" in sys.argv else None
    {"train": lambda a: train(a, smoke), "evaluate": evaluate}[sys.argv[1]](sys.argv[2])
