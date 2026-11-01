"""Fine-tune P1 world seed 1's denoiser from checkpoint 10 with the random fixation prior (arm `random`) or the guided
placement rule (arm `guided`, placement_map.py); everything else identical (PLAN.md).

Both arms: the checkpoint's model, AdamW moments, stream state (episode order continues), Mamba carry and torch RNG;
the prior is drawn for every frame in both arms (the guided arm then overrides attempt frames), so episode order,
noise draws and all other fixations match. Only the denoiser trains, on P1's own G1 term (train.p1_losses) divided by
the checkpoint's harmoniser sigma_G1, clipped at 1.0 over the denoiser; encoder, belief, heads and harmoniser are
frozen. Learning rate: linear from the checkpoint's 1e-5 to LR over WARMUP updates, then constant. Chunks of 16
streams x 256 steps in 4 microbatches (P1's second half). Checkpoints (with exact resume state) at CKPTS, each followed
by P1's fixed monitor.
Usage (project venv, GPU; long runs via systemd-run): python -m artifacts.experiments.20261030_fovea_placement.finetune
  ARM UNTIL_UPDATE [--smoke N]   (resumes from finetune/ARM/latest.pt if present; --smoke: N updates, nothing saved;
  --out DIR --ckpts A,B --no-monitor: resume tests only)
"""
import argparse, json, time
from pathlib import Path

import numpy as np
import torch

from saccade import data as D, model as M, train as TR

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
BASE = ROOT / "artifacts/saccade_m0/p1_world1/ckpt_10.pt"
LR, WARMUP, CKPTS, STEPS = 5.8e-5, 100, (300, 750, 1500), 256           # LR: P1's at checkpoint 5 (accounting H)


class GuidedStreams(D.Streams):
    """D.Streams.next with each new frame's fixation replaced by `place[frame]` where it is >= 0. The prior is always
    drawn first, so the random state matches the random arm exactly."""

    def __init__(self, *args, place=None, **kw):
        super().__init__(*args, **kw)
        self.place = place

    def next(self, steps: int) -> dict:
        recs = []
        for i in range(len(self.cur)):
            new = []
            for _ in range(steps):
                (e, uid), t = self.cur[i], self.t[i]
                fix = tuple(int(x) for x in D.fixation_prior(self.rng, ()))
                if self.place is not None and self.place[self.offsets[e] + t, 0] >= 0:
                    fix = tuple(int(x) for x in self.place[self.offsets[e] + t])
                new.append((e, uid, t, *fix))
                if t == self.eps[e].T:
                    self.cur[i], self.t[i] = self._pull(), 0
                else:
                    self.t[i] = t + 1
            pad = [None] * (self.prefix - len(self.hist[i]))
            recs.append(pad + self.hist[i] + new)
            self.hist[i] = (self.hist[i] + new)[-self.prefix:]
        return D.gather(self.eps, recs, self.offsets)


def lr_at(update, lr0):
    return lr0 + (LR - lr0) * min(1.0, update / WARMUP)


def coverage(chunk, bits, place, P):
    """Manipulation check over the chunk's new frames t >= 1: [frames, guided frames, target frames, target frames
    whose cells the frame's glimpse covers]."""
    idx = chunk["target"][:, P:]
    ok = idx >= 0
    tb = np.unpackbits(bits[idx[ok]], axis=1)[:, :81].astype(bool)
    win = M.window(torch.as_tensor(chunk["fixation"][:, P:][ok])).numpy()
    has = tb.any(1)
    covered = has & np.take_along_axis(tb, win, 1).any(1)
    guided = 0 if place is None else int((place[idx[ok], 0] >= 0).sum())
    return np.array([int(ok.sum()), guided, int(has.sum()), int(covered.sum())])


def save(path, model, opt, streams, carry, cache, st, arm, ident):
    torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "streams": streams.state(), "carry": carry,
                "cache": cache, "state": dict(st), "arm": arm, "identity": ident,
                "rng": {"cpu": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state()}}, path)


def run(arm, until, smoke=None, root=None, ckpts=CKPTS, monitor=True):
    dev = torch.device("cuda")
    ck = torch.load(BASE, weights_only=False, map_location=dev)
    cfg, d, P = ck["cfg"], ck["cfg"]["data"], ck["cfg"]["data"]["context_prefix"]
    out = (root or HERE / "finetune") / arm
    out.mkdir(parents=True, exist_ok=True)
    split = D.load_split(cfg)
    offsets = D.frame_offsets(split["train"])
    targets = np.load(ROOT / cfg["p1"]["out_dir"] / "targets/change_targets.npy", mmap_mode="r")
    place = np.load(HERE / "place_train.npy")
    bits = np.load(HERE / "targets_train.npy")
    assert len(place) == len(bits) == int(offsets[-1]) == len(targets)
    prov = TR.run_provenance(cfg)
    ident = {"base_sha256": TR.file_sha256(BASE), "place_sha256": TR.file_sha256(HERE / "place_train.npy"),
             "targets_bits_sha256": TR.file_sha256(HERE / "targets_train.npy"),
             "script_sha256": TR.file_sha256(Path(__file__)), "lr": LR, "warmup": WARMUP, "steps": STEPS,
             **{k: v for k, v in prov.items() if k != "git_commit"}}      # commits of logs must not block a resume
    model = M.M0(cfg).to(dev)
    model.load_state_dict(ck["model"])
    model.denoiser.checkpoint = cfg["p1"]["activation_checkpointing"]
    for name, prm in model.named_parameters():
        prm.requires_grad_(name.startswith("denoiser."))
    o = cfg["optimizer"]
    opt = torch.optim.AdamW(model.param_groups(), lr=o["lr"], betas=tuple(o["betas"]), eps=o["eps"])
    opt.load_state_dict(ck["opt"])
    streams = GuidedStreams(split["train"], d["streams"], P, seed=ck["world_seed"], targets_offsets=offsets,
                            place=place if arm == "guided" else None)
    streams.load_state(ck["streams"])
    carry, cache = ck["carry"], ck["cache"]
    torch.set_rng_state(ck["rng"]["cpu"].cpu()), torch.cuda.set_rng_state(ck["rng"]["cuda"].cpu())
    lr0 = o["lr"] * TR.lr_factor(ck["state"]["frames"], cfg)               # the checkpoint's own learning rate
    w_g1 = 1.0 / float(model.harmoniser.log_sigma.exp()[model.harmoniser.groups.index("G1")])
    st = {"update": 0, "frames": 0, "target_frames": 0, "covered": 0}
    latest = out / "latest.pt"
    if latest.exists() and not smoke:
        r = torch.load(latest, weights_only=False, map_location=dev)
        if r["arm"] != arm or r["identity"] != ident:
            raise ValueError("Resume identity differs (arm, base, placement, script, sources or settings)")
        model.load_state_dict(r["model"]), opt.load_state_dict(r["opt"]), streams.load_state(r["streams"])
        carry, cache, st = r["carry"], r["cache"], r["state"]
        torch.set_rng_state(r["rng"]["cpu"].cpu()), torch.cuda.set_rng_state(r["rng"]["cuda"].cpu())
        print(f"resumed at update {st['update']}", flush=True)
    del ck
    params = [p for p in model.parameters() if p.requires_grad]
    micro = d["accumulate_at_256"]
    log = None if smoke else open(out / "log.jsonl", "a")
    if log:
        log.write(json.dumps({"start_update": st["update"], "until": until, "git_commit": prov["git_commit"],
                              "identity": ident, "lr0": lr0, "g1_weight": w_g1}) + "\n"), log.flush()
    stop = st["update"] + smoke if smoke else until
    t0, u0, logs, cov = time.time(), st["update"], [], np.zeros(4, np.int64)
    while st["update"] < stop:
        chunk = streams.next(STEPS)
        lr = lr_at(st["update"], lr0)
        for g in opt.param_groups:
            g["lr"] = lr * g["lr_scale"]
        opt.zero_grad(set_to_none=True)
        for rows in np.array_split(np.arange(d["streams"]), micro):
            terms, c_new, cache_new = TR.p1_losses(model, D.select(chunk, rows), M.carry_rows(carry, rows),
                                                   cache[rows], targets, cfg)
            (terms["g1"] * w_g1 / micro).backward()
            M.carry_put(carry, rows, c_new)
            cache[rows] = cache_new
            logs.append({k: float(v.detach()) for k, v in terms.items()})
        del terms, c_new, cache_new
        gn = float(torch.nn.utils.clip_grad_norm_(params, o["grad_clip"]))
        opt.step()
        c = coverage(chunk, bits, place if arm == "guided" else None, P)
        cov += c
        st["update"] += 1
        st["frames"] += int(chunk["valid"][:, P:].sum())
        st["target_frames"] += int(c[2]); st["covered"] += int(c[3])
        if st["update"] % 25 == 0 or st["update"] == 1 or smoke:
            rec = {"update": st["update"], "frames": st["frames"], "lr": lr, "grad_norm": gn,
                   "clip": min(1.0, o["grad_clip"] / max(gn, 1e-12)),
                   **{k: float(np.mean([l[k] for l in logs])) for k in logs[0]},
                   "guided_share": float(cov[1] / max(cov[0], 1)), "target_frames": int(cov[2]),
                   "covered_share": float(cov[3] / max(cov[2], 1)),
                   "sec_per_update": (time.time() - t0) / (st["update"] - u0),
                   "peak_gib": torch.cuda.max_memory_allocated() / 2 ** 30}
            print(json.dumps(rec), flush=True)
            if log:
                log.write(json.dumps(rec) + "\n"), log.flush()
            t0, u0, logs, cov = time.time(), st["update"], [], np.zeros(4, np.int64)
        if not smoke and st["update"] in ckpts:
            save(out / f"ft_{st['update']:04d}.pt", model, opt, streams, carry, cache, st, arm, ident)
            save(latest, model, opt, streams, carry, cache, st, arm, ident)
            mon = TR.monitor(model, cfg, split["monitor"]) if monitor else None
            rec = {"update": st["update"], "checkpoint": f"ft_{st['update']:04d}.pt",
                   "covered_share_cumulative": st["covered"] / max(st["target_frames"], 1), "monitor": mon}
            log.write(json.dumps(rec) + "\n"), log.flush()
            print("monitor", json.dumps(rec), flush=True)


def main():
    a = argparse.ArgumentParser()
    a.add_argument("arm", choices=["random", "guided"])
    a.add_argument("until", type=int)
    a.add_argument("--smoke", type=int)
    a.add_argument("--out", help="tests only: another output root")
    a.add_argument("--ckpts", help="tests only: comma-separated checkpoint updates")
    a.add_argument("--no-monitor", action="store_true", help="tests only")
    args = a.parse_args()
    ckpts = tuple(int(x) for x in args.ckpts.split(",")) if args.ckpts else CKPTS
    if args.until not in ckpts:
        raise SystemExit(f"until must be one of {ckpts}")
    run(args.arm, args.until, args.smoke, Path(args.out) if args.out else None, ckpts, not args.no_monitor)


if __name__ == "__main__":
    main()
