"""v3 fine-tune (PLAN.md): P1 world seed 1 from checkpoint 10 with P1's own update, every module training: the full
P1 objective (train.p1_losses through the harmoniser, as train.p1), clipped at 1.0 over ALL parameters (P1's clip;
v2 clipped the denoiser alone, which fed Adam gradients ~20x its second-moment estimate), P1's chunks (16 streams x
256 steps, 4 microbatches). Learning rate: linear from the checkpoint's 1e-5 to v2's LR over v2's WARMUP updates,
then constant, each group keeping its lr_scale. Arm `guided`: every attempt frame (placement_map's rule, no coin:
Q = 1) takes the rule's window; arm `random`: the prior. The prior is drawn for every frame in both arms (v2's
GuidedStreams), so episode order, noise and all other fixations match. Checkpoints at v2's CKPTS with exact resume
state, each followed by P1's monitor. Logged every 25 updates: the update's norm per top-level module (Codex: whether
P1's clipping leaves a transient is a hypothesis; P1's last interval moved the denoiser 0.57 over 625 updates).
Usage (project venv, GPU; long runs via systemd-run): python -m artifacts.experiments.20261030_m05_guided_fovea.finetune
  ARM UNTIL_UPDATE [--smoke N]   (resumes from finetune/ARM/latest.pt; --smoke: N updates, nothing saved)
"""
import argparse, importlib, json, time
from pathlib import Path

import numpy as np
import torch

from saccade import data as D, model as M, train as TR

FT2 = importlib.import_module("artifacts.experiments.20261030_fovea_placement.finetune")
PM = importlib.import_module("artifacts.experiments.20261030_fovea_placement.placement_map")
ROOT, HERE, BASE = FT2.ROOT, Path(__file__).resolve().parent, FT2.BASE
PLACE, BITS = HERE / "place_q1_train.npy", FT2.HERE / "targets_train.npy"


def place_all(split):
    """[train frames, 2] int8: the rule's window on every attempt frame (frame t decided at state t - 1), else -1."""
    L = dict(np.load(PM.LABELS))
    rule = {int(k): tuple(v) for k, v in json.loads((PM.HERE / "map.json").read_text())["rule"].items()}
    s_off, _ = PM.offsets(L)
    where = {(int(s), int(o)): e for e, (s, o) in enumerate(zip(L["ep_shard"], L["ep_slot"]))}
    order = [where[(ep.shard, ep.slot)] for ep in split["train"]]
    off = D.frame_offsets(split["train"])
    win = PM.window_of_state(L, rule)
    place = np.full((int(off[-1]), 2), -1, np.int8)
    for i, e in enumerate(order):
        assert s_off[e + 1] - s_off[e] == off[i + 1] - off[i]
        place[off[i] + 1:off[i + 1]] = win[s_off[e]:s_off[e + 1] - 1]
    return place


def run(arm, until, smoke=None):
    dev = torch.device("cuda")
    ck = torch.load(BASE, weights_only=False, map_location=dev)
    cfg, d, P = ck["cfg"], ck["cfg"]["data"], ck["cfg"]["data"]["context_prefix"]
    out = HERE / "finetune" / arm
    out.mkdir(parents=True, exist_ok=True)
    split = D.load_split(cfg)
    offsets = D.frame_offsets(split["train"])
    targets = np.load(ROOT / cfg["p1"]["out_dir"] / "targets/change_targets.npy", mmap_mode="r")
    if not PLACE.exists():
        np.save(PLACE, place_all(split))
    place, bits = np.load(PLACE), np.load(BITS)
    assert len(place) == len(bits) == int(offsets[-1]) == len(targets)
    prov = TR.run_provenance(cfg)
    ident = {"base_sha256": TR.file_sha256(BASE), "place_sha256": TR.file_sha256(PLACE), "targets_bits_sha256": TR.file_sha256(BITS),
             "script_sha256": TR.file_sha256(Path(__file__)), "lr": FT2.LR, "warmup": FT2.WARMUP, "steps": FT2.STEPS,
             **{k: v for k, v in prov.items() if k != "git_commit"}}
    model = M.M0(cfg).to(dev)
    model.load_state_dict(ck["model"])
    model.denoiser.checkpoint = cfg["p1"]["activation_checkpointing"]
    o = cfg["optimizer"]
    opt = torch.optim.AdamW(model.param_groups(), lr=o["lr"], betas=tuple(o["betas"]), eps=o["eps"])
    opt.load_state_dict(ck["opt"])
    streams = FT2.GuidedStreams(split["train"], d["streams"], P, seed=ck["world_seed"], targets_offsets=offsets,
                                place=place if arm == "guided" else None)
    streams.load_state(ck["streams"])
    carry, cache = ck["carry"], ck["cache"]
    torch.set_rng_state(ck["rng"]["cpu"].cpu()), torch.cuda.set_rng_state(ck["rng"]["cuda"].cpu())
    lr0 = o["lr"] * TR.lr_factor(ck["state"]["frames"], cfg)
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
    micro = d["accumulate_at_256"]
    log = None if smoke else open(out / "log.jsonl", "a")
    if log:
        log.write(json.dumps({"start_update": st["update"], "until": until, "git_commit": prov["git_commit"],
                              "identity": ident, "lr0": lr0}) + "\n"), log.flush()
    stop = st["update"] + smoke if smoke else until
    t0, u0, logs, cov = time.time(), st["update"], [], np.zeros(4, np.int64)
    while st["update"] < stop:
        chunk = streams.next(FT2.STEPS)
        lr = FT2.lr_at(st["update"], lr0)
        for g in opt.param_groups:
            g["lr"] = lr * g["lr_scale"]
        opt.zero_grad(set_to_none=True)
        for rows in np.array_split(np.arange(d["streams"]), micro):
            terms, c_new, cache_new = TR.p1_losses(model, D.select(chunk, rows), M.carry_rows(carry, rows),
                                                   cache[rows], targets, cfg)
            (model.harmoniser(TR.groups(terms)) / micro).backward()
            M.carry_put(carry, rows, c_new)
            cache[rows] = cache_new
            logs.append({k: float(v.detach()) for k, v in terms.items()})
        del terms, c_new, cache_new
        gn = float(torch.nn.utils.clip_grad_norm_(model.parameters(), o["grad_clip"]))
        log_now = (st["update"] + 1) % 25 == 0 or st["update"] == 0 or smoke
        snap = {n: p.detach().clone() for n, p in model.named_parameters()} if log_now else None
        opt.step()
        if snap is not None:                                             # the update's size per module (transient check)
            dsq = {}
            for n, p in model.named_parameters():
                dsq[n.split(".")[0]] = dsq.get(n.split(".")[0], 0.0) + float((p.detach() - snap[n]).float().square().sum())
            update_norm = {k: v ** 0.5 for k, v in dsq.items()}
            del snap
        c = FT2.coverage(chunk, bits, place if arm == "guided" else None, P)
        cov += c
        st["update"] += 1
        st["frames"] += int(chunk["valid"][:, P:].sum())
        st["target_frames"] += int(c[2]); st["covered"] += int(c[3])
        if st["update"] % 25 == 0 or st["update"] == 1 or smoke:
            rec = {"update": st["update"], "frames": st["frames"], "lr": lr, "grad_norm": gn,
                   "clip": min(1.0, o["grad_clip"] / max(gn, 1e-12)), "update_norm": update_norm,
                   "sigma": model.harmoniser.log_sigma.exp().tolist(),
                   **{k: float(np.mean([l[k] for l in logs])) for k in logs[0]},
                   "guided_share": float(cov[1] / max(cov[0], 1)), "target_frames": int(cov[2]),
                   "covered_share": float(cov[3] / max(cov[2], 1)),
                   "sec_per_update": (time.time() - t0) / (st["update"] - u0),
                   "peak_gib": torch.cuda.max_memory_allocated() / 2 ** 30}
            print(json.dumps(rec), flush=True)
            if log:
                log.write(json.dumps(rec) + "\n"), log.flush()
            t0, u0, logs, cov = time.time(), st["update"], [], np.zeros(4, np.int64)
        if not smoke and st["update"] in FT2.CKPTS:
            FT2.save(out / f"ft_{st['update']:04d}.pt", model, opt, streams, carry, cache, st, arm, ident)
            FT2.save(latest, model, opt, streams, carry, cache, st, arm, ident)
            mon = TR.monitor(model, cfg, split["monitor"])
            rec = {"update": st["update"], "checkpoint": f"ft_{st['update']:04d}.pt",
                   "covered_share_cumulative": st["covered"] / max(st["target_frames"], 1), "monitor": mon}
            log.write(json.dumps(rec) + "\n"), log.flush()
            print("monitor", json.dumps(rec), flush=True)


def main():
    a = argparse.ArgumentParser()
    a.add_argument("arm", choices=["random", "guided"])
    a.add_argument("until", type=int, choices=FT2.CKPTS)
    a.add_argument("--smoke", type=int)
    args = a.parse_args()
    run(args.arm, args.until, args.smoke)


if __name__ == "__main__":
    main()
