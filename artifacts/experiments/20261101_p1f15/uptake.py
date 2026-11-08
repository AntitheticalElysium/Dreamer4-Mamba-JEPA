"""P1-F1.5 reading 5 (PLAN.md): does a useful past glimpse reach the next physical prediction through the belief?
F2 of the fovea probes (`../20261029_fovea_probes/measure.py`, its episode function unchanged; CPU; P1 world 1's F2
episodes, frames and seeds): at sampled frames t, the 49 counterfactual real glimpses of frame t enter the belief (and
the context, whose physical tokens never read them in a factorised world), then the G1 loss of the real frame t + 1
on its map rows, HUD rows and lighting at the 8 monitor levels, common noise across windows, two noise sets. Strata per
(frame, window): the window held a mob (labels), held a mob the periphery missed (S1's frozen reader on frame t's real
periphery and lighting reads another kind at that cell); light (F2's night set: night with a mob in view, beside its
random frames). Readouts per frame, per element, averaged over levels and sets, paired within the frame: windows that
held a missed mob minus windows that held none (negative = that evidence improves the next prediction through the
belief); windows that held any mob minus none; the recorded window minus the other 48; the spread of the window
effect across the 49 against the two noise sets' disagreement about it.
Usage (project venv, CPU): python -m artifacts.experiments.20261101_p1f15.uptake CKPT OUT_DIR [N_EPISODES]
"""
import importlib, json, os, sys
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")
import numpy as np
import torch

from saccade import data as D, model as M, train as TR
from saccade.eval import s1 as S, gates as G

FP = importlib.import_module("artifacts.experiments.20261029_fovea_probes.measure")
N_EPISODES = 250                                                         # F2's (fovea probes RESULTS)


def kinds(reader, p, v):
    """Codex's sampler_followup.kind_grid: the frozen S1 reader's kind per kept map cell."""
    x = torch.cat([p[:, :63].float().reshape(-1, p.shape[-1]), v.float().repeat_interleave(63, 0)], -1)
    return reader(x)["kind"].argmax(-1).reshape(len(p), 63)[:, S.KEEP[:63]].cpu().numpy()


def summary(z, cfg):
    """The readouts above, pooled per light set with episode-bootstrap SEs."""
    pc = cfg["perception"]
    size = np.array([63 * pc["periphery"]["token_dim"], 18 * pc["periphery"]["token_dim"], pc["lighting"]["dim"]], float)
    l = z["loss"].mean((1, 2)) / size                                    # [n, 49, 3] per element
    eff = z["loss"].mean(2) / size                                       # [n, 2 sets, 49, 3]
    eff = eff - eff.mean(2, keepdims=True)                               # window effect within each set
    n = len(l)
    def paired(a, b):                                                    # mean over windows a minus over windows b, per frame
        ok = a.any(1) & b.any(1)
        d = np.stack([(l[i][a[i]].mean(0) - l[i][b[i]].mean(0)) if ok[i] else np.full(3, np.nan) for i in range(n)])
        return d, ok
    none = ~z["held_mob"]
    d_miss, ok_miss = paired(z["held_missed"], none)
    d_mob, ok_mob = paired(z["held_mob"], none)
    rec = np.zeros((n, 49), bool); rec[np.arange(n), z["recorded"]] = True
    d_rec, ok_rec = paired(rec, ~rec)
    spread, noise = eff.mean(1).std(1), ((eff[:, 0] - eff[:, 1]) / 2).std(1)
    out = {}
    for sname, m in (("random", ~z["night_set"]), ("night", z["night_set"])):
        r = out[sname] = {"frames": int(m.sum())}
        for gi, g in enumerate(("map_rows", "hud_rows", "lighting")):
            est = lambda d, ok: G.estimate(np.nan_to_num(d[:, gi]), m & ok, z["e"])
            r[g] = {"missed_mob_minus_none": est(d_miss, ok_miss), "mob_minus_none": est(d_mob, ok_mob),
                    "recorded_minus_others": est(d_rec, ok_rec), "loss_mean": G.estimate(l[:, :, gi].mean(1), m, z["e"]),
                    "window_effect_spread": G.estimate(spread[:, gi], m, z["e"]), "noise_sets_disagreement": G.estimate(noise[:, gi], m, z["e"])}
    return out


@torch.no_grad()
def main():
    ckpt, out_dir = Path(sys.argv[1]), Path(sys.argv[2])
    n_ep = int(sys.argv[3]) if len(sys.argv) > 3 else N_EPISODES
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "4")))
    ck = torch.load(ckpt, weights_only=False, map_location="cpu"); cfg = ck["cfg"]
    model = M.M0(cfg); model.load_state_dict(ck["model"]); model.eval(); del ck
    reader_path = ckpt.parent / "s1/results.reader.pt"
    reader = S.load_reader(reader_path, ckpt, "cpu")
    L = S.first_episodes(dict(np.load(FP.ROOT / "artifacts/eda/s0/s0_dev_labels.npz")), n_ep)
    s_off, sig = np.r_[0, np.cumsum(L["ep_T"] + 1)], FP.levels(cfg)
    rng, gen = np.random.default_rng(2029), torch.Generator().manual_seed(2029)  # F2's
    grids, keep = G.episode_grids(L), np.asarray(S.KEEP[:63])
    win = np.zeros((49, 63), bool)                                       # map cells (row-major, rows 0-6) per window
    for w, cells in enumerate(M.window(FP.CENTRES).numpy()):
        win[w, cells[cells < 63]] = True
    win = win[:, keep]
    cen = {tuple(c): i for i, c in enumerate(FP.CENTRES.tolist())}
    recs = {k: [] for k in ("e", "t", "night_set", "light", "recorded", "loss", "held_mob", "held_missed")}
    for e, ep in enumerate(S.episodes(L)):
        f1, f2 = FP.episode(model, cfg, L, e, ep, s_off, sig, rng, gen)
        if f2:
            obs = D.RP.episode_fields(ep.shard, ep.slot)["observations"]
            ts = [s["t"] for s in f2]
            per = model.perception(torch.stack([torch.as_tensor(obs[t]) for t in ts]), FP.CENTRES[:1].expand(len(ts), 2))
            kp = kinds(reader, per.p, per.v)                             # periphery and lighting do not depend on the fixation
            for i, s in enumerate(f2):
                truth = grids[e]["grid"][s["t"]].reshape(63)[keep]
                mob, missed = truth > 0, (truth > 0) & (kp[i] != truth)
                for k, x in (("e", e), ("t", s["t"]), ("night_set", s["night_set"]), ("light", float(L["s_light"][s_off[e] + s["t"]])),
                             ("recorded", cen[tuple(int(c) for c in f1["fix"][s["t"]])]), ("loss", s["loss"]),
                             ("held_mob", (win & mob).any(1)), ("held_missed", (win & missed).any(1))):
                    recs[k].append(x)
        if e % 10 == 9 or e == len(L["ep_T"]) - 1:
            print(f"episode {e + 1}/{len(L['ep_T'])}, frames {len(recs['e'])}", flush=True)
    z = {k: np.array(v) for k, v in recs.items()}
    np.savez_compressed(out_dir / "uptake.npz", **z, sigmas=np.array(sig))
    res = {"checkpoint": str(ckpt), "sha256": TR.file_sha256(ckpt), "reader_sha256": TR.file_sha256(reader_path),
           "script_sha256": TR.file_sha256(Path(__file__)), "measure_sha256": TR.file_sha256(Path(FP.__file__)),
           "episodes": n_ep, "summary": summary(z, cfg)}
    (out_dir / "uptake.json").write_text(json.dumps(res, indent=1, default=float))
    print(json.dumps(res["summary"], default=float)[:3000], flush=True)


if __name__ == "__main__":
    main()
