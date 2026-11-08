"""P1-F2 under its own gaze (PLAN.md): F2 checkpoint 10's saved online head chooses the windows along every real dev
episode exactly as in training; then the actor interface (F1.6 random, F2 random, F2 own conditional, F2 own closed-loop:
one process, paired) and S1 for F2 with its encoding bound to the own gaze. Conditional: the generated frame keeps the
window chosen on the RECORDED frame (its surprise saw the recorded outcome: Codex, 11-04); closed-loop (the amendment):
the head chooses it from the generated physical stage. Requires MAMBA_DETERMINISTIC=1 (the archived readings' mode).
Usage (project venv, GPU): MAMBA_DETERMINISTIC=1 python -m artifacts.experiments.20261104_f2_own_gaze.own_gaze [--smoke]
"""
import importlib, json, os, sys
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import numpy as np
import torch

from saccade import data as D, model as M, train as TR
from saccade.eval import s1 as S, gates as G

IF = importlib.import_module("artifacts.experiments.20261030_actor_interface.interface")
CH = importlib.import_module("artifacts.experiments.20261103_p1f2.chooser")
Q = CH.Q
HERE = Path(__file__).resolve().parent
EXP = TR.ROOT / "artifacts/experiments"
F16, F2 = EXP / "20261102_p1f16/run/ckpt_10.pt", EXP / "20261103_p1f2/run/ckpt_10.pt"
ARCHIVE = EXP / "20261103_p1f2/readings/interface_random.json"
DEV, GAZE_SEED = IF.DEV, 612401


def head_of(path):
    """The saved online head of a learned-gaze checkpoint, frozen."""
    fc = torch.load(path, weights_only=False, map_location="cpu")["fixation_controller"]
    assert fc["config"]["policy"] == "existing_combined_score_softmax" and fc["config"]["uniform_floor"] == .2
    head = Q.Head().to(DEV); head.load_state_dict(fc["head"]); head.eval()
    return head


@torch.no_grad()
def own_windows(model, head, L, cfg, seed=GAZE_SEED):
    """S.encode's window array ([states, 2]; frame t >= 1 at row offsets[e] + t - 1; -1 keeps the seeded prior) from the
    chooser's training path (chooser.Controller.prepare): coarse encoding without a crop, eight-frame history with
    episode-start padding, float32 actual / NOOP forecasts, C / S window maps, the existing policy, a seeded sample.
    Also returns, per row, the actual forecast, the C window map (both history-only) and the uniform draw: closed_loop's
    inputs."""
    offsets = np.r_[0, np.cumsum(L["ep_T"] + 1)]
    W, index = np.full((offsets[-1], 2), -1, np.int64), np.full(offsets[-1], -1, np.int64)
    taken_all, c_all, u_all = torch.zeros(offsets[-1], 81, 16), torch.zeros(offsets[-1], 49), torch.zeros(offsets[-1], 1, dtype=torch.float64)
    gen, win = torch.Generator(device=DEV).manual_seed(seed), Q.WIN.to(DEV)
    for e, ep in enumerate(S.episodes(L)):
        c = D.gather([ep], [[(0, 1, t, 1, 1) for t in range(ep.T + 1)]])
        frames = torch.as_tensor(c["frames"][0], device=DEV)
        parts = []
        for a in range(0, len(frames), 512):
            with TR.autocast(DEV, cfg):
                parts.append(CH.coarse(model.perception, frames[a:a + 512]))
        p, v = (torch.cat([x[i] for x in parts]) for i in (0, 1))
        pad = lambda a, fill: np.concatenate([np.full((1, 8, *a.shape[2:]), fill, a.dtype), a], 1)
        chunk = {"episode": pad(c["episode"], 0), "valid": pad(c["valid"], False), "prev_action": pad(c["prev_action"], -1),
                 "reward": pad(c["reward"], 0), "terminated": pad(c["terminated"], False)}
        zero = lambda a: torch.cat([a.new_zeros(8, *a.shape[1:]), a])[None]
        xx, y = CH.inputs(zero(p), zero(v), chunk, 8)
        ok = torch.as_tensor(chunk["valid"][:, 8:], device=DEV).flatten() & (xx[2] >= 0)
        for ix in torch.where(ok)[0].split(4096):
            x = tuple(z[ix] for z in xx)
            with torch.autocast(DEV.type, enabled=False):
                taken, noop = head(x), head((x[0], x[1], torch.zeros_like(x[2])))
                c_win = (taken - noop).square().sum(-1)[:, win].sum(-1)
                prob = Q.policy(c_win, (y[ix] - taken).square().sum(-1)[:, win].sum(-1))
            u = torch.rand(len(ix), 1, device=DEV, generator=gen, dtype=torch.float64)
            choice = torch.searchsorted(prob.double().cumsum(-1), u).squeeze(-1).clamp(max=48).cpu()
            t = ix.cpu().numpy()
            assert (t >= 1).all()
            W[offsets[e] + t - 1], index[offsets[e] + t - 1] = Q.CENTRES[choice].numpy(), choice.numpy()
            taken_all[offsets[e] + t - 1], c_all[offsets[e] + t - 1], u_all[offsets[e] + t - 1] = taken.cpu(), c_win.cpu(), u.cpu()
    return W, index, (taken_all, c_all, u_all)


def closed_loop(L, forecasts, index):
    """IF.futures' `choose`: the generated frame's window from the head's history-only forecasts (C) and the generated
    periphery (S: the same error, against the generated coarse frame instead of the recorded one), the training policy,
    sampled with the conditional run's own uniform draw (common random numbers: identical maps give identical windows,
    so closed - conditional is the policy change alone; the world's own draws are untouched, so the two pair exactly).
    `choose.same` collects, per future, whether the window equals the conditional one."""
    offsets = np.r_[0, np.cumsum(L["ep_T"] + 1)]
    taken, cmap, draw = forecasts
    win, centres = Q.WIN.to(DEV), Q.CENTRES.to(DEV)
    def choose(rows):
        i = np.array([offsets[e] + t - 1 for e, t in rows])
        y, c, u = taken[i].to(DEV), cmap[i].to(DEV), draw[i].to(DEV)
        def hook(p, v):
            with torch.autocast(DEV.type, enabled=False):
                prob = Q.policy(c, (p[:, 0].float() - y).square().sum(-1)[:, win].sum(-1))
            k = torch.searchsorted(prob.double().cumsum(-1), u).squeeze(-1).clamp(max=48)
            choose.same.append(k.cpu().numpy() == index[i])
            return centres[k][:, None]
        return hook
    choose.same = []
    return choose


def gaze_descriptives(L, enc_prior, W, index):
    """Share of frames refixated, window histogram, and crop coverage of any visible mob (labels, reading only)."""
    grids = G.episode_grids(L)
    offsets = np.r_[0, np.cumsum(L["ep_T"] + 1)]
    cov = {"own": [0, 0], "prior": [0, 0]}
    changed = []
    for e, r in enumerate(enc_prior):
        for t in range(1, len(r["fix"])):
            g = grids[e]["grid"][t].reshape(63) > 0
            if not g.any():
                continue
            for name, fx in (("own", W[offsets[e] + t - 1]), ("prior", r["fix"][t].numpy())):
                if fx[0] < 0:
                    fx = r["fix"][t].numpy()
                cells = M.window(torch.as_tensor(fx)).numpy()
                cov[name][0] += int(g[cells[cells < 63]].any()); cov[name][1] += 1
            changed.append(bool((W[offsets[e] + t - 1] != r["fix"][t].numpy()).any()))
    hist = np.bincount(index[index >= 0], minlength=49)
    return {"frames_with_own_window": int((index >= 0).sum()), "refixated_share_on_mob_frames": float(np.mean(changed)),
            "mob_crop_coverage": {k: v[0] / max(v[1], 1) for k, v in cov.items()}, "mob_frames": cov["own"][1],
            "window_histogram": hist.tolist(), "hud_touching_share": float(hist[(Q.WIN >= 63).any(-1).numpy()].sum() / hist.sum())}


def interface(L, cls, runs, out_path):
    """interface.main's loop for (label, path, gaze) triples in one process; paired differences as interface.main."""
    res = {"classes": {k: len(v) for k, v in cls.items()}, "policy": {l: g for l, _, g in runs}, "checkpoints": {}, "differences": {},
           "gaze": {}}
    per, head = {}, head_of(F2)
    for label, path, gaze in runs:
        model, cfg = IF.load(path)
        choose = None
        if gaze == "random":
            enc = S.encode(model, L, cfg)
        else:
            W, index, fc = own_windows(model, head, L, cfg)
            enc = S.encode(model, L, cfg, windows=W)
            if gaze == "own":
                res["gaze"][label] = gaze_descriptives(L, S.encode(model, L, cfg), W, index)
                print(label, "gaze", json.dumps({k: v for k, v in res["gaze"][label].items() if k != "window_histogram"}), flush=True)
            else:
                choose = closed_loop(L, fc, index)
            del fc
        res["checkpoints"][label] = {"path": str(path), "sha256": TR.file_sha256(path), "gaze": gaze, "classes": {}}
        for i, (name, tuples) in enumerate(cls.items()):
            if not tuples:
                continue
            if choose is not None:
                choose.same.clear()
            o = IF.futures(model, enc, L, tuples, 4, cfg, torch.Generator(device=DEV).manual_seed(3000 + i), choose)
            ep = np.array([tuples[j][0] for j in o["tuple"]])
            per[(label, name)] = (o, ep)
            res["checkpoints"][label]["classes"][name] = IF.summary(o, ep)
            if choose is not None:
                res["checkpoints"][label]["classes"][name]["window_as_conditional"] = float(np.concatenate(choose.same).mean())
        out_path.write_text(json.dumps(res, indent=1, default=str))
        print(label, "done", flush=True)
        del model, enc
        torch.cuda.empty_cache()
    for a, b in (("p1f16", "p1f2_closed"), ("p1f2", "p1f2_closed"), ("p1f2_own", "p1f2_closed"), ("p1f16", "p1f2")):
        res["differences"][f"{b}-{a}"] = {}
        for name in cls:
            if (a, name) not in per:
                continue
            (oa, ep), (ob, _) = per[(a, name)], per[(b, name)]
            res["differences"][f"{b}-{a}"][name] = {}
            for q in ("unlock_p", "exp_reward", "neg_p", "term_p"):
                est = G.estimate(ob[f"gen_{q}"] - oa[f"gen_{q}"], np.ones(len(ep), bool), ep)
                res["differences"][f"{b}-{a}"][name][q] = {**est, "ci95": [est["est"] - 1.96 * est["se"], est["est"] + 1.96 * est["se"]]}
    res["closed_pairs_conditional"] = max(float(np.abs(per[("p1f2_own", n)][0][k].astype(float) - per[("p1f2_closed", n)][0][k]).max())
                                          for n in cls if ("p1f2_own", n) in per for k in per[("p1f2_own", n)][0] if k.startswith("rec_"))
    out_path.write_text(json.dumps(res, indent=1, default=str))
    return res


def reproduction(res):
    """Every numeric field of the random-gaze worlds against the archived F2 queue output (same labels)."""
    old = json.loads(ARCHIVE.read_text())["checkpoints"]
    diffs = []
    def walk(a, b):
        if isinstance(a, dict):
            for k in a:
                walk(a[k], b.get(k) if isinstance(b, dict) else None)
        elif isinstance(a, (int, float)) and not isinstance(a, bool) and isinstance(b, (int, float)):
            diffs.append(abs(a - b))
    for label in ("p1f16", "p1f2"):
        walk(res["checkpoints"][label]["classes"], old[label]["classes"])
    return {"fields": len(diffs), "max_abs_difference": max(diffs) if diffs else None}


def main():
    assert os.environ.get("MAMBA_DETERMINISTIC") == "1"
    smoke = "--smoke" in sys.argv
    out = HERE / ("smoke" if smoke else "readings"); out.mkdir(parents=True, exist_ok=True)
    L = S.first_episodes(dict(np.load(TR.ROOT / "artifacts/eda/s0/s0_dev_labels.npz")), 20 if smoke else 0)
    cls = IF.classes(L, np.random.default_rng(2031))
    if smoke:
        cls = {k: v[:8] for k, v in cls.items()}
    res = interface(L, cls, [("p1f16", F16, "random"), ("p1f2", F2, "random"), ("p1f2_own", F2, "own"),
                             ("p1f2_closed", F2, "closed")], out / "interface.json")
    if not smoke:
        res["reproduction_of_archived_random"] = reproduction(res)
        print("reproduction", res["reproduction_of_archived_random"], flush=True)
        (out / "interface.json").write_text(json.dumps(res, indent=1, default=str))
    head, encode = head_of(F2), S.encode       # S1 with every encoding bound to the own gaze (S1.3's generated frame:
                                               # conditional window; its C2ST / generated outcomes are not closed-loop)
    S.encode = lambda model_, L_, cfg_, untrained=None: encode(model_, L_, cfg_, untrained, windows=own_windows(model_, head, L_, cfg_)[0])
    sys.argv = ["s1", str(F2), str(out / "s1_p1f2_own.json")] + (["20", "20"] if smoke else [])
    S.main()
    S.encode = encode
    print("saved", out, flush=True)


if __name__ == "__main__":
    main()
