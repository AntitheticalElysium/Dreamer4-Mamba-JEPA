"""T1 evaluation (T1_PROTOCOL.md measurements). Usage: t1_eval.py --data DIR --run RUN_DIR
DEV windows of 65 frames (stride 64, episodes >= 65 frames); teacher-forced predictions of frames 1..64.
Events use simulator entity identity and world positions (evaluation only), relative to the window."""
import argparse, json, os, sys
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from t1_models import make
from t1_train import load, windows, batch_at, T

MOB_CODE = {"zombies": 1, "cows": 2, "skeletons": 3}


@torch.no_grad()
def predict(model, d, starts, estar, bs=8):
    P = {"block": [], "mob": [], "vit": [], "inv": []}
    for i in range(0, len(starts), bs):
        inp, tgt, sh = batch_at(d, starts[i:i + bs], "cuda")
        with torch.autocast("cuda", dtype=torch.bfloat16):
            lb, lm, lv, li = model(inp, true_shift=sh if estar else None)
        P["block"].append(lb.float().softmax(-1).cpu().numpy().astype(np.float16))
        P["mob"].append(lm.float().softmax(-1).cpu().numpy().astype(np.float32))
        P["vit"].append(lv.float().softmax(-1).cpu().numpy()); P["inv"].append(li.float().softmax(-1).cpu().numpy())
    return {k: np.concatenate(v) for k, v in P.items()}                     # [N, T, ...] for frames 1..64


def events(d, s):
    """For one window starting at frame s: lists of (t_target, cell, code, kind, gap) for mobs, and terrain entries."""
    pos = d["pos"][s:s + T + 1].astype(int)
    mob_ev = []
    for m, code in MOB_CODE.items():
        P = d[f"{m}_p"][s:s + T + 1].astype(int); M = d[f"{m}_m"][s:s + T + 1]
        rel = P - pos[:, None, :]
        vis = M & (np.abs(rel[..., 0]) <= 3) & (np.abs(rel[..., 1]) <= 4)
        for j in range(M.shape[1]):
            for t in range(1, T + 1):
                if not vis[t, j]:
                    continue
                cell = (rel[t, j, 0] + 3) * 9 + (rel[t, j, 1] + 4)
                if vis[t - 1, j]:
                    mob_ev.append((t, cell, code, "persist", 0)); continue
                seen = np.nonzero(vis[:t - 1, j])[0]
                if len(seen) and M[seen[-1]:t + 1, j].all():
                    mob_ev.append((t, cell, code, "reentry", t - seen[-1]))
                else:
                    mob_ev.append((t, cell, code, "first", 0))
    ter_ev = []
    seen_world = set()
    for t in range(T + 1):
        cells = [(r, c, (pos[t, 0] + r - 3, pos[t, 1] + c - 4)) for r in range(7) for c in range(9)]
        if t >= 1:
            prev_view = {(pos[t - 1, 0] + r - 3, pos[t - 1, 1] + c - 4) for r in range(7) for c in range(9)}
            for r, c, w in cells:
                if w not in prev_view:
                    ter_ev.append((t, r * 9 + c, "revisit" if w in seen_world else "new"))
            seen_world |= prev_view
    return mob_ev, ter_ev


def main():
    p = argparse.ArgumentParser(); p.add_argument("--data", required=True); p.add_argument("--run", required=True)
    a = p.parse_args()
    st = torch.load(os.path.join(a.run, "final.pt"), weights_only=False)
    arch = st["args"]["arch"]; model = make(arch).cuda().eval(); model.load_state_dict(st["model"])
    d = load(os.path.join(a.data, "dev_sym.npz")); starts = windows(d, stride=T)
    P = predict(model, d, starts, arch == "Estar")
    mob = {k: [] for k in ("persist", "first")}; reb = {"1-2": [], "3-5": [], "6-15": [], "16+": []}
    ter = {"revisit": [], "new": []}; vit_change = []
    nll = {"block": [], "mob": [], "vit": [], "inv": []}
    for w, s in enumerate(starts):
        idx = s + np.arange(1, T + 1)
        vb = d["view"][idx].reshape(T, 63).astype(int); vm = d["mob"][idx].reshape(T, 63).astype(int)
        vv = d["vit"][idx].astype(int).clip(0, 9); vi = d["inv"][idx].astype(int).clip(0, 9)
        pb, pm, pv, pi = P["block"][w], P["mob"][w], P["vit"][w], P["inv"][w]
        tt = np.arange(T)[:, None]; cc = np.arange(63)[None]
        nll["block"].append(-np.log(pb[tt, cc, vb].astype(np.float64) + 1e-9).mean())
        nll["mob"].append(-np.log(pm[tt, cc, vm] + 1e-9).mean())
        nll["vit"].append(-np.log(pv[tt, np.arange(4)[None], vv] + 1e-9).mean())
        nll["inv"].append(-np.log(pi[tt, np.arange(12)[None], vi] + 1e-9).mean())
        prev_v = d["vit"][s + np.arange(0, T)].astype(int)
        for t_, k in zip(*np.nonzero(vv != prev_v.clip(0, 9))):
            vit_change.append(-np.log(pv[t_, k, vv[t_, k]] + 1e-9))
        mev, tev = events(d, s)
        for t, cell, code, kind, gap in mev:
            pr = float(pm[t - 1, cell, code])
            if kind == "reentry":
                reb["1-2" if gap <= 2 else "3-5" if gap <= 5 else "6-15" if gap <= 15 else "16+"].append(pr)
            else:
                mob[kind].append(pr)
        for t, cell, kind in tev:
            ter[kind].append(float(pb[t - 1, cell].argmax() == vb[t - 1, cell]))
    allre = sum(reb.values(), [])
    res = {"arch": arch, "seed": st["args"]["seed"], "params": st["params"], "windows": int(len(starts)),
           "mob_prob": {"reentry": float(np.mean(allre)), "first": float(np.mean(mob["first"])),
                        "persist": float(np.mean(mob["persist"])),
                        "reentry_by_gap": {k: (float(np.mean(v)) if v else None) for k, v in reb.items()},
                        "n": {"reentry": len(allre), "first": len(mob["first"]), "persist": len(mob["persist"])}},
           "terrain_entry_acc": {k: float(np.mean(v)) for k, v in ter.items()},
           "terrain_n": {k: len(v) for k, v in ter.items()},
           "vit_change_nll": float(np.mean(vit_change)), "nll": {k: float(np.mean(v)) for k, v in nll.items()}}
    json.dump(res, open(os.path.join(a.run, "eval.json"), "w"), indent=1)
    print(json.dumps(res))


if __name__ == "__main__":
    main()
