"""Descriptive analysis (CPU, existing data only; user 11-06: "let's see that analysis"): does one quantity, how much
training NEEDED a relation's determinant, order how well P1-F3 generates the relation, and does it need a second
axis, where the determinant lives relative to the consequence token?
Per relation: frequency per pass over the train split (labels); change size D = the token-space distance between the
two outcomes at the consequence token(s) (F3's own tokens, CPU perception of train400 episodes); the share of
training noise draws at which the target does NOT already reveal the outcome, E_sigma[1 - r_ideal(sigma; D)] under
the checkpoint's lognormal(-0.4, 1.2) (r_ideal: T1's equal-prior reader); need = frequency x D^2 x that share (the
squared-error reduction available from the determinant when the model otherwise predicts the default; derivation in
RESULTS.md 4). Success: frame readers where measured (determinant curves / T2 / T1), else the actor interface's head
on generated futures (marked: the head also reads the belief, so it may credit an event the frame did not draw).
Usage: python -m artifacts.experiments.20261106_decision_levels.need   -> readings/need.json
"""
import importlib, json, os
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import numpy as np
import torch

from saccade import data as D, model as M, train as TR

CV = importlib.import_module("artifacts.experiments.20261106_determinant_curves.curves")
FZ_IDEAL = importlib.import_module("artifacts.experiments.20261106_decision_levels.frozen").ideal_reliance
HERE = Path(__file__).resolve().parent
HUD = {"health": 63, "food": 64, "drink": 65, "energy": 66, "sapling": 67, "wood": 68, "stone": 69, "coal": 70, "iron": 71,
       "diamond": 72, "wood_pickaxe": 73, "stone_pickaxe": 74, "iron_pickaxe": 75, "wood_sword": 76, "stone_sword": 77,
       "iron_sword": 78}                                                # renderer: (x, y) -> 63 + 9 y + x
INV = {"wood": 0, "stone": 1, "coal": 2, "iron": 3, "diamond": 4, "sapling": 5, "wood_pickaxe": 6, "stone_pickaxe": 7,
       "iron_pickaxe": 8, "wood_sword": 9, "stone_sword": 10, "iron_sword": 11}
BLOCK = {"grass": 2, "water": 3, "stone": 4, "tree": 5, "path": 7, "coal": 8, "iron": 9, "diamond": 10, "table": 11,
         "furnace": 12, "plant": 15, "ripe_plant": 16, "sand": 13}
N_EP = 120


def token_means():
    """F3's periphery tokens on train400 episodes (CPU): per HUD cell and value, per map block type (cells with no mob,
    not the player's) and per mob kind on a map cell."""
    ck = torch.load(TR.ROOT / CV.CKPT, map_location="cpu", weights_only=False)
    model = M.M0(ck["cfg"]); model.load_state_dict(ck["model"]); model.eval()
    L = dict(np.load(CV.LABELS / "s0_train400_labels.npz"))
    s_off = np.r_[0, np.cumsum(L["ep_T"] + 1)]
    hud = {k: {} for k in HUD}
    blocks, mobs = {}, {}
    acc = lambda d, k, x: d.setdefault(k, [np.zeros(16), 0]).__setitem__(slice(None), [d[k][0] + x.sum(0), d[k][1] + len(x)])
    for e in range(N_EP):
        obs = D.RP.episode_fields(int(L["ep_shard"][e]), int(L["ep_slot"][e]))["observations"]
        with torch.no_grad():
            p = torch.cat([model.perception(obs[a:a + 512], torch.full((len(obs[a:a + 512]), 2), 4)).p for a in range(0, len(obs), 512)]).numpy()
        s = slice(s_off[e], s_off[e + 1])
        body, inv, view, pos = L["s_body"][s], L["s_inv"][s], L["s_view"][s].reshape(-1, 63), L["s_pos"][s].astype(int)
        values = {"health": body[:, 0].clip(0, 9), "food": body[:, 1], "drink": body[:, 2], "energy": body[:, 3],
                  **{k: inv[:, i] for k, i in INV.items()}}
        for k, c in HUD.items():
            for v in np.unique(values[k]):
                acc(hud[k], int(v), p[values[k] == v, c])
        mobcell = np.zeros((len(p), 63), int)                          # 0 none, else kind 1..4 (zombie, cow, skeleton, arrow)
        for kind, name in enumerate(("zombies", "cows", "skeletons", "arrows"), 1):
            mp, mm = L[f"s_{name}_p"][s].astype(int), L[f"s_{name}_m"][s]
            rel = mp - pos[:, None]
            ok = mm & (np.abs(rel[..., 0]) <= 3) & (np.abs(rel[..., 1]) <= 4)
            t, i = np.nonzero(ok)
            mobcell[t, (3 + rel[t, i, 0]) * 9 + 4 + rel[t, i, 1]] = kind
        keep = np.ones(63, bool); keep[31] = False
        for name, b in BLOCK.items():
            m = (view == b) & (mobcell == 0) & keep
            if m.any():
                acc(blocks, name, p[:, :63][m])
        for kind, name in ((1, "zombie"), (2, "cow")):
            m = (mobcell == kind) & keep
            if m.any():
                acc(mobs, name, p[:, :63][m])
        print("encoded", e, flush=True)
    mean = lambda d: {k: (v[0] / v[1]).tolist() for k, v in d.items() if v[1] >= 30}
    return {"hud": {k: mean(v) for k, v in hud.items()}, "blocks": mean(blocks), "mobs": mean(mobs)}


def frequencies():
    """Per pass over the train split: how often each consequence occurs (labels, transitions state g -> g + 1)."""
    L = dict(np.load(CV.LABELS / "s0_train_labels.npz"))
    s_off, a_off = np.r_[0, np.cumsum(L["ep_T"] + 1)], np.r_[0, np.cumsum(L["ep_T"])]
    ep = np.repeat(np.arange(len(L["ep_T"])), L["ep_T"]); g = s_off[ep] + np.arange(len(ep)) - a_off[ep]
    a, inv0, inv1 = L["t_action"].astype(int), L["s_inv"][g], L["s_inv"][g + 1]
    b0, b1 = L["s_body"][g], L["s_body"][g + 1]
    d = CV.IF.DIRS[L["s_dir"][g]]
    face0 = L["s_view"][g, 3 + d[:, 0], 4 + d[:, 1]]
    face1 = L["s_view"][g + 1, 3 + d[:, 0], 4 + d[:, 1]]
    awake = ~L["s_sleep"][g]
    f = {"move": int(((L["s_pos"][g + 1] != L["s_pos"][g]).any(1)).sum()),
         "stay_asleep": int((L["s_sleep"][g] & L["s_sleep"][g + 1]).sum()),
         "wake_energy9": int((L["s_sleep"][g] & ~L["s_sleep"][g + 1] & (b0[:, 3] >= 9)).sum()),
         "damage": int((b1[:, 0] < b0[:, 0]).sum()), "death": int(L["t_terminated"].sum()),
         "drink_up": int((b1[:, 2] > b0[:, 2]).sum())}
    T = CV.transitions(L)                                               # DO on a faced cow that is gone afterwards
    f["cow_killed"] = int(((a == CV.DO) & awake & T["on"]["cows"].any(1) & ~T["true_on"]["cows"] & (b1[:, 1] > b0[:, 1])).sum())
    for k in ("wood", "stone", "coal", "iron", "diamond", "sapling"):
        f[f"{k}_up"] = int((inv1[:, INV[k]] == inv0[:, INV[k]] + 1).sum())
        f[f"{k}_appear"] = int(((inv1[:, INV[k]] == 1) & (inv0[:, INV[k]] == 0)).sum())
        f[f"{k}_increment"] = f[f"{k}_up"] - f[f"{k}_appear"]
    for k in ("wood_pickaxe", "stone_pickaxe", "iron_pickaxe", "wood_sword", "stone_sword", "iron_sword"):
        f[f"make_{k}"] = int((inv1[:, INV[k]] > inv0[:, INV[k]]).sum())
    for k, blk in (("table", 11), ("furnace", 12), ("plant", 15)):
        f[f"place_{k}"] = int(((face1 == blk) & (face0 != blk) & awake).sum())
    f["place_stone"] = int(((face1 == 4) & (face0 != 4) & (inv1[:, 1] == inv0[:, 1] - 1)).sum())
    f["cell_cleared_coal"] = int(((face0 == 8) & (face1 == 7)).sum())
    f["cell_cleared_tree"] = int(((face0 == 5) & (face1 == 2)).sum())
    f["cell_cleared_stone"] = int(((face0 == 4) & (face1 == 7)).sum())
    f["cell_cleared_iron"] = int(((face0 == 9) & (face1 == 7)).sum())
    f["transitions"] = int(len(g))
    return f


def leak_share(Dist):
    """E_sigma[1 - r_ideal(sigma; D)] under the training lognormal(-0.4, 1.2) clipped to [0.002, 20] (quadrature)."""
    z = np.linspace(-6, 6, 241); w = np.exp(-z ** 2 / 2); w /= w.sum()
    sig = np.clip(np.exp(-0.4 + 1.2 * z), 0.002, 20.0)
    return float((w * (1 - FZ_IDEAL(Dist, sig))).sum())


def main():
    means = token_means() if not (HERE / "readings/token_means.json").exists() else json.loads((HERE / "readings/token_means.json").read_text())
    (HERE / "readings/token_means.json").write_text(json.dumps(means))
    freq = frequencies()
    H, Bk, Mb = ({k: {int(a): np.array(b) for a, b in v.items()} for k, v in means["hud"].items()},
                 {k: np.array(v) for k, v in means["blocks"].items()}, {k: np.array(v) for k, v in means["mobs"].items()})
    dist = lambda a, b: float(np.linalg.norm(a - b))
    step = lambda k, n, m: dist(H[k][n], H[k][m]) if n in H[k] and m in H[k] else None
    geo = json.loads((CV.HERE / "readings/geometry.json").read_text())
    t1 = json.loads((HERE / "readings/t1.json").read_text())
    passes = 25.6e6 / freq["transitions"]
    rows = []

    def add(name, kind, n_per_pass, Dist, success, instrument, note=""):
        if not Dist:
            rows.append({"relation": name, "determinant_lives": kind, "per_pass": n_per_pass, "D": None, "need": None, "success": success,
                         "instrument": instrument, "note": "no token mean for an outcome in the encoded episodes"})
            return
        share = leak_share(Dist)
        rows.append({"relation": name, "determinant_lives": kind, "per_pass": n_per_pass, "D": round(Dist, 2),
                     "target_hidden_share": round(share, 4), "need": round(n_per_pass * Dist ** 2 * share, 1),
                     "success": success, "instrument": instrument, "note": note})

    # own past + action (a global input every token reads)
    add("movement", "own_past+action", freq["move"], t1["movement"]["d_median"], 1.00, "frames (T1 bank, K=16)")
    for k, blk in (("table", "grass"), ("furnace", "grass"), ("plant", "grass")):
        Dist = dist(Bk[k], Bk[blk]) if k in Bk and blk in Bk else None
        if Dist:
            add(f"place_{k} (faced cell)", "own_past+action", freq[f"place_{k}"], Dist, None, "-", "frame success not measured")
    for k in ("wood_pickaxe", "stone_pickaxe", "iron_pickaxe", "wood_sword", "stone_sword", "iron_sword"):
        Dist = step(k, 0, 1)
        add(f"make_{k} (tool slot 0 -> 1)", "own_past+action", freq[f"make_{k}"], Dist, None, "-", "frame success not measured; head/rec_head below")
    # the faced cell: its own past token is the block, the action DO global
    exp2 = json.loads((CV.HERE / "readings/curves.json").read_text())["curves"]["mining"]
    t2 = json.loads((HERE / "readings/t2/curves.json").read_text())["curves"]["mining"]
    for item, (blk, left) in {"coal": ("coal", "path"), "wood": ("tree", "grass"), "stone": ("stone", "path"), "iron": ("iron", "path")}.items():
        cc = np.mean([t2[s]["cell_cleared"]["gen"] for s in t2 if s.startswith(item + "_")])
        add(f"mine {item}: faced cell cleared", "own_past+action", freq[f"cell_cleared_{'tree' if item == 'wood' else item}"],
            dist(Bk[blk], Bk[left]), round(float(cc), 3), "frames (T2, prefix-20)")
    # the HUD slot: its own past is the count; WHAT happened lives at the faced cell (another token) or in memory
    for item in ("wood", "stone", "coal", "iron"):
        ap = t2[f"{item}_0"]["count_next"]["gen"]
        add(f"mine {item}: slot 0 -> 1", "another_token", freq[f"{item}_appear"], step(item, 0, 1), round(ap, 3), "frames (T2, prefix-20)")
        inc = [(geo["exposure"][item][str(n)]["mining"], t2[f"{item}_{n}"]["count_next"]["gen"], step(item, n, n + 1))
               for n in range(1, 9) if f"{item}_{n}" in t2 and t2[f"{item}_{n}"]["histories"] >= 25 and step(item, n, n + 1)]
        tot = sum(x[0] for x in inc)
        add(f"mine {item}: slot n -> n+1 (n >= 1)", "another_token", freq[f"{item}_increment"],
            float(np.sqrt(sum(x[0] * x[2] ** 2 for x in inc) / tot)) if tot else None,
            round(float(sum(x[0] * x[1] for x in inc) / tot), 3) if tot else None, "frames (T2, prefix-20), exposure-weighted over n")
    curves2 = json.loads((HERE / "readings/t2/curves.json").read_text())["curves"]
    add("energy 9 -> wake (map + lighting)", "another_token", freq["wake_energy9"], t1["energy"]["d_median"],
        round(curves2["asleep"]["e9"]["awake"]["gen"], 3), "frames (T2, prefix-20)", "and 0.06-0.11 false wakes at 6-8")
    # hidden: needs history
    add("damage when an adjacent zombie is ready", "history (hidden)", freq["damage"], step("health", 7, 5) or 3.0,
        round(curves2["zombie"]["awake_ready_safe"]["damage"]["gen"] / curves2["zombie"]["awake_ready_safe"]["damage"]["true"], 3),
        "frames (T2), share of the recorded rate")
    add("cow killed: leaves the faced cell", "history (hidden)", freq["cow_killed"], dist(Mb["cow"], Bk["grass"]),
        round(curves2["hit"]["cow_kill"]["mob_gone"]["gen"], 3), "frames (T2)", "non-kill hits 0.62 (truth 0.38)")
    add("cow killed: food +6", "history (hidden) + another_token", freq["cow_killed"], step("food", 3, 9) or 5.0,
        round(curves2["hit"]["cow_kill"]["food_up"]["gen"] / curves2["hit"]["cow_kill"]["food_up"]["true"], 3), "frames (T2), share")
    # persistence (the default, no change): reported for completeness, need undefined
    head = json.loads((TR.ROOT / "artifacts/experiments/20261105_p1f3/readings/interface_steps.json").read_text())["checkpoints"]["f3_16steps"]["classes"]
    hr = {k[4:]: round(v["gen"]["unlock_p"]["est"] / max(v["rec_head"]["unlock_p"]["est"], 1e-9), 3) for k, v in head.items() if k.startswith("ach_")}
    out = {"passes": round(passes, 2), "frequencies_per_pass": freq, "rows": rows, "head_ratio_gen_over_rec_head": hr}
    (HERE / "readings/need.json").write_text(json.dumps(out, indent=1))
    for r in rows:
        print(f"{r['relation'][:46]:46s} {r['determinant_lives'][:26]:26s} n/pass {r['per_pass']:>8} D {r['D']} hidden {r.get('target_hidden_share')} need {r['need']} success {r['success']}", flush=True)
    print(json.dumps(hr), flush=True)


if __name__ == "__main__":
    main()
