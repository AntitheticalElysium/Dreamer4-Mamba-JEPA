"""Innovation EDA (simulator, CPU): how much of each next frame is genuinely random given the full state and the
action? Dev states replayed exactly (artifacts/eda/replay.py), stepped with the recorded action under K fresh keys,
rendered as seen and in full daylight (night static is drawn from the step's key, renderer.py 414: pixel noise, not
content); per 7 x 7 cell of the 9 x 9 view, does the next frame vary across keys (map rows 0-6 vs HUD rows 7-8), and do
the HUD body values or the inventory vary. Sanity: the replayed state renders the stored frame, and the recorded next frame is reachable
in spirit (its HUD equals every key's when the HUD never varies).
Usage: python -m artifacts.experiments.20261107_f4_followups.innovation N K"""
import json, os, sys
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")
import jax
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "eda"))
import replay as RP                                                      # noqa: E402

HERE = Path(__file__).resolve().parent
LABELS = HERE.parents[1] / "eda/s0"


def main(n, k):
    L = np.load(LABELS / "s0_dev_labels.npz")
    T = L["ep_T"]
    rng = np.random.default_rng(2118)
    eps = rng.choice(len(T), n)
    _, _, _, step, frame = RP.env_and_render()
    keys = jax.random.split(jax.random.PRNGKey(2118), k)
    out = {"cells": [], "content_map": [], "content_hud": [], "hud_body": 0, "inventory": 0, "zombies": 0, "cows": 0, "skeletons": 0, "arrows": 0, "map": 0,
           "any": 0, "render_check_max": 0, "hud_equal_recorded": 0, "samples": 0}
    for i, e in enumerate(eps):
        sh, sl = int(L["ep_shard"][e]), int(L["ep_slot"][e])
        t = int(rng.integers(1, T[e]))
        f = RP.episode_fields(sh, sl)
        s = RP.advance_to(sh, sl, t)
        out["render_check_max"] = max(out["render_check_max"], int(np.abs(np.asarray(frame(s)).astype(int) - f["observations"][t].numpy().astype(int)).max()))
        a = int(f["actions_taken"][t])
        nxt = [step(key, s, a)[1] for key in keys]
        img = np.stack([np.asarray(frame(x)) for x in nxt]).astype(int)          # [K, 63, 63, 3]
        vary = lambda im: (im != im[:1]).any(0).any(-1).reshape(9, 7, 9, 7).any((1, 3))   # [9, 9] varies across keys
        cells = vary(img)
        day = vary(np.stack([np.asarray(frame(x.replace(light_level=np.float32(1.0)))) for x in nxt]).astype(int))
        out["cells"].append(int(cells.sum()))
        out["content_map"].append(int(day[:7].sum())), out["content_hud"].append(int(day[7:].sum()))
        same = lambda u, v: all(same(p, q) for p, q in zip(u, v)) if isinstance(u, (tuple, list)) else np.array_equal(np.asarray(u), np.asarray(v))
        var = lambda get: any(not same(get(x), get(nxt[0])) for x in nxt[1:])
        flags = {"hud_body": var(lambda x: (x.player_health, x.player_food, x.player_drink, x.player_energy, x.is_sleeping)),
                 "inventory": var(lambda x: [getattr(x.inventory, f_) for f_ in ("wood", "stone", "coal", "iron", "diamond", "sapling")]),
                 "zombies": var(lambda x: (x.zombies.position, x.zombies.mask)), "cows": var(lambda x: (x.cows.position, x.cows.mask)),
                 "skeletons": var(lambda x: (x.skeletons.position, x.skeletons.mask)), "arrows": var(lambda x: (x.arrows.position, x.arrows.mask)),
                 "map": var(lambda x: x.map)}
        for kk, v in flags.items():
            out[kk] += int(v)
        out["any"] += int(cells.any())
        rec = f["observations"][t + 1].numpy().astype(int)
        out["hud_equal_recorded"] += int(np.array_equal(rec[49:], img[0][49:]))    # rows 7-8 of the view (HUD) as pixels
        out["samples"] += 1
        if i % 50 == 0:
            print(i, {kk: out[kk] for kk in flags}, "any", out["any"], flush=True)
    c, cm, ch = (np.array(out.pop(x)) for x in ("cells", "content_map", "content_hud"))
    res = {"samples": out["samples"], "keys": k, "render_check_max_abs_pixel_diff": out["render_check_max"],
           "share of steps with any varying cell": out["any"] / out["samples"],
           "varying cells per step (of 81): mean, median, p90, max": [float(c.mean()), float(np.median(c)), float(np.percentile(c, 90)), int(c.max())],
           "CONTENT (daylight render): share of steps with a varying map cell": float((cm > 0).mean()),
           "CONTENT: varying map cells per step (of 63): mean, median, p90, max": [float(cm.mean()), float(np.median(cm)), float(np.percentile(cm, 90)), int(cm.max())],
           "CONTENT: share of steps with a varying HUD cell": float((ch > 0).mean()),
           "share of steps where this part varies across keys": {kk: out[kk] / out["samples"] for kk in ("hud_body", "inventory", "zombies", "cows", "skeletons", "arrows", "map")},
           "share of steps whose recorded HUD pixels equal key 0's": out["hud_equal_recorded"] / out["samples"]}
    (HERE / "readings/danger/innovation.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1), flush=True)


if __name__ == "__main__":
    main(int(sys.argv[1]), int(sys.argv[2]))
