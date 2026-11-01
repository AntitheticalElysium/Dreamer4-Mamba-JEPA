"""Is P1-F's broad regression (RESULTS.md 5: C2ST, G1 by class, movement, counts, reward calibration, all read under
random gaze) real, or its evaluation off P1-F's training gaze? User 10-31: run, then report to the user and Codex.
Each world (P1 world 1 ckpt 10, P1-F ckpt 10) under each gaze:
- `random`: S1's seeded fixations (world 1's training distribution; S1 as read so far);
- `guided`: the same draws, except frame t >= 1 decided by a rule context fixates the rule's window (M0.5's Q = 1
  rule, as GuidedStreams placed it in P1-F's training; M0.5's evaluator's guided route).
Readings, all through the existing instruments, the only change being the encoding's fixations:
1. S1 in full (saccade.eval.s1.main, its encode bound to the gaze): S1.3's C2ST, power controls, reader counts, reward
   calibration primary; S1.1 / S1.2 / S1.4 descriptive.
2. Dev G1 loss by class (M0.5's class_losses: same windows and noise for every world and gaze) and its total over
   periphery cells.
3. Movement shift (M0.5's MOVE rows, catalogue.run, its periphery-route seed for every world and gaze).
Declared reading (before the run): P1-F guided against world 1 random (each world's own training gaze). If P1-F's
readings come back within measurement spread of world 1's (C2ST fits vary by up to ~0.07), the regression read under
random gaze is the placement's off-distribution effect. If they stay clearly worse, it is a regression of the recipe,
and guidance in training vs the harmoniser re-weighting is not separable by reading. World 1 guided vs random is the
instrument check (world 1 read the same under both gazes at the actor interface). Random rows reproduce earlier
readings up to the fit spread.
Usage (project venv, GPU): python -m artifacts.experiments.20261030_p1_guided.own_gaze OUT_DIR [--smoke]
"""
import importlib, json, os, sys
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")
import numpy as np
import torch

from saccade import model as M, train as TR
from saccade.eval import s1 as S

PM = importlib.import_module("artifacts.experiments.20261030_fovea_placement.placement_map")
EV = importlib.import_module("artifacts.experiments.20261030_fovea_placement.evaluate")
M05 = importlib.import_module("artifacts.experiments.20261030_m05_guided_fovea.evaluate")
HERE = Path(__file__).resolve().parent
WORLDS = {"w1": TR.ROOT / "artifacts/saccade_m0/p1_world1/ckpt_10.pt", "p1f": HERE / "run/ckpt_10.pt"}


def main():
    out_dir, smoke = Path(sys.argv[1]), "--smoke" in sys.argv
    out_dir.mkdir(parents=True, exist_ok=True)
    rule = {int(k): tuple(v) for k, v in json.loads((PM.HERE / "map.json").read_text())["rule"].items()}
    L = S.first_episodes(dict(np.load(S.LABELS / "s0_dev_labels.npz")), 20 if smoke else 0)
    windows = {"random": lambda L_: None, "guided": lambda L_: PM.window_of_state(L_, rule)}
    encode, res = S.encode, {}
    for name, path in WORLDS.items():
        ck = torch.load(path, weights_only=False, map_location="cpu")
        cfg = ck["cfg"]
        model = M.M0(cfg).to(S.DEV); model.load_state_dict(ck["model"]); model.eval()
        res[name] = {"checkpoint": str(path), "sha256": TR.file_sha256(path)}
        del ck
        enc = {g: encode(model, L, cfg, windows=w(L)) for g, w in windows.items()}
        res[name]["frames_refixated"] = float(np.mean([(r["fix"][1:] != enc["random"][e]["fix"][1:]).any(-1).float().mean()
                                                    for e, r in enumerate(enc["guided"])]))
        rows, _ = M05.rows_of(L, enc["random"], 50 if smoke else M05.N)
        hm, tm = M05.CAT.hud_means(L, enc["random"]), M05.CAT.table_means(L, enc["random"])
        for g in windows:
            cl = EV.class_losses(model, enc[g], L, cfg)
            cells = [k for k in cl if k not in ("glimpse", "lighting")]
            cl["periphery_all"] = {"se_per_element": sum(cl[k]["se_per_element"] * cl[k]["elements"] for k in cells)
                                   / sum(cl[k]["elements"] for k in cells), "elements": sum(cl[k]["elements"] for k in cells)}
            o = M05.CAT.run(model, enc[g], L, rows["move"], "move", hm, tm, cfg,
                            torch.Generator(device=S.DEV).manual_seed(EV.SEEDS["move"]), heun=False, keep=True)
            ep = np.array([rows["move"][i][0] for i in o["row_ctx"]])
            res[name][g] = {"class_losses": cl, "movement_shift_sample": M05.est(o["f_gen"], o["valid"], ep)}
            print(name, g, "G1 periphery", round(cl["periphery_all"]["se_per_element"], 4),
                  "movement", round(res[name][g]["movement_shift_sample"]["est"], 4), flush=True)
            (out_dir / "g1_movement.json").write_text(json.dumps(res, indent=1))
        del model, enc
        torch.cuda.empty_cache()
        for g, w in windows.items():
            S.encode = lambda model_, L_, cfg_, untrained=None, w=w: encode(model_, L_, cfg_, untrained, windows=w(L_))
            sys.argv = ["s1", str(path), str(out_dir / f"s1_{name}_{g}.json")] + (["20", "20"] if smoke else [])
            S.main()
        S.encode = encode
    print("saved", out_dir, flush=True)


if __name__ == "__main__":
    main()
