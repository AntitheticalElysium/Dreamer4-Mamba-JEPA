"""Review of 20261106_mamba_hostile_path (Codex), post-hoc on its saved generated futures (latents/NNN.pt, 6 conditions x
16 draws per scene; no generation): is the danger response DRAWN in the generated frame or read by the death head from
the after-action belief? The health digit of every generated frame is read by the determinant-curves health reader
(../20261106_determinant_curves/curves.py fit_readers: F3 tokens of train400, S0 probe protocol, seed 0; validated here
on the first 100 dev episodes), beside Codex's own per-draw head scores (native_results.json). Calibration: the same
reader on the bank's simulator successors (bank 'health'). Writes frame_reads.json.
Usage (project venv, GPU): MAMBA_DETERMINISTIC=1 python -m artifacts.experiments.20261106_mamba_hostile_path_review.frame_reads
"""
import copy, importlib, json, os
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")
import numpy as np
import torch

from saccade import train as TR
from saccade.eval import s1 as S

CV = importlib.import_module("artifacts.experiments.20261106_determinant_curves.curves")
HERE = Path(__file__).resolve().parent
SRC = TR.ROOT / "artifacts/experiments/20261106_mamba_hostile_path"
NAMES = ["empty", "both", "fine", "coarse", "cow_fine", "natural_both"]


def main():
    model, cfg = CV.IF.load(CV.CKPT)
    cfg = copy.deepcopy(cfg)
    Ld = S.first_episodes(dict(np.load(CV.LABELS / "s0_dev_labels.npz")), 100)
    R, valid = CV.fit_readers(model, cfg, Ld, S.encode(model, Ld, cfg), 0)
    s1 = S.load_reader(CV.S1_READER, TR.ROOT / CV.CKPT, CV.DEV)
    rows = json.loads((SRC / "native_results.json").read_text())
    bank = torch.load(SRC / "bank.pt", map_location="cpu", weights_only=False)
    hp = np.array([r["meta"]["hp"] for r in rows])
    gen = {n: {"health": [], "head_death": [], "head_neg": []} for n in NAMES}
    real = {"read": [], "true": []}
    for r, b in zip(rows, bank):
        lat = torch.load(SRC / f"latents/{r['index']:03d}.pt", weights_only=False)
        for name, (p, _, v), scores in zip(NAMES, lat, r["native"]):
            assert scores["condition"] == name
            h = CV.read(R, s1, p[:, 0], v[:, 0], np.full(len(p), 31))["hud63"]
            gen[name]["health"].append(h)
            gen[name]["head_death"].append(np.array(scores["death"])), gen[name]["head_neg"].append(np.array(scores["negative_reward"]))
        frames = torch.as_tensor(b["future"]).reshape(-1, 63, 63, 3).to(CV.DEV)
        with torch.no_grad(), TR.autocast(CV.DEV, cfg):
            P = model.perception(frames, torch.tensor([7, 1], device=CV.DEV).expand(len(frames), -1))
        real["read"].append(CV.read(R, s1, P.p.float().cpu(), P.v.float().cpu(), np.full(len(frames), 31))["hud63"].reshape(3, -1))
        real["true"].append(np.clip(b["health"], 0, 9))
    out = {"readers": {k: valid[k] for k in ("63",)}, "conditions": {}}
    rr, rt = np.stack(real["read"]), np.stack(real["true"])
    out["real_successor"] = {"reader_agreement": float((rr == rt).mean()),
                             "dead_read_by_source": dict(zip(["empty", "zombie", "cow"], (rr == 0).mean((0, 2)).tolist())),
                             "dead_true_by_source": dict(zip(["empty", "zombie", "cow"], (rt == 0).mean((0, 2)).tolist()))}
    for name in NAMES:
        h = np.stack(gen[name]["health"])                                # [scenes, draws]
        d, ng = np.stack(gen[name]["head_death"]), np.stack(gen[name]["head_neg"])
        dead, hurt = h == 0, h < hp[:, None]
        out["conditions"][name] = {
            "frame_dead": float(dead.mean()), "frame_damaged": float(hurt.mean()), "head_death": float(d.mean()),
            "head_negative": float(ng.mean()),
            "head_death_on_frame_dead": float(d[dead].mean()) if dead.any() else None,
            "head_death_on_frame_alive": float(d[~dead].mean()),
            "share_of_head_death_mass_on_alive_frames": float(d[~dead].sum() / d.sum()),
            "health_read_counts": np.bincount(h.ravel(), minlength=10).tolist()}
    (HERE / "frame_reads.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1), flush=True)


if __name__ == "__main__":
    main()
