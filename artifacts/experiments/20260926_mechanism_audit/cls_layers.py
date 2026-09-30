"""Read-only, exploratory CLS/patch information trajectory on existing roots.

Same FIT and 54k roots at three immutable Raw joint checkpoints. Labels are
current-frame visible zombie/lava adjacency. Fixed ridge probe (lambda/n=.1)
is fitted only on FIT seeds. This localizes decodable information; it does not
prove why the training objective favored one feature.
"""
import hashlib
import json
import random
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[3]
OLD = ROOT / "artifacts/experiments/20260921_readout_ladder"
sys.path[:0] = [str(ROOT), str(OLD)]
from d4mj.checkpoint import read_lewm_bundle
from d4mj.config import config_from_dict
from d4mj.lewm import LeWMEncoder
from d4mj.data import _sha256
from frozen_ladder import strata, NEIGHBOURS

FIT_PIX = ROOT / "artifacts/eda/broad_forks_v2"
FIT_STATE = ROOT / "artifacts/eda/observe_fit_v1"
JUDGE = ROOT / "artifacts/eda/observe_fresh_v5"
JOINT = ROOT / "artifacts/lewm_m4_canonical/raw/joint"
OUT = Path(__file__).with_name("cls_layers.json")
LAYERS = (1, 3, 6, 9, 12)


def sample():
    partition = json.loads((OLD / "evidence/root_partition.json").read_text())
    fit = list(partition["fit_train"]["seeds"])
    random.Random(406).shuffle(fit)
    pix = {int(p.stem.split("-")[1]): p for p in FIT_PIX.glob("seed-*.pt")}
    state = {int(p.stem.split("-")[1]): p for p in FIT_STATE.glob("seed-*.pt")}
    fit_rows = []
    for seed in fit:
        if seed not in state or seed not in pix:
            continue
        s = {int(r["step"]): r for r in torch.load(state[seed], weights_only=False)}
        p = {int(r["step"]): r for r in torch.load(pix[seed], weights_only=False)}
        fit_rows.extend((seed, step, p[step]["frames"][-1], row["visible"])
                        for step, row in s.items() if step in p)
        if len(fit_rows) >= 2400:
            break
    files = list(JUDGE.glob("seed-*.pt"))
    random.Random(407).shuffle(files)
    judge_rows = []
    for path in files:
        judge_rows.extend((int(r["seed"]), int(r["step"]), r["frames"][-1], r["visible"])
                          for r in torch.load(path, weights_only=False))
        if len(judge_rows) >= 1200:
            break
    def stack(rows):
        frames = torch.stack([r[2] for r in rows]).to(torch.uint8)
        labels = strata(torch.stack([r[3] for r in rows]).float())
        return frames, {name: labels[name] for name in ("zombie_adjacent", "lava_adjacent")}, [(r[0], r[1]) for r in rows]
    return stack(fit_rows), stack(judge_rows)


def auc(score, label):
    pos = score[label].float().cpu()
    neg = score[~label].float().cpu().sort().values
    if not len(pos) or not len(neg):
        return None
    less = torch.searchsorted(neg, pos, right=False).float()
    leq = torch.searchsorted(neg, pos, right=True).float()
    return float(((less + leq) / 2).mean() / len(neg))


def ridge(train, test, y, yj):
    train, test = train.float(), test.float()
    mean, sd = train.mean(0), train.std(0).clamp_min(1e-5)
    a = ((train - mean) / sd).cuda()
    b = ((test - mean) / sd).cuda()
    cy = y.float().mean().cuda()
    eye = torch.eye(a.shape[-1], device=a.device)
    w = torch.linalg.solve(a.T @ a + .1 * len(a) * eye, a.T @ (y.float().cuda() - cy))
    score = (b @ w + cy).cpu()
    return auc(score, yj), score


def paired_auc_difference(a, b, labels, ids, draws=1000):
    """Resample whole evaluation episodes, preserving paired rows and labels."""
    groups = {}
    for i, (seed, _) in enumerate(ids):
        groups.setdefault(seed, []).append(i)
    groups = [torch.tensor(v, dtype=torch.long) for v in groups.values()]
    rng = random.Random(408)
    deltas = []
    for _ in range(draws):
        ix = torch.cat([groups[rng.randrange(len(groups))] for _ in groups])
        aa, bb = auc(a[ix], labels[ix]), auc(b[ix], labels[ix])
        if aa is not None and bb is not None:
            deltas.append(aa - bb)
    deltas.sort()
    return {"difference": auc(a, labels) - auc(b, labels),
            "episode_cluster_95pct": [deltas[int(.025 * len(deltas))], deltas[int(.975 * len(deltas))]],
            "episodes": len(groups), "draws": len(deltas)}


@torch.inference_mode()
def encode(encoder, frames):
    result = {f"cls_L{layer}": [] for layer in LAYERS}
    result.update({"z": [], "local_patch_mean": [], "global_patch_mean": []})
    ids = [r * 9 + c for r, c in NEIGHBOURS]
    for f in frames.split(64):
        pixels = f.cuda().permute(0, 3, 1, 2).float() / 255
        pixels = (pixels - encoder.pixel_mean) / encoder.pixel_std
        h = encoder.backbone(pixels, interpolate_pos_encoding=True,
                             output_hidden_states=True).hidden_states
        for layer in LAYERS:
            result[f"cls_L{layer}"].append(h[layer][:, 0].float().cpu())
        last = h[-1]
        result["z"].append(encoder.projector(last[:, 0]).float().cpu())
        result["local_patch_mean"].append(last[:, 1:][:, ids].mean(1).float().cpu())
        result["global_patch_mean"].append(last[:, 1:].mean(1).float().cpu())
    return {k: torch.cat(v) for k, v in result.items()}


def main():
    (fit, yl, fids), (judge, yj, jids) = sample()
    report = {"status": "exploratory_existing_54k", "rows": {"fit": len(fit), "judge": len(judge)},
              "fit_identity": hashlib.sha256(repr(fids).encode()).hexdigest(),
              "judge_identity": hashlib.sha256(repr(jids).encode()).hexdigest(),
              "counts": {name: {"fit": int(yl[name].sum()), "judge": int(yj[name].sum())}
                         for name in yl}, "lambda_per_n": .1, "checkpoints": {}}
    saved = {}
    for step in (0, 2000, 10000):
        path = JOINT / f"step-{step:06d}.pt"
        payload = read_lewm_bundle(path)
        config = config_from_dict(payload["config"])
        encoder = LeWMEncoder(config).cuda().eval()
        encoder.load_state_dict(payload["modules"]["encoder"])
        xf, xj = encode(encoder, fit), encode(encoder, judge)
        row = {"sha256": _sha256(path), "scores": {}}
        for label in yl:
            fitted = {name: ridge(xf[name], xj[name], yl[label], yj[label]) for name in xf}
            row["scores"][label] = {name: result[0] for name, result in fitted.items()}
            if label == "zombie_adjacent" and step in (0, 10000):
                saved[step] = {name: result[1] for name, result in fitted.items()}
        report["checkpoints"][str(step)] = row
        print(json.dumps({"step": step, "scores": row["scores"]}), flush=True)
        del encoder, xf, xj
        torch.cuda.empty_cache()
    y = yj["zombie_adjacent"]
    report["paired_episode_intervals"] = {
        "trained_vs_initial_cls": paired_auc_difference(saved[10000]["cls_L12"], saved[0]["cls_L12"], y, jids),
        "trained_vs_initial_z": paired_auc_difference(saved[10000]["z"], saved[0]["z"], y, jids),
        "trained_local_vs_cls": paired_auc_difference(saved[10000]["local_patch_mean"], saved[10000]["cls_L12"], y, jids),
        "trained_global_patch_vs_cls": paired_auc_difference(saved[10000]["global_patch_mean"], saved[10000]["cls_L12"], y, jids),
    }
    report["script_sha256"] = _sha256(Path(__file__))
    OUT.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"saved": str(OUT), "counts": report["counts"]}), flush=True)


if __name__ == "__main__":
    main()
