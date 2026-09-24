"""A declared one-seed CPU screen of action input in real-successor readouts.

Frozen published encodings; same 200-step M03 binary probe recipe. Both conditions
are refitted on CPU. This is a diagnostic screen, not a gate or capacity ceiling.
"""
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from d4mj.m03.gate import M03Settings, _fit_probe_many
from d4mj.diagnostics import binary_auc
from verify_cached_readouts import features, sha


def summary(logits, truth):
    score = logits.reshape(*truth.shape, -1)[..., 0]
    eligible = truth.any(1) & (~truth).any(1)
    selected = score.argmin(1)
    safe = (~truth)[torch.arange(len(truth)), selected]
    valid = (~truth)[:, :, None] & truth[:, None, :]
    comparisons = (score[:, :, None] < score[:, None, :]).float()
    comparisons += .5 * (score[:, :, None] == score[:, None, :]).float()
    within = (comparisons * valid).sum((1, 2)) / valid.sum((1, 2)).clamp_min(1)
    return {"correct": int(safe[eligible].sum()), "opportunity_roots": int(eligible.sum()),
            "safe_selection": float(safe[eligible].float().mean()),
            "within_root_auc": float(within[eligible].mean()),
            "global_death_auc": binary_auc(score.flatten(), truth.flatten()),
            "selected_action_histogram": torch.bincount(selected[eligible], minlength=17).tolist()}


def main():
    torch.set_num_threads(2)
    settings = M03Settings()
    run = ROOT / "artifacts/lewm_gates_20260906/m03_bootstrap/evaluation_v2"
    sidepath = run / "sidecar/sidecar.probe_only.pt"
    rows = torch.load(sidepath, map_location="cpu", weights_only=False, mmap=True)["splits"]
    ty = rows["train"]["outcomes"].flatten(0, 1).float()
    truth = rows["dev"]["outcomes"][..., 0].bool()
    result = {"status": "exploratory_one_seed_screen", "device": "cpu", "threads": 2,
              "settings": vars(settings), "sidecar_sha256": sha(sidepath),
              "script_sha256": sha(__file__), "arms": {}, "m4_authorized": False}
    for arm in ("raw", "tc", "direct_mamba"):
        tr, _ = features(run, arm, "train")
        dv, identity = features(run, arm, "dev")
        tx = tr["observed_successor"].flatten(0, 1).float()
        dx = dv["observed_successor"].flatten(0, 1).float()
        at, ad = torch.eye(17).repeat(len(tx) // 17, 1), torch.eye(17).repeat(len(dx) // 17, 1)
        out = {"identity": identity, "readouts": {}}
        for hidden, family in ((False, "linear"), (True, "mlp")):
            out["readouts"][family] = {}
            for name, train_x, dev_x in (("successor_only", tx, dx),
                                          ("successor_plus_action", torch.cat((tx, at), 1), torch.cat((dx, ad), 1))):
                logits = _fit_probe_many(train_x, ty, {"dev": dev_x}, settings, hidden=hidden, binary=True)["dev"]
                out["readouts"][family][name] = summary(logits, truth)
        result["arms"][arm] = out
        print(arm, out["readouts"], flush=True)
    path = Path(__file__).parent / "action_input_screen.json"
    if path.exists():
        raise FileExistsError(path)
    path.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
