"""Stage 1: pre-encode everything once through the FROZEN Raw LeWM encoder.

The three arms differ only in which targets their world sees. Pre-encoding guarantees that:
the encoder never moves, every arm reads identical latents, and no arm can gain from encoder
drift or from a different preprocessing path.

Two pools:

  factual   windows drawn by the gate's own `JointSampler` from support_v2, at the recipe's
            window layout. This is exactly what Raw-10k was trained on, so arm A's continuation
            is a faithful continuation rather than a new distribution.

  fork      artifacts/eda/broad_forks_v2 -- the ~15.5k hazard-choice roots Direct was trained on
            and LeWM has never seen. Each keeps a 32-frame history and all 17 successors; we
            encode the last `lewm_context` frames plus every successor, so a root can supply
            EITHER its single factual transition (arm A') or all 17 siblings (arm B).

Their latents were previously only ever encoded by the legacy 64-slot tokenizer, so this is a
genuine re-encode, not a cache reuse.

Leakage: fork seeds are 15000-16504. The sealed evaluation roots are 14000-14511 and the
confirmation panel drew 13000-14xxx. Asserted below.
"""

import argparse
import glob
import json
import sys
from pathlib import Path

import torch

ROOT = Path("/home/antithetical/EPITA/PERSO/DynamicHorizons-Mamba-JEPA")
HERE = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from d4mj.config import load_recipe
from d4mj.data import JointSampler, _sha256, load_joint_corpus
from d4mj.m03.gate import M03Settings, load_m03_bundle

RAW = ROOT / "artifacts/lewm_gates_20260906/paired/raw/joint/step-010000.pt"
DATASET = ROOT / "artifacts/craftax_support_v2/manifest.json"
FORKS = ROOT / "artifacts/eda/broad_forks_v2"
RECIPE = ROOT / "d4mj/recipes/lewm_mamba_raw.json"
EVAL_SEEDS = range(13000, 14512)      # sealed evaluation + confirmation panel


@torch.inference_mode()
def encode(encoder, frames, batch=256):
    """frames [N, H, W, 3] uint8 -> z [N, latent_dim]. Framewise; the encoder has no memory."""
    out = []
    for start in range(0, len(frames), batch):
        chunk = frames[start:start + batch].unsqueeze(1).to(encoder.pixel_mean.device)
        z, _ = encoder.projected_and_cls(chunk)
        out.append(z[:, 0, 0].cpu())
    return torch.cat(out)


def factual_pool(encoder, config, episodes, windows, device):
    """`windows` draws from the gate's own sampler, encoded once."""
    sampler = JointSampler(episodes, config, torch.Generator().manual_seed(config.seed + 1))
    frames, actions = [], []
    drawn = 0
    while drawn < windows:
        batch = sampler.sample()
        frames.append(batch.frames)
        actions.append(batch.actions)
        drawn += len(batch.frames)
        if drawn % 4096 < config.joint.batch:
            print(json.dumps({"stage": "factual", "drawn": drawn, "of": windows}), flush=True)
    frames = torch.cat(frames)[:windows]
    actions = torch.cat(actions)[:windows]
    n, t = frames.shape[:2]
    z = encode(encoder, frames.reshape(-1, *frames.shape[2:])).reshape(n, t, -1)
    return {"z": z, "actions": actions}


def fork_pool(encoder, settings, device):
    """Every hazard-choice root: its context, its 17 successors, its factual action, its labels."""
    paths = sorted(glob.glob(str(FORKS / "*.pt")))
    width = settings.lewm_context
    hist, past, branch, factual, health, terminated, seeds = [], [], [], [], [], [], []
    for position, path in enumerate(paths):
        for row in torch.load(path, map_location="cpu", weights_only=False):
            if row["seed"] in EVAL_SEEDS:
                raise RuntimeError(f"fork root on a sealed evaluation seed: {row['seed']}")
            hist.append(row["frames"][-width:])
            past.append(row["led_to_action"][-width + 1:])
            branch.append(row["successors"])
            factual.append(int(row["bc_action"]))
            health.append(row["health_delta"])
            terminated.append(row["terminated"])
            seeds.append(int(row["seed"]))
        if position % 200 == 0:
            print(json.dumps({"stage": "fork_load", "shard": position, "of": len(paths),
                              "roots": len(hist)}), flush=True)
    hist = torch.stack(hist)
    branch = torch.stack(branch)
    n, t = hist.shape[:2]
    z_hist = encode(encoder, hist.reshape(-1, *hist.shape[2:])).reshape(n, t, -1)
    z_branch = encode(encoder, branch.reshape(-1, *branch.shape[2:])).reshape(n, 17, -1)
    return {"z_history": z_hist, "past_actions": torch.stack(past),
            "z_branch": z_branch, "factual_action": torch.tensor(factual),
            "health_delta": torch.stack(health), "terminated": torch.stack(terminated),
            "seed": torch.tensor(seeds)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=HERE / "cache")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--windows", type=int, default=40960)
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)

    config = load_recipe(RECIPE)
    settings = M03Settings()
    bundle, _, _ = load_m03_bundle(RAW, device=args.device, dataset_sha256=_sha256(DATASET))
    encoder = bundle.encoder.eval()
    if any(m.training for m in encoder.modules()):
        raise RuntimeError("encoder is not fully frozen")

    if not (args.out / "fork.pt").exists():
        pool = fork_pool(encoder, settings, args.device)
        torch.save(pool, args.out / "fork.pt")
        print(json.dumps({"stage": "fork_done", "roots": len(pool["z_branch"]),
                          "seeds": [int(pool["seed"].min()), int(pool["seed"].max())]}), flush=True)
    if not (args.out / "factual.pt").exists():
        episodes, contract = load_joint_corpus(str(DATASET), config)
        pool = factual_pool(encoder, config, episodes, args.windows, args.device)
        pool["dataset_sha256"] = _sha256(DATASET)
        torch.save(pool, args.out / "factual.pt")
        print(json.dumps({"stage": "factual_done", "windows": len(pool["z"])}), flush=True)
    manifest = {"schema": "d4mj_cf_arms_pool_v1", "encoder": str(RAW.relative_to(ROOT)),
                "encoder_sha256": _sha256(RAW), "recipe": str(RECIPE.relative_to(ROOT)),
                "lewm_context": settings.lewm_context,
                "fork_sha256": _sha256(args.out / "fork.pt"),
                "factual_sha256": _sha256(args.out / "factual.pt")}
    (HERE / "evidence").mkdir(exist_ok=True)
    (HERE / "evidence/pool_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"status": "pools_complete"}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
