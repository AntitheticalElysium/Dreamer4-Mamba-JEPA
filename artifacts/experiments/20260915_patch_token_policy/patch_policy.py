"""Does a token-preserving head beat pooled export at predicting the logged action?

TC-LeWM's downstream policy keeps, per camera, the CLS token *and* a 4x4 spatially
pooled grid of patch tokens, and lets action queries cross-attend to that sequence
(paper Table 4: "34 visual tokens via cross-attention").  Patch tokens are never
regularized by SIGReg and never predicted by the world model -- the policy reads
*observed* tokens.  Our export is projected CLS only, and our head mean-pools
before an MLP, so we have never tested that pathway.

This does, on the frozen M03 encoders, with one fixed head.

This is real behaviour cloning.  ``support_v2`` carries no BC-eligible episodes
(0 of 10,080; its rollouts are epsilon-noisy), but ``craftax_expert_v1`` does:
320 PPO-expert episodes, 696,746 transitions, 20.6 of 22 mean achievements.
``expert.load_archive`` converts it exactly -- the 64x64 frames are Craftax's
native 63x63 plus a zero-padded row and column, so the crop is lossless and the
CHW->HWC permute is a view (S49).  Both are verified: the discarded row and column
contain only zeros.  Splits are whole-episode 80/10/10 by ``episode_splits``.

No retrain is required for this pathway.  The paper freezes the encoder before
policy learning and never regularizes or predicts patch tokens, so a faithful
patch implementation is an export and policy-interface change, not a world-model
change.  Frame skip is the part that does need a retrain; see the README.

Conditions, all reading the same frames with one parameter-identical head:

* ``z``            -- projected latent, 1 token (today's export)
* ``cls``          -- CLS token, 1 token
* ``patch16``      -- 4x4 pooled patch grid, 16 tokens
* ``cls_patch16``  -- CLS + 4x4 grid, 17 tokens (the paper-shaped condition)
* ``patch16_mean`` -- the 16 tokens averaged to 1, isolating cross-attention over
  tokens from the information in them; this is what ``agent.py`` does today
* ``direct``       -- Direct's native spatial latent, 32 tokens of 32 dims, at its
  native 64-frame contextual encoding.  Its 64x16 bottleneck is repacked at
  ``packing=2`` into the 32x32 spatial axis the world actually consumes.  Direct
  trained on this archive and the LeWM arms did not, so it is a flagged anchor,
  never a matched comparison

Every token is zero-padded to width 192, so the head is parameter-identical in
every condition and only the sequence length changes.  The LeWM-internal rows are
matched; the Direct row is not.

Scope: one-step action prediction from a single observed frame.  Behaviour cloning
of an expert, not control, not a rollout, and it cannot authorize M4.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[3]
SCHEMA = "d4mj_patch_token_policy"
ARCHIVE = ROOT / "d4_mamba_jepa/artifacts/expert/craftax_expert_v1.pt"
CONDITIONS = ("z", "cls", "patch16", "cls_patch16", "patch16_mean", "direct")
TOKEN_WIDTH = 192
N_ACTIONS = 17
PREFIX = 63  # Direct's native contextual prefix; LeWM is framewise and ignores it


class Head(nn.Module):
    """One learned action query cross-attending to the visual tokens.

    Identical parameters in every condition: tokens are padded to a common width
    before the input projection, so only the sequence length varies.
    """

    def __init__(self, width: int, heads: int, seed: int):
        super().__init__()
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.input = nn.Linear(TOKEN_WIDTH, width)
            self.query = nn.Parameter(torch.randn(1, 1, width) * 0.02)
            self.attention = nn.MultiheadAttention(width, heads, batch_first=True)
            self.norm_q = nn.LayerNorm(width)
            self.norm_v = nn.LayerNorm(width)
            self.mlp = nn.Sequential(nn.Linear(width, 4 * width), nn.GELU(), nn.Linear(4 * width, width))
            self.out = nn.Linear(width, N_ACTIONS)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        values = self.norm_v(self.input(tokens))
        query = self.norm_q(self.query.expand(len(tokens), -1, -1))
        attended, _ = self.attention(query, values, values, need_weights=False)
        hidden = attended + query
        return self.out(hidden + self.mlp(hidden))[:, 0]


def _episodes(count: int | None, frames: int, seed: int) -> dict[str, list[dict]]:
    """Whole-episode BC splits from the expert archive; windows never cross one."""
    from d4mj.config import Config
    from d4mj.data import episode_splits
    from d4mj.expert import load_archive

    config = Config()
    archive = load_archive(ARCHIVE, config)
    train, dev, _ = episode_splits(len(archive), config.seed)
    rng = np.random.default_rng(seed)
    out = {}
    for name, index in (("train", train), ("dev", dev)):
        selected = index.tolist()[: count] if count else index.tolist()
        rows = []
        for number, slot in enumerate(selected):
            episode = archive[slot]
            usable = len(episode.actions_taken)
            if usable < frames + PREFIX:
                continue
            start = int(rng.integers(PREFIX, usable - frames))
            rows.append({"observations": episode.observations[start - PREFIX:start + frames].clone(),
                         "actions": episode.actions_taken[start:start + frames].clone(),
                         "episode": number, "slot": int(slot)})
        if len(rows) < 2:
            raise RuntimeError(f"patch_policy: {name} has {len(rows)} usable expert episodes")
        out[name] = rows
    return out


@torch.inference_mode()
def _tokens(bundle, episodes: list[dict], *, direct: bool, batch: int) -> dict[str, torch.Tensor]:
    """Frozen visual tokens for every sampled frame, padded to a common width."""
    captured: dict[str, torch.Tensor] = {}
    rows: dict[str, list[torch.Tensor]] = {name: [] for name in CONDITIONS if (name == "direct") == direct}
    handle = None
    if not direct:
        handle = bundle.encoder.backbone.register_forward_hook(
            lambda module, args, output: captured.__setitem__("h", output.last_hidden_state))
    try:
        for record in episodes:
            observations = record["observations"]
            target = observations[63:]
            if direct:
                # Native contextual encoding: one causal pass, then the positions
                # that carry a full prefix.  Never a single-frame MAE call.
                latents = bundle.encode(observations[None].to(bundle.device))[0, 63:]
                rows["direct"].append(latents.flatten(1, -2).cpu() if latents.ndim > 3 else latents.cpu())
                continue
            for start in range(0, len(target), batch):
                chunk = target[start:start + batch].to(bundle.device)
                z, cls = bundle.encoder.projected_and_cls(chunk[:, None])
                grid = captured["h"][:, 1:]
                side = int(round(grid.shape[1] ** 0.5))
                pooled = nn.functional.adaptive_avg_pool2d(
                    grid.transpose(1, 2).reshape(len(chunk), -1, side, side), 4
                ).flatten(2).transpose(1, 2)
                rows["z"].append(z[:, 0].cpu())
                rows["cls"].append(cls[:, 0, None].cpu())
                rows["patch16"].append(pooled.cpu())
                rows["cls_patch16"].append(torch.cat((cls[:, 0, None], pooled), 1).cpu())
                rows["patch16_mean"].append(pooled.mean(1, keepdim=True).cpu())
    finally:
        if handle is not None:
            handle.remove()
    out = {}
    for name, values in rows.items():
        stacked = torch.cat(values).float()
        if stacked.shape[-1] < TOKEN_WIDTH:
            stacked = torch.cat((stacked, torch.zeros(*stacked.shape[:-1], TOKEN_WIDTH - stacked.shape[-1])), -1)
        out[name] = stacked
    return out


def _train_head(train_x, train_y, dev_x, settings, device) -> torch.Tensor:
    model = Head(settings["width"], settings["heads"], settings["seed"]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings["lr"], weight_decay=settings["weight_decay"])
    generator = torch.Generator().manual_seed(settings["seed"] + 1)
    for _ in range(settings["steps"]):
        index = torch.randint(len(train_x), (settings["batch"],), generator=generator)
        loss = nn.functional.cross_entropy(model(train_x[index].to(device)), train_y[index].to(device))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
    model.eval()
    with torch.no_grad():
        return torch.cat([model(dev_x[start:start + 1024].to(device)).argmax(-1).cpu()
                          for start in range(0, len(dev_x), 1024)])


def _accuracy(prediction, truth, roots, draws, seed) -> dict:
    from d4mj.m03.gate import _root_bootstrap
    correct = (prediction == truth).float()
    point, interval = _root_bootstrap(correct, roots, lambda rows: float(correct[rows].mean()),
                                      draws=draws, seed=seed)
    return {"top1": point, "interval": interval,
            "status": "measured" if interval is not None else "insufficient_coverage"}


def run(device: str, episodes: int, frames: int, batch: int, settings: dict,
        frozen_eval_proof: Path | None = None) -> dict:
    from d4mj.data import _sha256
    from d4mj.m03.gate import _legacy_anchor, load_m03_bundle

    os.environ["TRITON_F32_DEFAULT"] = "ieee"
    source = ROOT / "artifacts/lewm_gates_20260906/m03_bootstrap/evaluation_v2"
    contract = json.loads((source / "run.json").read_text())["contract"]
    dataset_sha256 = contract["dataset"]["sha256"]

    splits = _episodes(episodes, frames, settings["seed"])
    labels = {s: torch.cat([r["actions"] for r in rows]).long() for s, rows in splits.items()}
    roots = {s: torch.cat([torch.full((len(r["actions"]),), r["episode"]) for r in rows])
             for s, rows in splits.items()}
    majority = int(torch.bincount(labels["train"], minlength=N_ACTIONS).argmax())

    report = {"schema": SCHEMA, "device": device,
              "data": {"archive": str(ARCHIVE), "archive_sha256": _sha256(ARCHIVE),
                       "kind": "bc_eligible PPO expert, lossless 64->63 crop"},
              "frozen_eval_proof": ({"path": str(frozen_eval_proof), "sha256": _sha256(Path(frozen_eval_proof))}
                                    if frozen_eval_proof else None),
              "script_sha256": _sha256(Path(__file__)),
              "target": "expert action; behaviour cloning on whole-episode 80/10/10 splits",
              "episodes": {s: len(rows) for s, rows in splits.items()},
              "frames_per_episode": frames, "samples": {s: len(v) for s, v in labels.items()},
              "head": settings, "parameter_note": {
                  "ours_encoder_vit_tiny_p7_63": 6175872, "ours_world_mamba2_w256": 2503496,
                  "tclewm_encoder_vit_tiny_p14_224": "same ViT-Tiny family; 256 patch tokens vs our 81",
                  "tclewm_predictor_arpredictor_d6_h16": 9456384,
                  "note": "their predictor is ~3.8x ours; their encoder sees 3.2x the patch tokens"},
              "majority_action_floor": _accuracy(torch.full_like(labels["dev"], majority), labels["dev"],
                                                 roots["dev"], settings["draws"], settings["seed"] + 9),
              "arms": {}, "m4_authorized": False}

    arms = [("raw", False), ("tc", False), ("direct_attention", True)]
    for arm, direct in arms:
        if direct:
            bundle, identity = _legacy_anchor(arm, device=device)
        else:
            bundle, payload, _ = load_m03_bundle(Path(contract[f"{arm}_checkpoint"]["path"]),
                                                 device=device, dataset_sha256=dataset_sha256,
                                                 frozen_eval_proof=frozen_eval_proof)
            identity = {"checkpoint": _sha256(Path(contract[f"{arm}_checkpoint"]["path"]))}
            del payload
        features = {s: _tokens(bundle, splits[s], direct=direct, batch=batch) for s in splits}
        del bundle
        if device == "cuda":
            torch.cuda.empty_cache()
        arm_report = {"identity": identity, "conditions": {}}
        for name, values in features["train"].items():
            prediction = _train_head(values, labels["train"], features["dev"][name], settings, device)
            arm_report["conditions"][name] = {
                "tokens": int(values.shape[1]),
                **_accuracy(prediction, labels["dev"], roots["dev"], settings["draws"],
                            settings["seed"] + 20 + CONDITIONS.index(name))}
        report["arms"][arm] = arm_report
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path(__file__).parent / "evidence")
    parser.add_argument("--device", choices=("cpu", "cuda"),
                        default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--episodes", type=int, default=0, help="cap on episodes per split; 0 uses all")
    parser.add_argument("--frames", type=int, default=256, help="consecutive frames sampled per episode")
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--limit", action="store_true", help="tiny structural smoke; never a result")
    parser.add_argument("--frozen-eval-proof", type=Path,
                        default=ROOT / "d4mj/m03/frozen_eval_compat.json",
                        help="measured source-delta proof; evaluation only, never training resume")
    args = parser.parse_args(argv)

    import sys
    sys.path.insert(0, str(ROOT))
    settings = {"width": 128, "heads": 4, "lr": 3e-4, "weight_decay": 1e-4,
                "batch": 256, "steps": 3000, "seed": 20260915, "draws": 1000}
    episodes, frames = args.episodes, args.frames
    if args.limit:
        settings = {**settings, "steps": 20, "batch": 32, "draws": 16}
        episodes, frames = 4, 24
    args.out.mkdir(parents=True, exist_ok=True)
    destination = args.out / ("policy.smoke.json" if args.limit else "policy.json")
    if destination.exists():
        raise FileExistsError(f"patch_policy: refusing to replace {destination}")
    report = run(args.device, episodes, frames, args.batch, settings, args.frozen_eval_proof)
    if args.limit:
        report["mode"] = "structural_smoke_not_a_result"
    destination.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"status": "complete", "report": str(destination)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
