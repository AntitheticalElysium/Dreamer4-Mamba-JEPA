"""P1-F (PLAN.md): saccade.train.p1 itself, with saccade.data.Streams replaced by v2's GuidedStreams bound to M0.5's
Q = 1 placement (the prior is drawn first on every frame, so every random draw matches P1 world 1) and the pixel
decoder's mean loss set by --decoder (faithful by the user's decision, 10-31; nll = P1 world 1's).
Identity (Codex's review of 3d5eea60): the guided placement's identity and the decoder choice are written INTO the
config, which the checkpoint stores and saccade.train.validate_resume compares on every restore (with the source,
corpus and target hashes), so a checkpoint of another recipe (e.g. P1 world 1's) is refused. On first launch the
config minus those two declared changes must equal P1 world 1's checkpoint config; a directory holding a checkpoint
but no identity file is refused.
Usage (project venv, GPU): python -m artifacts.experiments.20261030_p1_guided.train_guided OUT_DIR [--updates N]
  [--no-checkpoints] [--unguided] [--decoder faithful|nll]   (--unguided --decoder nll: P1 world 1's recipe, the match
  check)
"""
import argparse, copy, functools, importlib, json, os
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")          # in case a helper imports craftax: JAX would preallocate the GPU
import numpy as np
import torch

from saccade import data as D, train as TR

FT2 = importlib.import_module("artifacts.experiments.20261030_fovea_placement.finetune")
M05 = importlib.import_module("artifacts.experiments.20261030_m05_guided_fovea.finetune")
PLACE_SHA256 = "0bd48526c8116afba58af702964bfc78655e3e770f806eb0e1875f035277e99b"   # M0.5's identity (log 646af765)
CONTROL = TR.ROOT / "artifacts/saccade_m0/p1_world1/ckpt_10.pt"
WORLD_SEED = 1


def declared(cfg):
    """The config without the two declared changes (guided placement, decoder mean loss)."""
    c = copy.deepcopy(cfg)
    c["p1"].pop("guided", None)
    c["heads"]["pixel_decoder"].pop("mean_loss", None)
    return c


def main():
    a = argparse.ArgumentParser()
    a.add_argument("out")
    a.add_argument("--updates", type=int)
    a.add_argument("--no-checkpoints", action="store_true")
    a.add_argument("--unguided", action="store_true")
    a.add_argument("--decoder", choices=["faithful", "nll"], default="faithful")
    args = a.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    assert TR.file_sha256(M05.PLACE) == PLACE_SHA256, "placement file differs from M0.5's"
    cfg = TR.load_cfg()
    cfg["heads"]["pixel_decoder"]["mean_loss"] = args.decoder
    if not args.unguided:
        cfg["p1"]["guided"] = {"place_sha256": PLACE_SHA256, "map_sha256": TR.file_sha256(M05.PM.HERE / "map.json"),
                               "guided_streams_sha256": TR.file_sha256(Path(FT2.__file__)), "rule": "m05 Q=1"}
    control = torch.load(CONTROL, weights_only=False, map_location="cpu")
    assert declared(cfg) == declared(control["cfg"]) and control["world_seed"] == WORLD_SEED, \
        "the recipe differs from P1 world 1's beyond the declared changes"
    prov = TR.run_provenance(cfg)
    ident = {"world_seed": WORLD_SEED, "guided": not args.unguided, "decoder": args.decoder,
             "script_sha256": TR.file_sha256(Path(__file__)), "cfg_guided": cfg["p1"].get("guided"),
             "recipe": {k: v for k, v in prov.items() if k != "git_commit"},
             "control": {"path": str(CONTROL), "sha256": TR.file_sha256(CONTROL)}}
    f = out / "identity.json"
    if f.exists():
        old = json.loads(f.read_text())
        assert {k: v for k, v in old.items() if k != "git_commit"} == ident, "resume with a different identity"
    else:
        assert not (out / "latest.pt").exists(), "a checkpoint without an identity file: refusing to adopt it"
        f.write_text(json.dumps({**ident, "git_commit": prov.get("git_commit")}, indent=1))
    D.Streams = functools.partial(FT2.GuidedStreams, place=None if args.unguided else np.load(M05.PLACE))
    TR.p1(cfg, WORLD_SEED, out, args.updates, None, None, not args.no_checkpoints)


if __name__ == "__main__":
    main()
