"""P1-F1.5 (PLAN.md): saccade.train.p1 itself, from scratch, with P1 world 1's streams (fixations uniform over the 49
windows, saccade.data.fixation_prior; no hand placement, no learned chooser), P1-F's faithful pixel decoder and
decision 1's factorised denoiser (learned-fixation TEACHING 10 / 10a). The three declared changes are written INTO the
config, which the checkpoint stores and saccade.train.validate_resume compares on every restore; on first launch the
config minus them must equal P1 world 1's checkpoint config; a directory holding a checkpoint but no identity file is
refused. mamba_ssm's deterministic mode is required (its autotuning differs per process: a resume or re-run would
otherwise not continue the same computation).
Usage (project venv, GPU): MAMBA_DETERMINISTIC=1 python -m artifacts.experiments.20261101_p1f15.train_p1f15 OUT_DIR
  [--updates N] [--no-checkpoints]
"""
import argparse, copy, json, os
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")          # in case a helper imports craftax: JAX would preallocate the GPU
import torch

from saccade import train as TR

CONTROL = TR.ROOT / "artifacts/saccade_m0/p1_world1/ckpt_10.pt"
WORLD_SEED = 1


def recipe(cfg):
    """P1 world 1's config with the three declared changes (m0.yaml documents the two decision-1 keys)."""
    cfg["heads"]["pixel_decoder"]["mean_loss"] = "faithful"
    cfg["denoiser"]["factorised"] = True
    cfg["edm"]["sampler"]["glimpse_conditioning_sigma"] = 0.1
    return cfg


def declared(cfg):
    """The config without the three declared changes."""
    c = copy.deepcopy(cfg)
    c["heads"]["pixel_decoder"].pop("mean_loss", None)
    c["denoiser"].pop("factorised", None)
    c["edm"]["sampler"].pop("glimpse_conditioning_sigma", None)
    return c


def main():
    a = argparse.ArgumentParser()
    a.add_argument("out")
    a.add_argument("--updates", type=int)
    a.add_argument("--no-checkpoints", action="store_true")
    args = a.parse_args()
    assert os.environ.get("MAMBA_DETERMINISTIC") == "1", "set MAMBA_DETERMINISTIC=1"
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    cfg = recipe(TR.load_cfg())
    control = torch.load(CONTROL, weights_only=False, map_location="cpu")
    assert declared(cfg) == declared(control["cfg"]) and control["world_seed"] == WORLD_SEED, \
        "the recipe differs from P1 world 1's beyond the declared changes"
    prov = TR.run_provenance(cfg)
    ident = {"world_seed": WORLD_SEED, "recipe": "P1 world 1 + faithful decoder + factorised denoiser, uniform fixations",
             "script_sha256": TR.file_sha256(Path(__file__)), "mamba_deterministic": True,
             "provenance": {k: v for k, v in prov.items() if k != "git_commit"},
             "control": {"path": str(CONTROL), "sha256": TR.file_sha256(CONTROL)}}
    f = out / "identity.json"
    if f.exists():
        old = json.loads(f.read_text())
        assert {k: v for k, v in old.items() if k != "git_commit"} == ident, "resume with a different identity"
    else:
        assert not (out / "latest.pt").exists(), "a checkpoint without an identity file: refusing to adopt it"
        f.write_text(json.dumps({**ident, "git_commit": prov.get("git_commit")}, indent=1))
    TR.p1(cfg, WORLD_SEED, out, args.updates, None, None, not args.no_checkpoints)


if __name__ == "__main__":
    main()
