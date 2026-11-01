"""Overnight loss test (Codex's ../20261031_grounding_loss_review/OVERNIGHT_PROPOSAL.md; user, Codex and Claude agreed
10-31): P1-F's recipe exactly (train_guided: guided Q = 1 placement, faithful pixel decoder, world seed 1, same
initialization, streams and RNG), with ONE declared change, heads.pixel_decoder.tsp_mean_loss = scale_weighted_nll
(TSP's mean teacher; fovea reconstruction stays faithful; scale fitting, G1 and everything else unchanged).
Identity, as train_guided: the change is written INTO the config, which the checkpoint stores and validate_resume
compares; the config minus it must equal P1-F's checkpoint config; a checkpoint without an identity file is refused.
--teacher faithful is the disabled path (P1-F's recipe itself): the parity check.
Usage (project venv, GPU): python -m artifacts.experiments.20261101_beta_tsp.train_beta OUT_DIR [--updates N]
  [--no-checkpoints] [--teacher scale_weighted_nll|faithful]
"""
import argparse, copy, functools, importlib, json, os
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")
import numpy as np
import torch

from saccade import data as D, train as TR

TG = importlib.import_module("artifacts.experiments.20261030_p1_guided.train_guided")
P1F = TR.ROOT / "artifacts/experiments/20261030_p1_guided/run/ckpt_10.pt"


def main():
    a = argparse.ArgumentParser()
    a.add_argument("out")
    a.add_argument("--updates", type=int)
    a.add_argument("--no-checkpoints", action="store_true")
    a.add_argument("--teacher", choices=["scale_weighted_nll", "faithful"], default="scale_weighted_nll")
    args = a.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    assert TR.file_sha256(TG.M05.PLACE) == TG.PLACE_SHA256, "placement file differs from M0.5's"
    cfg = TR.load_cfg()
    cfg["heads"]["pixel_decoder"]["mean_loss"] = "faithful"
    cfg["p1"]["guided"] = {"place_sha256": TG.PLACE_SHA256, "map_sha256": TR.file_sha256(TG.M05.PM.HERE / "map.json"),
                           "guided_streams_sha256": TR.file_sha256(Path(TG.FT2.__file__)), "rule": "m05 Q=1"}
    control = torch.load(P1F, weights_only=False, map_location="cpu")
    assert cfg == control["cfg"] and control["world_seed"] == TG.WORLD_SEED, "the base recipe differs from P1-F's"
    if args.teacher != "faithful":
        cfg["heads"]["pixel_decoder"]["tsp_mean_loss"] = args.teacher
    prov = TR.run_provenance(cfg)
    ident = {"world_seed": TG.WORLD_SEED, "teacher": args.teacher, "script_sha256": TR.file_sha256(Path(__file__)),
             "guided_script_sha256": TR.file_sha256(Path(TG.__file__)), "cfg_pixel_decoder": cfg["heads"]["pixel_decoder"],
             "cfg_guided": cfg["p1"]["guided"], "recipe": {k: v for k, v in prov.items() if k != "git_commit"},
             "control": {"path": str(P1F), "sha256": TR.file_sha256(P1F), "provenance": control["provenance"]}}
    del control
    f = out / "identity.json"
    if f.exists():
        old = json.loads(f.read_text())
        assert {k: v for k, v in old.items() if k != "git_commit"} == ident, "resume with a different identity"
    else:
        assert not (out / "latest.pt").exists(), "a checkpoint without an identity file: refusing to adopt it"
        f.write_text(json.dumps({**ident, "git_commit": prov.get("git_commit")}, indent=1))
    D.Streams = functools.partial(TG.FT2.GuidedStreams, place=np.load(TG.M05.PLACE))
    TR.p1(cfg, TG.WORLD_SEED, out, args.updates, None, None, not args.no_checkpoints)


if __name__ == "__main__":
    main()
