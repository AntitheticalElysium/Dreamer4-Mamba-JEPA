"""Guided P1 from scratch (PLAN.md): saccade.train.p1 itself, with saccade.data.Streams replaced by v2's GuidedStreams
bound to M0.5's Q = 1 placement (the prior is drawn first on every frame, so every random draw matches P1 world 1);
nothing else differs. An identity file is written at the first start and asserted on resume.
Usage (project venv, GPU): python -m artifacts.experiments.20261030_p1_guided.train_guided OUT_DIR [--updates N]
  [--no-checkpoints] [--unguided]   (--unguided: placement disabled, for the match check against P1 world 1)
"""
import argparse, functools, importlib, json, os
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")          # in case a helper imports craftax: JAX would preallocate the GPU
import numpy as np

from saccade import data as D, train as TR

FT2 = importlib.import_module("artifacts.experiments.20261030_fovea_placement.finetune")
M05 = importlib.import_module("artifacts.experiments.20261030_m05_guided_fovea.finetune")
PLACE_SHA256 = "0bd48526c8116afba58af702964bfc78655e3e770f806eb0e1875f035277e99b"   # M0.5's identity (log 646af765)
WORLD_SEED = 1


def main():
    a = argparse.ArgumentParser()
    a.add_argument("out")
    a.add_argument("--updates", type=int)
    a.add_argument("--no-checkpoints", action="store_true")
    a.add_argument("--unguided", action="store_true")
    args = a.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    cfg = TR.load_cfg()
    assert TR.file_sha256(M05.PLACE) == PLACE_SHA256, "placement file differs from M0.5's"
    prov = TR.run_provenance(cfg)
    ident = {"world_seed": WORLD_SEED, "guided": not args.unguided, "script_sha256": TR.file_sha256(Path(__file__)),
             "guided_streams_sha256": TR.file_sha256(Path(FT2.__file__)), "place_sha256": PLACE_SHA256,
             "map_sha256": TR.file_sha256(M05.PM.HERE / "map.json"),
             "recipe": {k: v for k, v in prov.items() if k != "git_commit"}}
    f = out / "identity.json"
    if f.exists():
        old = json.loads(f.read_text())
        assert {k: v for k, v in old.items() if k != "git_commit"} == ident, "resume with a different identity"
    else:
        f.write_text(json.dumps({**ident, "git_commit": prov.get("git_commit")}, indent=1))
    D.Streams = functools.partial(FT2.GuidedStreams, place=None if args.unguided else np.load(M05.PLACE))
    TR.p1(cfg, WORLD_SEED, out, args.updates, None, None, not args.no_checkpoints)


if __name__ == "__main__":
    main()
