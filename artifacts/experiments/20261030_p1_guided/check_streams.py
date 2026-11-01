"""Check 1 of PLAN.md (CPU): D.Streams vs GuidedStreams on P1's training split, world seed 1, 16 streams, prefix 8,
records only (gather replaced by the identity): with no placement the records are identical; with M0.5's placement
they differ only in the fixation of placed frames. Chunks of 64 steps, then of 256 (P1's two phases).
Usage: python -m artifacts.experiments.20261030_p1_guided.check_streams
"""
import importlib, json
from pathlib import Path

import numpy as np

from saccade import data as D, train as TR

FT2 = importlib.import_module("artifacts.experiments.20261030_fovea_placement.finetune")
M05 = importlib.import_module("artifacts.experiments.20261030_m05_guided_fovea.finetune")
HERE = Path(__file__).resolve().parent


def main():
    cfg = TR.load_cfg()
    split = D.load_split(cfg)
    offsets = D.frame_offsets(split["train"])
    d, P = cfg["data"], cfg["data"]["context_prefix"]
    D.gather = lambda eps, recs, offsets=None: recs                       # records only, no frames
    place = np.load(M05.PLACE)
    make = lambda cls, **kw: cls(split["train"], d["streams"], P, seed=1, targets_offsets=offsets, **kw)
    base, unguided, guided = make(D.Streams), make(FT2.GuidedStreams, place=None), make(FT2.GuidedStreams, place=place)
    res = {"chunks": 0, "frames": 0, "unguided_identical": True, "guided_other_fields_identical": True,
           "placed_frames": 0, "placed_fixation_equals_rule": True, "unplaced_identical": True}
    for steps, n in ((d["chunk_steps"][0], 300), (d["chunk_steps"][1], 60)):
        for _ in range(n):
            rb, ru, rg = base.next(steps), unguided.next(steps), guided.next(steps)
            res["chunks"] += 1
            res["unguided_identical"] &= rb == ru
            for sb, sg in zip(rb, rg):
                for a, b in zip(sb[P:], sg[P:]):                          # the chunk's new records
                    res["frames"] += 1
                    res["guided_other_fields_identical"] &= a[:3] == b[:3]
                    p = place[offsets[a[0]] + a[2]]
                    if p[0] >= 0:
                        res["placed_frames"] += 1
                        res["placed_fixation_equals_rule"] &= tuple(b[3:]) == (int(p[0]), int(p[1]))
                    else:
                        res["unplaced_identical"] &= a == b
    res["placed_share"] = res["placed_frames"] / res["frames"]
    (HERE / "check_streams.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))
    assert res["unguided_identical"] and res["guided_other_fields_identical"] and res["placed_fixation_equals_rule"] \
        and res["unplaced_identical"]


if __name__ == "__main__":
    main()
