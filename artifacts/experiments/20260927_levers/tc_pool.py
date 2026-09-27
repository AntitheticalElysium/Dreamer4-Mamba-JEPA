"""E3 prerequisite. The spatial pool's exact windows re-encoded by the TC joint encoder.

spatial_pool_v1 holds 32,647 six-frame windows (ids = (episode_id, start)) encoded by the Raw H2 encoder:
layer-normed z [N,6,192] and layer-normed patch tokens [N,6,81,192] fp16 (spatial.encode). This writes the same
windows, same order, every non-state field copied, with z and tokens from the TC joint step-10000 encoder
(frozen, eval mode, fp32 forward), so a Raw-vs-TC world comparison differs only in the encoder.
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path("/home/antithetical/EPITA/PERSO/DynamicHorizons-Mamba-JEPA")
sys.path.insert(0, str(ROOT))
from d4mj.data import _sha256  # noqa: E402

SRC = ROOT / "artifacts/eda/spatial_pool_v1"
OUT = ROOT / "artifacts/eda/spatial_pool_tc_v1"
TC = ROOT / "artifacts/lewm_m4_canonical/tc/joint/step-010000.pt"


@torch.no_grad()
def main():
    from d4mj.checkpoint import read_lewm_bundle
    from d4mj.config import config_from_dict
    from d4mj.data import load_joint_corpus
    from d4mj.world_api import ModelBundle
    os.chdir(ROOT)
    device = torch.device("cuda")
    payload = read_lewm_bundle(TC)
    config = config_from_dict(payload["config"])
    bundle = ModelBundle.create(config)
    bundle.encoder.load_state_dict(payload["modules"]["encoder"])
    enc = bundle.encoder.to(device).freeze()
    record = json.loads((ROOT / "artifacts/lewm_m4_canonical/raw/dataset.json").read_text())
    episodes, contract = load_joint_corpus(record["paths"], config)
    if contract != record["contract"]:
        raise SystemExit("the M4 corpus differs from its dataset contract")
    by_id = {e.episode_id: e for e in episodes}
    src = torch.load(SRC / "pool.pt", weights_only=False, mmap=True)
    ids = src["ids"]
    n = len(ids)
    z = torch.empty(n, 6, 192)
    tokens = torch.empty(n, 6, 81, 192, dtype=torch.float16)
    for b in range(0, n, 128):
        chunk = ids[b:b + 128]
        frames = torch.as_tensor(np.stack([np.asarray(by_id[e].observations[s:s + 6]) for e, s in chunk])).to(device)
        zz, _, tt, _, _ = enc._hidden(frames)
        z[b:b + len(chunk)] = F.layer_norm(zz.float(), (192,)).view(len(chunk), 6, 192).cpu()
        tokens[b:b + len(chunk)] = F.layer_norm(tt.float(), (192,)).view(len(chunk), 6, 81, 192).half().cpu()
        if b % 8192 == 0:
            print(json.dumps({"done": b + len(chunk), "of": n}), flush=True)
    pool = {k: v for k, v in src.items() if k not in ("z", "tokens", "meta")}
    pool.update({"z": z, "tokens": tokens,
                 "meta": dict(src["meta"]) | {"encoder": "tc_joint_step_10000", "encoder_sha256": _sha256(TC),
                                              "source_pool_sha256": json.loads((SRC / "pool.json").read_text())["pool_sha256"]}})
    OUT.mkdir(parents=True, exist_ok=True)
    torch.save(pool, OUT / "pool.pt")
    (OUT / "pool.json").write_text(json.dumps({**pool["meta"], "pool_sha256": _sha256(OUT / "pool.pt")}, indent=2) + "\n")
    print(json.dumps({"status": "tc_pool_complete", "windows": n}), flush=True)


if __name__ == "__main__":
    main()
