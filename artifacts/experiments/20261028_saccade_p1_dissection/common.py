"""Frozen-checkpoint diagnostic utilities, intentionally outside canonical saccade code."""
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from saccade import data as D, model as M, train as TR
from saccade.eval import s1, gates as G

OUT = Path(__file__).resolve().parent
ROOT = TR.ROOT
RUN = ROOT / "artifacts/saccade_m0/p1_world1"
CKPT = RUN / "ckpt_10.pt"
READER = RUN / "s1/results.reader.pt"
LABELS = ROOT / "artifacts/eda/s0"
KINDS = ("background", "zombies", "cows", "skeletons", "arrows")
HUD = ("health", "food", "drink", "energy", "wood", "stone", "coal", "iron", "diamond", "sapling",
       "wood_pickaxe", "stone_pickaxe", "iron_pickaxe", "wood_sword", "stone_sword", "iron_sword")
HUD_CELLS = np.array([63, 64, 65, 66, 68, 69, 70, 71, 72, 67, 73, 74, 75, 76, 77, 78])
CELL_TO_HUD = np.full(81, -1, np.int64)
CELL_TO_HUD[HUD_CELLS] = np.arange(16)


def save(name, value):
    (OUT / name).write_text(json.dumps(value, indent=2, default=lambda x: x.item() if isinstance(x, np.generic) else str(x)) + "\n")


def fingerprints():
    paths = [CKPT, READER, *[ROOT / p for p in ("saccade/model.py", "saccade/train.py", "saccade/data.py",
             "saccade/m0.yaml", "saccade/eval/s1.py")]]
    return {str(p.relative_to(ROOT)): TR.file_sha256(p) for p in paths}


def model_hash(model):
    h = hashlib.sha256()
    for k, v in model.state_dict().items():
        h.update(k.encode()); h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def load_model():
    ck = torch.load(CKPT, map_location="cpu", weights_only=False)
    model = M.M0(ck["cfg"]).cuda().eval().requires_grad_(False)
    model.load_state_dict(ck["model"])
    return model, ck["cfg"]


def labels(split):
    return dict(np.load(LABELS / f"s0_{'train400' if split == 'train' else 'dev'}_labels.npz"))


def half_encoding(enc):
    return [{k: v.half() if k in ("p", "g", "v", "b", "bp") else v for k, v in r.items()} for r in enc]


def episode_rows(L):
    return np.repeat(np.arange(len(L["ep_T"])), L["ep_T"] + 1)


def hud_values(L):
    return np.minimum(np.concatenate([L["s_body"], L["s_inv"]], 1), 9).astype(np.int64)


def grids(L):
    out = G.episode_grids(L)
    for ep in out:
        ep["grid"][:, 3, 4] = 0  # the player sprite covers this location
    return out


def crop_cells(fix):
    f = np.asarray(fix)
    rr, cc = np.meshgrid(np.arange(-1, 2), np.arange(-1, 2), indexing="ij")
    return ((f[..., 0, None] + rr.ravel()) * 9 + f[..., 1, None] + cc.ravel()).astype(np.int64)
