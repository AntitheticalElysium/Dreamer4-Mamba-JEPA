"""The pinned base-LeWM predictor package, imported rather than reimplemented.

A comparison backend only. Mamba remains the architecture under study; this exists so a
faithful upstream predictor can be measured on the same task, which is what makes our own
adaptation's mistakes visible.

Nothing here is a copy of the source. `ARPredictor`, `Embedder` and `MLP` are instantiated
from the vendored file itself, loaded under a private module name so it can never collide
with anything on `sys.path`, and only after its bytes are checked against `SOURCES.lock`'s
commit. Reimplementing them would defeat the point of the control.
"""

import hashlib
import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SOURCE = ROOT / "third_party/sources/lucas-maes__le-wm"
# The audited bytes of the classes this backend instantiates, pinned so a vendor bump is a
# loud failure rather than a silently different control.
PINNED = {
    "module.py": "0b258a9e8dc24c29fcb1e8c50a09ec78b8ea85aeb79e21dd8adf712396646620",
    "jepa.py": "41bad7fd21e0f14aea4c9c3d39a9c87037e787746d953ab62cdc0677e938ce96",
    "config/train/model/lewm.yaml": "7be97eaa2c83f809b9ea7a6da5a7d3f3c70502c27537383f54db32335e21e42b",
}
PRIVATE_NAME = "_d4mj_pinned_lewm_module"


def _digest(path: Path) -> str:
    out = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            out.update(block)
    return out.hexdigest()


def source_digests() -> dict[str, str]:
    """Measured digests of every pinned file, for the source manifest and for tests."""
    return {name: _digest(SOURCE / name) for name in PINNED}


def source_module():
    """The vendored module, verified and loaded once under a private name."""
    if PRIVATE_NAME in sys.modules:
        return sys.modules[PRIVATE_NAME]
    for name, expected in PINNED.items():
        actual = _digest(SOURCE / name)
        if actual != expected:
            raise ValueError(f"lewm_transformer: pinned {name} differs from the audited bytes")
    path = SOURCE / "module.py"
    spec = importlib.util.spec_from_file_location(PRIVATE_NAME, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[PRIVATE_NAME] = module
    spec.loader.exec_module(module)
    return module


def build_package(*, width=192, depth=6, heads=16, head_dim=64, mlp_dim=2048, context=3,
                  n_actions=17, dropout=0.1, embedding_dropout=0.0,
                  action_smoothed_dim=10, action_mlp_scale=4):
    """The three pinned modules, with the constructor arguments from `lewm.yaml`.

    Returned separately rather than wrapped: the world owns how they are called, and the
    oracle test needs to compare these exact objects against its own instantiation.
    """
    import torch

    source = source_module()
    predictor = source.ARPredictor(num_frames=context, depth=depth, heads=heads,
                                   mlp_dim=mlp_dim, input_dim=width, hidden_dim=width,
                                   output_dim=width, dim_head=head_dim, dropout=dropout,
                                   emb_dropout=embedding_dropout)
    action_encoder = source.Embedder(input_dim=n_actions, smoothed_dim=action_smoothed_dim,
                                     emb_dim=width, mlp_scale=action_mlp_scale)
    projector = source.MLP(input_dim=width, hidden_dim=mlp_dim, output_dim=width,
                           norm_fn=torch.nn.BatchNorm1d)
    return predictor, action_encoder, projector
