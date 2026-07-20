"""Strict loaders for unchanged, pinned upstream source files.

The upstream MMBench2 repository is an executable-style checkout with flat
imports rather than an installable Python package. Loading only ``model.py``
through an isolated module name keeps its classes unchanged and avoids adding
the generic upstream ``src`` directory to global ``sys.path``.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import hashlib
import importlib.util
from pathlib import Path
import sys
from types import ModuleType


REPO_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class SourceIdentity:
    name: str
    path: Path
    commit: str
    sha256: str
    license: str


MMBENCH2_MODEL = SourceIdentity(
    name="nicklashansen/mmbench2:src/model.py",
    path=REPO_ROOT / "third_party/sources/nicklashansen__mmbench2/src/model.py",
    commit="3dda6ea5bc60382ad9e1dcd1c6c3af67d69326a9",
    sha256="40f0c763e3e2a62c1dee2786cc6faffb7b08c8145068d8cf7d853ae89c893510",
    license="MIT",
)

MAMBA2_SOURCE = SourceIdentity(
    name="state-spaces/mamba:mamba_ssm/modules/mamba2.py",
    path=REPO_ROOT
    / "third_party/sources/state-spaces__mamba/mamba_ssm/modules/mamba2.py",
    commit="f577286d052741c35d39cd43bdc3fad27120f22c",
    sha256="605e4439ff0baec8d8acaf4a191d9f0570eea9900065a065909124c472b08707",
    license="Apache-2.0",
)

COMPACT_DATA = SourceIdentity(
    name="local audited replay adapter:m3_hjwm_compact/data.py",
    path=REPO_ROOT / "m3_hjwm_compact/data.py",
    commit="d1ccfa1",
    sha256="861cf76325cc9e5473e6fac837c1206657afc02bc4e32121e6d552055fb51929",
    license="workspace-local",
)

CRAFTER_CANONICAL = SourceIdentity(
    name="local audited Crafter canonicalizer",
    path=REPO_ROOT / "m3_hjwm_compact/verification/crafter_canonical.py",
    commit="d1ccfa1",
    sha256="a10083a7eb990f65b53955b7e79f5c2491572be8c3961d4717b3a66b309bc2ea",
    license="workspace-local",
)


class SourceDriftError(RuntimeError):
    """A primary source no longer matches the registered implementation."""


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_source(identity: SourceIdentity) -> str:
    if not identity.path.is_file():
        raise SourceDriftError(f"missing pinned source: {identity.path}")
    actual = file_sha256(identity.path)
    if actual != identity.sha256:
        raise SourceDriftError(
            f"{identity.name} digest drift: {actual} != {identity.sha256}"
        )
    return actual


@lru_cache(maxsize=1)
def load_mmbench2_model() -> ModuleType:
    """Load the exact registered MMBench2 model file under an isolated name."""
    verify_source(MMBENCH2_MODEL)
    module_name = "d4_mamba_jepa._pinned_mmbench2_model"
    existing = sys.modules.get(module_name)
    if existing is not None:
        return existing
    spec = importlib.util.spec_from_file_location(module_name, MMBENCH2_MODEL.path)
    if spec is None or spec.loader is None:
        raise SourceDriftError(f"cannot create import spec for {MMBENCH2_MODEL.path}")
    module = importlib.util.module_from_spec(spec)
    # Dataclass decoration consults sys.modules while the module executes.
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(module_name, None)
        raise
    return module


def verify_installed_mamba2() -> str:
    """Require the imported Mamba-2 implementation to match the source pin."""
    import inspect
    from mamba_ssm.modules.mamba2 import Mamba2

    verify_source(MAMBA2_SOURCE)
    installed = Path(inspect.getsourcefile(Mamba2) or "")
    if not installed.is_file():
        raise SourceDriftError("cannot locate installed Mamba2 source")
    actual = file_sha256(installed)
    if actual != MAMBA2_SOURCE.sha256:
        raise SourceDriftError(
            f"installed Mamba2 digest drift: {actual} != {MAMBA2_SOURCE.sha256}"
        )
    return actual


def source_report() -> dict[str, dict[str, str]]:
    return {
        "mmbench2_model": {
            "path": str(MMBENCH2_MODEL.path),
            "commit": MMBENCH2_MODEL.commit,
            "sha256": verify_source(MMBENCH2_MODEL),
            "license": MMBENCH2_MODEL.license,
        },
        "mamba2": {
            "path": str(MAMBA2_SOURCE.path),
            "commit": MAMBA2_SOURCE.commit,
            "sha256": verify_installed_mamba2(),
            "license": MAMBA2_SOURCE.license,
        },
        "local_replay_adapter": {
            "path": str(COMPACT_DATA.path),
            "commit": COMPACT_DATA.commit,
            "sha256": verify_source(COMPACT_DATA),
            "license": COMPACT_DATA.license,
        },
        "crafter_canonicalizer": {
            "path": str(CRAFTER_CANONICAL.path),
            "commit": CRAFTER_CANONICAL.commit,
            "sha256": verify_source(CRAFTER_CANONICAL),
            "license": CRAFTER_CANONICAL.license,
        },
    }
