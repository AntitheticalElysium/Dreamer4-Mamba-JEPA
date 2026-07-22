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

GYMNASIUM_CARTPOLE = SourceIdentity(
    name="Farama-Foundation/Gymnasium:CartPole-v1",
    path=REPO_ROOT
    / "third_party/sources/Farama-Foundation__Gymnasium"
    / "gymnasium/envs/classic_control/cartpole.py",
    commit="a923da5d4415a1aa5195d99341069da5e16deed7",
    sha256="b758e3286711a2c44b0817265412c9fab1dce8b1b385e2126bc710ceedd47378",
    license="MIT",
)


LEJEPA_ROOT = REPO_ROOT / "third_party/sources/rbalestr-lab__lejepa"
LEJEPA_COMMIT = "c293d291ca87cd4fddee9d3fffe4e914c7272052"
LEJEPA_SLICING = SourceIdentity(
    name="rbalestr-lab/lejepa:lejepa/multivariate/slicing.py",
    path=LEJEPA_ROOT / "lejepa/multivariate/slicing.py",
    commit=LEJEPA_COMMIT,
    sha256="86c0fe3a714dc945ba3e23ab4093f6ed41966039f9681bd53c733e3ca5dff56b",
    license="see repo",
)
LEJEPA_EPPS = SourceIdentity(
    name="rbalestr-lab/lejepa:lejepa/univariate/epps_pulley.py",
    path=LEJEPA_ROOT / "lejepa/univariate/epps_pulley.py",
    commit=LEJEPA_COMMIT,
    sha256="e6554ee42de27b74d62befb5f353d4d7a4f92c6c1eade25edaa6595a6b593149",
    license="see repo",
)
LEJEPA_BASE = SourceIdentity(
    name="rbalestr-lab/lejepa:lejepa/univariate/base.py",
    path=LEJEPA_ROOT / "lejepa/univariate/base.py",
    commit=LEJEPA_COMMIT,
    sha256="08d4f115990656ea3459dacbba5991622725f9040ebc64ae8d34f4e76299eef6",
    license="see repo",
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


def verify_installed_cartpole() -> str:
    """Require Gymnasium's imported CartPole to match the source pin."""
    import inspect
    from gymnasium.envs.classic_control.cartpole import CartPoleEnv

    verify_source(GYMNASIUM_CARTPOLE)
    installed = Path(inspect.getsourcefile(CartPoleEnv) or "")
    if not installed.is_file():
        raise SourceDriftError("cannot locate installed Gymnasium CartPole source")
    actual = file_sha256(installed)
    if actual != GYMNASIUM_CARTPOLE.sha256:
        raise SourceDriftError(
            f"installed CartPole digest drift: "
            f"{actual} != {GYMNASIUM_CARTPOLE.sha256}"
        )
    return actual


@lru_cache(maxsize=1)
def load_lejepa_sigreg():
    """Return the pinned LeJEPA ``(SlicingUnivariateTest, EppsPulley)`` classes.

    SIGReg is used unchanged from ``rbalestr-lab/lejepa`` commit
    ``c293d291``: the sliced random-projection multivariate test
    (``lejepa/multivariate/slicing.py``) with the Epps-Pulley univariate
    normality test (``lejepa/univariate/epps_pulley.py``). Digests are verified
    before import so any source drift hard-fails.
    """
    for identity in (LEJEPA_SLICING, LEJEPA_EPPS, LEJEPA_BASE):
        verify_source(identity)
    root = str(LEJEPA_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)
    from lejepa.multivariate.slicing import SlicingUnivariateTest
    from lejepa.univariate.epps_pulley import EppsPulley

    return SlicingUnivariateTest, EppsPulley


def lejepa_source_report() -> dict[str, str]:
    """Separate SIGReg provenance (kept out of ``source_report`` so it does not
    change the checkpoint provenance contract for non-SIGReg checkpoints)."""
    return {
        "path": str(LEJEPA_SLICING.path),
        "commit": LEJEPA_SLICING.commit,
        "slicing_sha256": verify_source(LEJEPA_SLICING),
        "epps_pulley_sha256": verify_source(LEJEPA_EPPS),
        "base_sha256": verify_source(LEJEPA_BASE),
        "license": LEJEPA_SLICING.license,
    }


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
        "gymnasium_cartpole": {
            "path": str(GYMNASIUM_CARTPOLE.path),
            "commit": GYMNASIUM_CARTPOLE.commit,
            "sha256": verify_installed_cartpole(),
            "license": GYMNASIUM_CARTPOLE.license,
        },
    }
