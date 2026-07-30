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

# Craftax-Classic replaces the danijar/crafter-via-m3 environment (Craftax
# migration). We pin the installed distribution version and digest the exact
# files whose behaviour we depend on: reward/termination (game_logic), the
# action/achievement/observation constants, and the pixel renderer. The package
# is located via importlib without importing it, so this stays JAX-free and safe
# to call from the torch training process.
CRAFTAX_DISTRIBUTION = "craftax"
CRAFTAX_VERSION = "1.6.1"
CRAFTAX_LICENSE = "MIT"
CRAFTAX_CLASSIC_DIGESTS = {
    "craftax_classic/game_logic.py":
        "e5812a161b485a5edba6da0e34b7e3352550fe29ed7d0c8f66c8071ecac20755",
    "craftax_classic/constants.py":
        "5b00ec29b51f7d011bb01c98aa74e5fd6b8a7cee6ab717f61dc59d6407f6baa4",
    "craftax_classic/renderer.py":
        "e415a83a2ce6d859d960be3e2d591b2c347b77d6e91275981b22dd769390ba13",
    # Also executed, and previously unpinned: the env factory every adapter
    # calls, and the two env classes it can return. `game_logic` alone does not
    # cover the observation/step wiring we actually run.
    "craftax_env.py":
        "f74d828dbc9802984e026ace29293527dfd3901cc945d3c7792d199dc92affa3",
    "craftax_classic/envs/craftax_pixels_env.py":
        "37afb876d3677472a02cfa722a65737b23545098fe73ba5c3ad87d160ef64223",
    "craftax_classic/envs/craftax_symbolic_env.py":
        "ada5830c7a5af768ea1bffc7ad962300b11a6092e2943c4ecef7f1a9afbfcb65",
}

# Whole-package digest of the installed `mamba_ssm`. `verify_installed_mamba2`
# hashes only `modules/mamba2.py`, but that file imports and calls a further
# operator tree -- `ops/triton/ssd_combined.py` (`mamba_chunk_scan_combined`,
# the active `use_mem_eff_path=False` path), `ops/triton/layernorm_gated.py`,
# `ops/triton/selective_state_update.py`, `distributed/*` -- none of which were
# covered. Drift in any of them would change our numerics without failing any
# check. Recorded under its own source name so legacy checkpoints, which pin
# only `mamba2.py`, stay verifiable unchanged.
MAMBA_SSM_TREE_SHA256 = (
    "3633eb0755da1525f753684a7591285f1780010ae18f728ae2389b86c41ea830"
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


def _tree_sha256(root: Path, pattern: str = "*.py") -> str:
    """Order-independent digest of every matching file under ``root``."""
    digest = hashlib.sha256()
    for path in sorted(root.rglob(pattern)):
        digest.update(str(path.relative_to(root)).encode())
        digest.update(b"\0")
        digest.update(hashlib.sha256(path.read_bytes()).digest())
        digest.update(b"\0")
    return digest.hexdigest()


def verify_installed_mamba_tree() -> str:
    """Require the WHOLE installed ``mamba_ssm`` package to match its pin.

    `verify_installed_mamba2` covers one file; the kernels it dispatches to live
    in sibling modules that were never checked. See ``MAMBA_SSM_TREE_SHA256``.
    """
    import importlib.util

    spec = importlib.util.find_spec("mamba_ssm")
    if spec is None or not spec.origin:
        raise SourceDriftError("mamba_ssm is not installed")
    actual = _tree_sha256(Path(spec.origin).parent)
    if actual != MAMBA_SSM_TREE_SHA256:
        raise SourceDriftError(
            f"installed mamba_ssm tree digest drift: "
            f"{actual} != {MAMBA_SSM_TREE_SHA256}"
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


def _craftax_package_root() -> Path:
    """Locate the installed craftax package WITHOUT importing it (no JAX)."""
    import importlib.util

    spec = importlib.util.find_spec("craftax")
    if spec is None or not spec.origin:
        raise SourceDriftError("craftax is not installed")
    return Path(spec.origin).parent


def verify_installed_craftax() -> dict[str, str]:
    """Require the installed Craftax-Classic to match the pinned version+digests.

    JAX-free: uses importlib metadata and file hashing only, so it is safe to
    call from the torch training process.
    """
    import importlib.metadata as metadata

    try:
        version = metadata.version(CRAFTAX_DISTRIBUTION)
    except metadata.PackageNotFoundError as exc:
        raise SourceDriftError("craftax is not installed") from exc
    if version != CRAFTAX_VERSION:
        raise SourceDriftError(
            f"craftax version drift: {version} != {CRAFTAX_VERSION}"
        )
    root = _craftax_package_root()
    digests: dict[str, str] = {"version": version}
    for relpath, expected in CRAFTAX_CLASSIC_DIGESTS.items():
        path = root / relpath
        if not path.is_file():
            raise SourceDriftError(f"missing pinned craftax file: {path}")
        actual = file_sha256(path)
        if actual != expected:
            raise SourceDriftError(
                f"craftax {relpath} digest drift: {actual} != {expected}"
            )
        digests[relpath] = actual
    return digests


def craftax_source_report() -> dict[str, str]:
    """Craftax provenance, kept separate from ``source_report`` so the core
    checkpoint provenance contract does not change for non-Craftax runs."""
    return {
        "distribution": CRAFTAX_DISTRIBUTION,
        "license": CRAFTAX_LICENSE,
        **verify_installed_craftax(),
    }


def _load_isolated(identity: SourceIdentity, module_name: str, *, is_package=False):
    """Execute one verified file under an isolated module name."""
    existing = sys.modules.get(module_name)
    if existing is not None:
        return existing
    spec = importlib.util.spec_from_file_location(module_name, identity.path)
    if spec is None or spec.loader is None:
        raise SourceDriftError(f"cannot create import spec for {identity.path}")
    if is_package:
        spec.submodule_search_locations = [str(identity.path.parent)]
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(module_name, None)
        raise
    return module


@lru_cache(maxsize=1)
def load_lejepa_sigreg():
    """Return the pinned LeJEPA ``(SlicingUnivariateTest, EppsPulley)`` classes.

    SIGReg is used unchanged from ``rbalestr-lab/lejepa`` commit
    ``c293d291``: the sliced random-projection multivariate test
    (``lejepa/multivariate/slicing.py``) with the Epps-Pulley univariate
    normality test (``lejepa/univariate/epps_pulley.py``).

    The three files are executed under ISOLATED module names rather than
    imported through ``lejepa.*`` package paths. Importing by package path first
    executes ``lejepa/__init__.py``, ``lejepa/univariate/__init__.py`` and
    ``lejepa/multivariate/__init__.py``, which between them pull in 17 further
    modules (``bhep``, ``hz``, ``hv``, ``comb``, ``anderson_darling``,
    ``shapiro_wilk``, ``watson``, ``jarque_bera``, ...). None of those are
    digest-pinned, so the executed set was strictly larger than the verified
    set and the "digest-pinned SIGReg" claim did not hold for what actually
    ran. Isolated loading makes executed == verified: these three files import
    nothing from LeJEPA except ``epps_pulley -> .base``, which is pre-registered
    below so the relative import resolves to the verified copy.
    """
    for identity in (LEJEPA_SLICING, LEJEPA_EPPS, LEJEPA_BASE):
        verify_source(identity)
    root = "d4_mamba_jepa._pinned_lejepa"
    # Synthetic parent packages: `epps_pulley` does `from .base import ...`, so
    # `<root>.univariate` must exist in sys.modules for that relative import to
    # resolve to our verified `base`, and to nothing else.
    for package in (root, f"{root}.univariate", f"{root}.multivariate"):
        if package not in sys.modules:
            spec = importlib.util.spec_from_loader(package, loader=None,
                                                   is_package=True)
            module = importlib.util.module_from_spec(spec)
            module.__path__ = []  # no filesystem search: nothing else is loadable
            sys.modules[package] = module
    _load_isolated(LEJEPA_BASE, f"{root}.univariate.base")
    epps = _load_isolated(LEJEPA_EPPS, f"{root}.univariate.epps_pulley")
    slicing = _load_isolated(LEJEPA_SLICING, f"{root}.multivariate.slicing")
    return slicing.SlicingUnivariateTest, epps.EppsPulley


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


def _report_mmbench2() -> dict[str, str]:
    return {
        "path": str(MMBENCH2_MODEL.path),
        "commit": MMBENCH2_MODEL.commit,
        "sha256": verify_source(MMBENCH2_MODEL),
        "license": MMBENCH2_MODEL.license,
    }


def _report_mamba2() -> dict[str, str]:
    return {
        "path": str(MAMBA2_SOURCE.path),
        "commit": MAMBA2_SOURCE.commit,
        "sha256": verify_installed_mamba2(),
        "license": MAMBA2_SOURCE.license,
    }


def _report_cartpole() -> dict[str, str]:
    return {
        "path": str(GYMNASIUM_CARTPOLE.path),
        "commit": GYMNASIUM_CARTPOLE.commit,
        "sha256": verify_installed_cartpole(),
        "license": GYMNASIUM_CARTPOLE.license,
    }


# Every source name that can appear in a recorded provenance block, and the
# callable that recomputes+verifies it. `verify_recorded_sources` recomputes only
# the names a checkpoint actually recorded, so this registry is also the set of
# names a stored payload is allowed to contain.
_SOURCE_REPORTERS = {
    "mmbench2_model": _report_mmbench2,
    "mamba2": _report_mamba2,
    "mamba_ssm_tree": lambda: {
        "package": "mamba_ssm",
        "commit": MAMBA2_SOURCE.commit,
        "tree_sha256": verify_installed_mamba_tree(),
        "license": MAMBA2_SOURCE.license,
    },
    "gymnasium_cartpole": _report_cartpole,
    "craftax": craftax_source_report,
    "lejepa": lejepa_source_report,
}

# Provenance blocks written before schema versioning carried exactly these three
# names and no schema key. They are the ONLY name set a schema-less payload may
# present; anything else is an omitted or tampered block, not a legacy one.
LEGACY_SOURCE_NAMES = frozenset(
    {"mmbench2_model", "mamba2", "gymnasium_cartpole"}
)
PROVENANCE_SCHEMA = 2


def source_names_for(cfg) -> tuple[str, ...]:
    """The sources the CODE of a world with this config imports and executes.

    Deliberately excludes the environment. ``D4LiteConfig`` carries no
    environment identity -- it has ``n_actions`` and ``image_size``, neither of
    which names a simulator -- so inferring one here would tag `D4LiteConfig(
    n_actions=2)` and every tokenizer checkpoint as Craftax on no evidence.
    Callers that KNOW their environment pass it to ``source_report`` as an
    explicit ``environment_sources`` entry.

    Reporting is config-conditional because an unconditional report
    over-constrained checkpoints: a transformer-only world could not be saved or
    loaded without an installed, byte-matching Mamba-2 and Gymnasium CartPole,
    neither of which it uses.
    """
    names = ["mmbench2_model"]
    if getattr(cfg, "temporal_backend", None) == "mamba2":
        names += ["mamba2", "mamba_ssm_tree"]
    if (
        getattr(cfg, "representation_objective", None) == "jepa"
        and getattr(cfg, "jepa_anticollapse", None) == "sigreg"
    ):
        names.append("lejepa")
    return tuple(names)


def source_report(cfg=None, *, environment_sources=()) -> dict[str, dict]:
    """Verified provenance for the sources in scope.

    ``cfg=None`` keeps the historical three-source report (MMBench2, Mamba-2,
    Gymnasium CartPole) so existing callers are unaffected. Passing a
    ``D4LiteConfig`` selects the code dependencies of that config
    (``source_names_for``); ``environment_sources`` adds the simulator the
    caller knows it used, e.g. ``("craftax",)``.
    """
    if cfg is None:
        names: tuple[str, ...] = ("mmbench2_model", "mamba2", "gymnasium_cartpole")
    else:
        names = source_names_for(cfg) + tuple(environment_sources)
    unknown = sorted(set(names) - set(_SOURCE_REPORTERS))
    if unknown:
        raise SourceDriftError(f"unknown source names requested: {unknown}")
    return {name: _SOURCE_REPORTERS[name]() for name in dict.fromkeys(names)}


def verify_recorded_sources(
    stored: dict[str, dict], *, schema: int | None = None, required=()
) -> None:
    """Re-verify a recorded provenance block.

    Two failure modes have to be closed at once.

    * Comparing the block against a freshly computed *whole* report makes the
      check depend on what the CURRENT code happens to report, so adding a
      source name retroactively invalidates every existing checkpoint.
    * Verifying only the names present lets an EMPTY or truncated block pass --
      provenance that fails open. A payload claiming no sources is not a payload
      with no source drift.

    The schema key separates the two populations. A schema-less block is a
    pre-versioning checkpoint and must present exactly ``LEGACY_SOURCE_NAMES``.
    A ``PROVENANCE_SCHEMA`` block must cover everything ``required`` (the
    caller's ``source_names_for(config)``) and may carry more, e.g. the
    environment. Either way every recorded name is recomputed and compared.
    """
    if not isinstance(stored, dict):
        raise SourceDriftError("recorded provenance block is not a mapping")
    unknown = sorted(set(stored) - set(_SOURCE_REPORTERS))
    if unknown:
        raise SourceDriftError(f"unknown recorded source names: {unknown}")

    if schema is None:
        if set(stored) != LEGACY_SOURCE_NAMES:
            raise SourceDriftError(
                "provenance block carries no schema version, so it must record "
                f"exactly {sorted(LEGACY_SOURCE_NAMES)}; got {sorted(stored)}"
            )
    elif schema == PROVENANCE_SCHEMA:
        missing = sorted(set(required) - set(stored))
        if missing:
            raise SourceDriftError(
                f"provenance block omits required sources {missing}; this "
                "config cannot be verified from what was recorded"
            )
    else:
        raise SourceDriftError(f"unsupported provenance schema {schema!r}")

    for name, recorded in stored.items():
        current = _SOURCE_REPORTERS[name]()
        if current != recorded:
            raise SourceDriftError(
                f"recorded source {name!r} drifted: {current} != {recorded}"
            )
