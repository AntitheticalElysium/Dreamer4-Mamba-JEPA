import hashlib
from pathlib import Path

from .config import Config

ROOT = Path(__file__).resolve().parent.parent / "third_party"

PINNED = {
    "dreamer4_paper": "papers/2509.24527v1.pdf",
    "mamba2_module": "sources/state-spaces__mamba/mamba_ssm/modules/mamba2.py",
    "lejepa_minimal": "sources/rbalestr-lab__lejepa/MINIMAL.md",
    "vjepa2_ac_train": "vjepa2/app/vjepa_droid/train.py",
    "dreamerv3_agent": "sources/danijar__dreamerv3/dreamerv3/agent.py",
    "mop_jepa_paper": "papers/2607.05238v1.pdf",
}
"""Sources whose bytes a decision in spec/DECISIONS.md rests on.

Deliberately not every file we read: a manifest that lists everything consulted
stops being checked. Reproductions that only corroborate are cited in the spec
but not pinned here.
"""


def source_digests(config: Config) -> dict[str, str]:
    names = list(PINNED)
    if config.time_mixer != "mamba":
        names.remove("mamba2_module")
    return {name: _digest(ROOT / PINNED[name]) for name in names}


def verify_sources(recorded: dict[str, str], config: Config) -> None:
    expected = source_digests(config)
    missing = set(expected) - set(recorded)
    if missing:
        raise ValueError(f"checkpoint omits required sources: {sorted(missing)}")
    drifted = [name for name, value in expected.items() if recorded[name] != value]
    if drifted:
        raise ValueError(f"pinned sources changed since this checkpoint: {drifted}")


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
