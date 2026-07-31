import sys

from . import gates
from .config import Config

CHECKS = ("alignment", "scan_step_parity", "reset_parity", "firewall", "branch_nonmutation", "recurrent_carry")


def main() -> int:
    """Runs every gate across the Stage-A lattice. Gates come before results:
    an arm that fails one is not a result, it is a bug wearing a number."""
    arms = [
        Config(transition=transition, time_mixer=mixer)
        for transition in ("flow", "direct")
        for mixer in ("attention", "mamba")
    ]
    failures = 0
    for config in arms:
        name = f"{config.transition}-{config.time_mixer}"
        for check in CHECKS:
            try:
                getattr(gates, check)(config)
                print(f"  {name:16s} {check:20s} ok")
            except Exception as error:
                failures += 1
                print(f"  {name:16s} {check:20s} FAIL: {error}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
