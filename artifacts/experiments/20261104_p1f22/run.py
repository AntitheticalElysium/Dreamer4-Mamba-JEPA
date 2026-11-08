"""One-shot P1-F2.2 worker (launch requires user approval after review): F2.1's worker with this arm's launcher and
preflight gate (no corpus change, so no union gate)."""
import importlib
from pathlib import Path

R = importlib.import_module('artifacts.experiments.20261104_p1f21.run')
L = importlib.import_module('artifacts.experiments.20261104_p1f22.train_p1f22')

if __name__ == '__main__':
    R.main(L, Path(__file__).resolve().parent, 'artifacts.experiments.20261104_p1f22.train_p1f22', ('checks.json',))
