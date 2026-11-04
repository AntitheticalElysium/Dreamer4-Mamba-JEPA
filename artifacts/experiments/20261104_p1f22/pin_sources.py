"""Seal P1-F2.2's sources after its preflight passed (F2.1's seal). Never starts training."""
import importlib
from pathlib import Path

P = importlib.import_module('artifacts.experiments.20261104_p1f21.pin_sources')

if __name__ == '__main__':
    P.seal(Path(__file__).resolve().parent, ('PLAN.md', 'checks.json'))
