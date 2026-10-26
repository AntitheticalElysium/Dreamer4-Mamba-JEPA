"""Evaluation harness (saccade/design/TEST_SPEC.md): simulator labels, observable gates, oracle references, probes.

The corpus replay utility stays at artifacts/eda/replay.py, shared with the dated EDA scripts; it is put on the path
here, once, for the modules that replay episodes (labels, oracle, probe). Modules run from the repository root as
`python -m saccade.eval.<module>`. Runtime: the project .venv (JAX on CPU, torch with CUDA for probe); gates (c2st)
and calib also need lightgbm, which the project .venv does not contain (a separate numpy + lightgbm venv was used).
Regression checks of the harness: `.venv/bin/python -m pytest saccade/tests`.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "artifacts" / "eda"))
