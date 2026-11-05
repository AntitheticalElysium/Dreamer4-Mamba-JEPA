"""P1-F3 one-shot worker (PLAN.md "Run"): F2.1's worker unchanged (../20261104_p1f21/run.py: main) with this launcher,
gate checks.json, then readout.py on the selected checkpoint. Launch only on the user's approval after review."""
import os
os.environ.setdefault('JAX_PLATFORMS', 'cpu')
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
import importlib
from pathlib import Path
L = importlib.import_module('artifacts.experiments.20261105_p1f3.train_p1f3')
W = importlib.import_module('artifacts.experiments.20261104_p1f21.run')

if __name__ == '__main__':
    W.main(L=L, here=Path(__file__).resolve().parent, module='artifacts.experiments.20261105_p1f3.train_p1f3',
           gates=('checks.json',), evaluation_module='artifacts.experiments.20261105_p1f3.readout')
