"""P1-MultiStep-Mamba one-shot worker: preflight gates -> training -> fixed checkpoint rule -> evaluation."""
import os
os.environ.setdefault('JAX_PLATFORMS', 'cpu')
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
import importlib
from pathlib import Path

L = importlib.import_module('artifacts.experiments.20261111_p1_multistep_mamba.train_multistep')
W = importlib.import_module('artifacts.experiments.20261104_p1f21.run')

if __name__ == '__main__':
    W.main(L=L, here=Path(__file__).resolve().parent,
           module='artifacts.experiments.20261111_p1_multistep_mamba.train_multistep',
           gates=('checks.json',),
           evaluation_module='artifacts.experiments.20261111_p1_multistep_mamba.readout')
