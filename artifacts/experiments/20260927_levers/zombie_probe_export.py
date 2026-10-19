"""Export teval's zombie ridge readout as its exact affine map (2026-10-09 12:12; E22Z diagnostic: mark zombie cells in training windows,
whose pool has no simulator labels). Probes.zombie(q) = [(q - mu) / sd, 1] @ w is affine in q: b = f(0), W = f(I) - b. Checked on the
held tokens. -> artifacts/eda/health_chain_v1/zombie_probe_linear.pt {'W': [192, 1], 'b': [1]}
Usage: zombie_probe_export.py
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, 'artifacts/experiments/20260927_levers')
OUT = Path('artifacts/eda/health_chain_v1/zombie_probe_linear.pt')


def main():
    import teval as T
    meta, train_roots, train_seeds = T.split()
    cache = T.build_cache('raw', torch.device('cpu'))
    probes = T.Probes(cache, meta, train_roots, train_seeds)
    b = probes.zombie(torch.zeros(1, 192))[0]
    W = probes.zombie(torch.eye(192)) - b
    x = cache['fut'][:64].float().flatten(0, 2)
    err = float((x @ W + b - probes.zombie(x)).abs().max())
    assert err < 1e-3, err
    torch.save({'W': W, 'b': b, 'check_max_abs_err': err}, OUT)
    print({'W': tuple(W.shape), 'b': float(b), 'check_max_abs_err': err})


if __name__ == '__main__':
    sys.path.insert(0, 'artifacts/experiments/20260926_diagnosis')
    sys.path.insert(0, 'artifacts/experiments/20260921_readout_ladder')
    main()
