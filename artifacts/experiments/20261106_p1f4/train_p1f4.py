"""P1-F4 (PLAN.md): F3's world (F2's recipe, trainer hooks, chooser, corpus, seed and budget) with Dreamer 4's denoiser
training and its dense long-context layers, both as Dreamer 4 specifies them; nothing else changed. From scratch,
world seed 1; no automatic launch.

- Shortcut forcing replaces EDM (and F3's x-space weighting, which its x-space flow loss subsumes): flow matching in
  x-prediction on the finest grid, the bootstrap self-consistency term at coarser steps, the ramp weight, K = 4
  sampling from pure noise, past inputs slightly corrupted (2509.24527 eq. 6-8, section 3.2). Hyperparameters the
  paper leaves open are both public reimplementations' (nicklashansen/dreamer4 b8abafbf, edwhu/dreamer4-jax 8144b940,
  identical): k_max 8, bootstrap on 25% of the batch, from update 5,000; the context level k_max - 1.
- Dense temporal attention in the long-context layers (blocks 4 and 8): Dreamer 4 Table 2's design before its "time
  factorized long context" speed step (FVD 71 -> 91). Decision 1's factorisation is kept (physical queries read
  physical keys only).
The code is canonical behind config keys (saccade/model.py, train.py; absent keys = the previous behaviour, which
reproduces F3's stored readings bitwise)."""
import os
os.environ.setdefault('JAX_PLATFORMS', 'cpu')
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
import argparse
import copy
import importlib
import json
from pathlib import Path
import torch
from saccade import train as TR
F2L = importlib.import_module('artifacts.experiments.20261103_p1f2.train_p1f2')
T = importlib.import_module('artifacts.experiments.20261106_p1f4.trainer')
C = F2L.C
HERE = Path(__file__).resolve().parent
CONTROL = HERE.parent / '20261105_p1f3/run/ckpt_10.pt'
CONTROL_SHA = '9695c6d65b1b5a50d32460f18b9ca4a2d683da7e50745a7e8122b6cb1744d544'
WORLD_SEED = 1
SHORTCUT = {'k_max': 8,                    # reimplementations' default (paper: unstated); K = 4 needs k_max >= 4
            'self_fraction': 0.25,         # reimplementations' bootstrap share of the batch
            'bootstrap_start': 5000,       # reimplementations' warm-up (updates) before the bootstrap term
            'ramp': [0.9, 0.1],            # paper eq. 8: w(tau) = 0.9 tau + 0.1
            'context_signal': 7,           # k_max - 1: "slightly corrupt the past inputs" (both reimplementations' level)
            'init_seed': 1106}             # the two new embeddings' own draws (every other parameter keeps F3's)
MEMORY = {'accumulate_at_64': 2,               # 6 GB card: dense attention adds 0.5 GiB (its keys / values per window
          'accumulate_at_256': 8}              # slot); gradient accumulation, as F3 already did at 256 steps (4 x 4)
SOURCES = {'paper': 'Dreamer 4, arXiv 2509.24527 (third_party/papers/2509.24527v1.pdf)',
           'reimplementations': ['nicklashansen/dreamer4 b8abafbf (train_dynamics.py, model.py)',
                                 'edwhu/dreamer4-jax 8144b940 (scripts/train_dynamics.py, dreamer/sampler.py)']}


def recipe(cfg):
    cfg = F2L.recipe(copy.deepcopy(cfg))
    cfg['denoiser']['temporal_attention'] = 'dense'
    cfg['edm']['objective'] = 'shortcut_forcing'
    cfg['edm']['shortcut'] = copy.deepcopy(SHORTCUT)
    cfg['edm']['sampler']['steps'] = 4                                   # paper: K = 4 sampling steps per frame
    cfg['data'].update(MEMORY)
    return cfg


def check_recipe(cfg):
    """The recipe is F3's checkpoint configuration minus F3's G1 weighting plus exactly these keys (and the memory
    split of each update into microbatches); same seed, corpus, change targets, torch and chooser as F3."""
    assert TR.file_sha256(CONTROL) == CONTROL_SHA, 'F3 control changed'
    ck = torch.load(CONTROL, map_location='cpu', weights_only=False)
    expected = copy.deepcopy(ck['cfg'])
    del expected['edm']['g1_weighting']
    expected['denoiser']['temporal_attention'] = 'dense'
    expected['edm'].update(objective='shortcut_forcing', shortcut=copy.deepcopy(SHORTCUT))
    expected['edm']['sampler']['steps'] = 4
    expected['data'].update(MEMORY)
    assert cfg == expected and ck['world_seed'] == WORLD_SEED, 'changes beyond shortcut forcing and dense attention'
    saved, now = ck['provenance'], TR.run_provenance(cfg)
    assert all(saved[k] == now[k] for k in ('manifest_sha256', 'targets_sha256', 'torch')), 'corpus, targets or torch differ from F3'
    assert ck['fixation_controller']['config'] == C.config(), 'chooser differs from F3'
    return cfg


def identity(cfg, policy):
    files = [Path(__file__), HERE / 'trainer.py', Path(C.__file__), HERE.parent / '20261103_p1f2/chooser.json', Path(C.Q.__file__)]
    return dict(recipe='P1-F3 world with Dreamer 4 shortcut forcing + dense long-context layers; from scratch',
                world_seed=WORLD_SEED, chooser=policy, shortcut=SHORTCUT, memory=MEMORY, sources=SOURCES,
                source_sha256={str(p.relative_to(TR.ROOT)): TR.file_sha256(p) for p in files},
                control=dict(path=str(CONTROL), sha256=CONTROL_SHA), mamba_deterministic=True,
                cuda_alloc_conf=os.environ['PYTORCH_CUDA_ALLOC_CONF'],
                provenance={k: v for k, v in TR.run_provenance(cfg).items() if k != 'git_commit'})


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('out'); ap.add_argument('--updates', type=int)
    ap.add_argument('--no-checkpoints', action='store_true'); args = ap.parse_args()
    assert os.environ.get('MAMBA_DETERMINISTIC') == '1'
    cfg = check_recipe(recipe(TR.load_cfg())); policy = C.config()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    expected = identity(cfg, policy); file = out / 'identity.json'
    if file.exists():
        assert json.loads(file.read_text()) == expected, 'Different launch identity on resume'
    else:
        assert not (out / 'latest.pt').exists(), 'Checkpoint without identity'
        file.write_text(json.dumps(expected, indent=2) + '\n')
    T.p1(cfg, WORLD_SEED, out, updates=args.updates, checkpoints=not args.no_checkpoints, controller_cfg=policy)


if __name__ == '__main__':
    main()
