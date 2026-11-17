"""P1-MultiStep-Mamba launcher: P1-HUD-Change with Multi-Step Mamba belief self-consistency.
Supervises blind student belief evolution across horizons H in {2, 4, 8, 16} with plain MSE.
"""
import os
os.environ.setdefault('JAX_PLATFORMS', 'cpu')
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
import argparse
import copy
import importlib
import json
from pathlib import Path
import torch
from saccade import data as D, train as TR
C = importlib.import_module('artifacts.experiments.20261103_p1f2.chooser')
T = importlib.import_module('artifacts.experiments.20261111_p1_multistep_mamba.trainer')

HERE = Path(__file__).resolve().parent
CONTROL = HERE.parent / '20261106_p1f4/run/ckpt_10.pt'
CONTROL_SHA = '1b70edcc432daf3fb337c37c67af8d709d3728e25d02ac94523f8a7322ba1379'
WORLD_SEED = 1
SHORTCUT = {'k_max': 8,
            'self_fraction': 0.25,
            'bootstrap_start': 5000,
            'ramp': [0.9, 0.1],
            'context_signal': 7,
            'init_seed': 1106}
MEMORY = {'accumulate_at_64': 2,
          'accumulate_at_256': 8}
SOURCES = {'change_target': 'full_frame_a9b_v2 (artifacts/saccade_full_frame_change/targets)',
           'multistep': 'Multi-step Mamba belief self-consistency (H in {2, 4, 8, 16}, MSE)',
           'parent_arm': 'P1-HUD-Change (artifacts/experiments/20261109_p1_hud_change)'}


def recipe(cfg):
    cfg = copy.deepcopy(cfg)
    cfg['heads']['change_map'] = {'target': 'full_frame_a9b_v2', 'hidden': 256, 'position_dim': 32, 'belief_dim': 128}
    cfg['p1']['change_targets_dir'] = 'artifacts/saccade_full_frame_change/targets'
    cfg['data'].update(MEMORY)
    return cfg


def check_recipe(cfg):
    assert TR.file_sha256(CONTROL) == CONTROL_SHA, 'F4 control changed'
    ck = torch.load(CONTROL, map_location='cpu', weights_only=False)
    assert cfg['world_seed'] == WORLD_SEED if 'world_seed' in cfg else ck['world_seed'] == WORLD_SEED
    assert cfg['heads']['change_map']['target'] == 'full_frame_a9b_v2'
    assert cfg['p1']['change_targets_dir'] == 'artifacts/saccade_full_frame_change/targets'
    saved, now = ck['provenance'], TR.run_provenance(cfg)
    assert saved['manifest_sha256'] == now['manifest_sha256'], 'corpus differs from F4'
    assert ck['fixation_controller']['config'] == C.config(), 'chooser differs from F4'
    return cfg


def identity(cfg, policy):
    files = [Path(__file__), HERE / 'trainer.py', HERE / 'model_multistep.py', Path(C.__file__), Path(C.Q.__file__)]
    return dict(recipe='P1-MultiStep-Mamba world with belief self-consistency; from scratch',
                world_seed=WORLD_SEED, chooser=policy, shortcut=SHORTCUT, memory=MEMORY, sources=SOURCES,
                source_sha256={str(p.relative_to(TR.ROOT)): TR.file_sha256(p) for p in files},
                control=dict(path=str(CONTROL), sha256=CONTROL_SHA), mamba_deterministic=True,
                cuda_alloc_conf=os.environ['PYTORCH_CUDA_ALLOC_CONF'],
                provenance={k: v for k, v in TR.run_provenance(cfg).items() if k != 'git_commit'})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('out')
    ap.add_argument('--updates', type=int)
    ap.add_argument('--no-checkpoints', action='store_true')
    args = ap.parse_args()
    assert os.environ.get('MAMBA_DETERMINISTIC') == '1'
    cfg = check_recipe(recipe(TR.load_cfg()))
    policy = C.config()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    expected = identity(cfg, policy)
    file = out / 'identity.json'
    if file.exists():
        assert json.loads(file.read_text()) == expected, 'Different launch identity on resume'
    else:
        assert not (out / 'latest.pt').exists(), 'Checkpoint without identity'
        file.write_text(json.dumps(expected, indent=2) + '\n')
    T.p1(cfg, WORLD_SEED, out, updates=args.updates, checkpoints=not args.no_checkpoints, controller_cfg=policy)


if __name__ == '__main__':
    main()
