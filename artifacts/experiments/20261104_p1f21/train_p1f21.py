"""P1-F2.1: F2's recipe (F1.6 world + the original online cheap-head fixation, F2's trainer and chooser unchanged) on
the union corpus (union.py), from scratch, world seed 1; no automatic launch. The recipe differs from F2's checkpoint
cfg in exactly data.corpus and p1.out_dir (its change-map targets); saccade's sources are byte-identical to F2's."""
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
F2L = importlib.import_module('artifacts.experiments.20261103_p1f2.train_p1f2')
U = importlib.import_module('artifacts.experiments.20261104_p1f21.union')
T, C = F2L.T, F2L.C
HERE = Path(__file__).resolve().parent
CONTROL = HERE.parent / '20261103_p1f2/run/ckpt_10.pt'
CONTROL_SHA = 'e0f6a24ec51d2bf021ca88ba8080f804595800dc9295aecf0e37c7da7d7e1234'
WORLD_SEED = 1
OVERRIDE = {('data', 'corpus'): U.CORPUS, ('p1', 'out_dir'): U.OUT_DIR}


def recipe(cfg):
    cfg = F2L.recipe(copy.deepcopy(cfg))
    for (a, b), v in OVERRIDE.items():
        cfg[a][b] = v
    return cfg


def check_recipe(cfg):
    assert TR.file_sha256(CONTROL) == CONTROL_SHA, 'F2 control changed'
    ck = torch.load(CONTROL, map_location='cpu', weights_only=False)
    old = copy.deepcopy(cfg)
    for a, b in OVERRIDE:
        old[a][b] = ck['cfg'][a][b]
    assert old == ck['cfg'] and ck['world_seed'] == WORLD_SEED, 'changes beyond the corpus'
    saved, now = ck['provenance'], TR.run_provenance(cfg)
    assert now['source_sha256'] == saved['source_sha256'] and now['torch'] == saved['torch'], 'sources differ from F2'
    assert ck['fixation_controller']['config'] == C.config(), 'chooser differs from F2'
    return cfg


def identity(cfg, policy):
    files = [Path(__file__), HERE / 'union.py', HERE.parent / '20261103_p1f2/trainer.py', Path(C.__file__),
             HERE.parent / '20261103_p1f2/chooser.json', Path(C.Q.__file__), Path(D.RP.__file__), TR.ROOT / U.CORPUS / 'manifest.json']
    return dict(recipe='P1-F2 on the union corpus (original + coverage-v1 + risk-v1); from scratch', world_seed=WORLD_SEED,
                chooser=policy, source_sha256={str(p.relative_to(TR.ROOT)): TR.file_sha256(p) for p in files},
                control=dict(path=str(CONTROL), sha256=CONTROL_SHA), mamba_deterministic=True,
                cuda_alloc_conf=os.environ['PYTORCH_CUDA_ALLOC_CONF'],
                provenance={k: v for k, v in TR.run_provenance(cfg).items() if k != 'git_commit'})


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('out'); ap.add_argument('--updates', type=int)
    ap.add_argument('--no-checkpoints', action='store_true'); args = ap.parse_args()
    assert os.environ.get('MAMBA_DETERMINISTIC') == '1'
    cfg = check_recipe(recipe(TR.load_cfg())); policy = C.config()
    U.use(cfg)
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
