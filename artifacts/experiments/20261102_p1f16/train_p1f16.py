"""P1-F1.6: original P1 trainer, from scratch, F1.5 recipe plus local fusion only."""
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

F15 = importlib.import_module('artifacts.experiments.20261101_p1f15.train_p1f15')
HERE = Path(__file__).resolve().parent
CONTROL = HERE.parent / '20261101_p1f15/run/ckpt_10.pt'
CONTROL_SHA = '95fa2cc82cb9dc3cae11cbf9f44f880aff17e250de639911f39d159564978128'
WORLD_SEED = 1
FUSION = dict(kind='joint_fine', hidden=128, init_seed=611901)


def recipe(cfg):
    cfg = F15.recipe(copy.deepcopy(cfg))
    cfg['belief']['spatial_fusion'] = dict(FUSION)
    return cfg


def check_recipe(cfg):
    assert TR.file_sha256(CONTROL) == CONTROL_SHA, 'F1.5 control changed'
    ck = torch.load(CONTROL, map_location='cpu', weights_only=False)
    old = copy.deepcopy(cfg); old['belief'].pop('spatial_fusion')
    assert old == ck['cfg'] and ck['world_seed'] == WORLD_SEED, 'changes beyond spatial fusion'
    return cfg


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('out')
    ap.add_argument('--updates', type=int); ap.add_argument('--no-checkpoints', action='store_true')
    args = ap.parse_args()
    assert os.environ.get('MAMBA_DETERMINISTIC') == '1'
    cfg = check_recipe(recipe(TR.load_cfg()))
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    prov = TR.run_provenance(cfg)
    identity = dict(recipe='P1-F1.5 + mandatory joint64 spatial fusion; from scratch',
        world_seed=WORLD_SEED, fusion=FUSION, script_sha256=TR.file_sha256(__file__),
        mamba_deterministic=True, cuda_alloc_conf=os.environ['PYTORCH_CUDA_ALLOC_CONF'],
        control=dict(path=str(CONTROL), sha256=CONTROL_SHA),
        provenance={k:v for k,v in prov.items() if k!='git_commit'})
    file = out/'identity.json'
    if file.exists():
        assert json.loads(file.read_text()) == identity, 'resume with a different identity'
    else:
        assert not (out/'latest.pt').exists(), 'checkpoint without identity'
        file.write_text(json.dumps(identity, indent=2)+'\n')
    TR.p1(cfg, WORLD_SEED, out, updates=args.updates, checkpoints=not args.no_checkpoints)


if __name__=='__main__':main()
