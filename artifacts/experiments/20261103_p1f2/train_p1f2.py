"""F1.6 world recipe, original online cheap-head fixation; no automatic launch."""
import os
os.environ.setdefault('JAX_PLATFORMS','cpu')
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF','expandable_segments:True')
import argparse
import importlib
import json
from pathlib import Path
import torch
from saccade import train as TR
F16=importlib.import_module('artifacts.experiments.20261102_p1f16.train_p1f16')
T=importlib.import_module('artifacts.experiments.20261103_p1f2.trainer')
C=T.C
HERE=Path(__file__).resolve().parent
CONTROL=HERE.parent/'20261102_p1f16/run/ckpt_10.pt'
CONTROL_SHA='34bda77fe5ec5226c4f003301c972d918e2d4a4a554a5a6f279fcabbe922b045'
WORLD_SEED=1
recipe=F16.recipe


def check_recipe(cfg):
    assert TR.file_sha256(CONTROL)==CONTROL_SHA,'F1.6 control changed'
    ck=torch.load(CONTROL,map_location='cpu',weights_only=False)
    assert cfg==ck['cfg'] and ck['world_seed']==WORLD_SEED,'World recipe differs from F1.6'
    TR.validate_resume(ck,cfg,WORLD_SEED,TR.run_provenance(cfg))
    return cfg


def identity(cfg,policy):
    files=[Path(__file__),HERE/'trainer.py',HERE/'chooser.py',HERE/'chooser.json',Path(C.Q.__file__)]
    return dict(recipe='P1-F1.6 + original coarse-only learned fixation; from scratch',world_seed=WORLD_SEED,
        chooser=policy,source_sha256={str(p.relative_to(TR.ROOT)):TR.file_sha256(p) for p in files},
        control=dict(path=str(CONTROL),sha256=CONTROL_SHA),mamba_deterministic=True,
        cuda_alloc_conf=os.environ['PYTORCH_CUDA_ALLOC_CONF'],
        provenance={k:v for k,v in TR.run_provenance(cfg).items() if k!='git_commit'})


def main():
    ap=argparse.ArgumentParser();ap.add_argument('out');ap.add_argument('--updates',type=int)
    ap.add_argument('--no-checkpoints',action='store_true');args=ap.parse_args()
    assert os.environ.get('MAMBA_DETERMINISTIC')=='1'
    cfg=check_recipe(recipe(TR.load_cfg()));policy=C.config()
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    expected=identity(cfg,policy);file=out/'identity.json'
    if file.exists():assert json.loads(file.read_text())==expected,'Different launch identity on resume'
    else:
        assert not (out/'latest.pt').exists(),'Checkpoint without identity'
        file.write_text(json.dumps(expected,indent=2)+'\n')
    T.p1(cfg,WORLD_SEED,out,updates=args.updates,checkpoints=not args.no_checkpoints,controller_cfg=policy)


if __name__=='__main__':main()
