"""One-shot approved P1-F1.6 training -> fixed selection -> same F1.5 evaluations."""
import os
os.environ.setdefault('JAX_PLATFORMS','cpu')
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF','expandable_segments:True')
import argparse
from datetime import datetime, timezone
import importlib
import json
from pathlib import Path
import subprocess
import sys
import traceback
from saccade import train as TR
from saccade.eval.after_p1 import select_checkpoint

L=importlib.import_module('artifacts.experiments.20261102_p1f16.train_p1f16')
HERE=Path(__file__).resolve().parent


def now():return datetime.now(timezone.utc).isoformat()


def write(path,value):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(path)


def validate_sources():
    pin=json.loads((HERE/'SOURCE_PINS.json').read_text())
    for relative,sha in pin['files'].items():
        if TR.file_sha256(TR.ROOT/relative)!=sha:raise RuntimeError(f'Pinned source changed: {relative}')
    return pin


def select(run):
    records=[r for line in (run/'log.jsonl').read_text().splitlines() if 'monitor' in (r:=json.loads(line))]
    cfg=L.check_recipe(L.recipe(TR.load_cfg()))
    decision=select_checkpoint(records,cfg)
    for r in records:
        if not (run/f"ckpt_{r['checkpoint']:02d}.pt").is_file():raise FileNotFoundError(r)
    import torch
    final=torch.load(run/'ckpt_10.pt',map_location='cpu',weights_only=False)
    TR.validate_resume(final,cfg,L.WORLD_SEED,TR.run_provenance(cfg))
    if final['state']['frames']<cfg['p1']['frames']:raise RuntimeError('Incomplete training budget')
    chosen=run/f"ckpt_{decision['checkpoint']:02d}.pt"
    decision.update(checkpoint_path=str(chosen),checkpoint_sha256=TR.file_sha256(chosen),selected_at=now())
    out=run/'s1';out.mkdir(exist_ok=True)
    if (out/'selection.json').exists():raise RuntimeError('Evaluation already attempted; manual review required')
    write(out/'selection.json',decision)
    return chosen,decision


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--dry-run',action='store_true');args=ap.parse_args()
    pin=validate_sources();run=HERE/'run'
    if args.dry_run:
        print(json.dumps(dict(train=[sys.executable,'-u','-m','artifacts.experiments.20261102_p1f16.train_p1f16',str(run)],
            evaluation=['bash',str(HERE/'queue.sh'),'SELECTED_CHECKPOINT'],sources=len(pin['files'])),indent=2));return
    assert os.environ.get('MAMBA_DETERMINISTIC')=='1'
    status=HERE/'STATUS.json'
    if status.exists():raise RuntimeError('One-shot worker already attempted; inspect status before manual continuation')
    state=dict(phase='training',started_at=now(),source_pins_sha256=TR.file_sha256(HERE/'SOURCE_PINS.json'))
    write(status,state)
    try:
        with open(HERE/'training.log','a') as log:
            subprocess.run([sys.executable,'-u','-m','artifacts.experiments.20261102_p1f16.train_p1f16',str(run)],
                           cwd=TR.ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
        validate_sources()
        chosen,decision=select(run)
        state.update(phase='evaluation',training_completed_at=now(),selection=decision);write(status,state)
        with open(HERE/'evaluation.log','a') as log:
            subprocess.run(['bash',str(HERE/'queue.sh'),str(chosen)],cwd=TR.ROOT,
                           stdout=log,stderr=subprocess.STDOUT,check=True)
        validate_sources()
        state.update(phase='complete',completed_at=now());write(status,state)
    except BaseException as e:
        state.update(failed_phase=state['phase'],phase='failed',failed_at=now(),error=repr(e))
        write(status,state);traceback.print_exc();raise


if __name__=='__main__':main()
