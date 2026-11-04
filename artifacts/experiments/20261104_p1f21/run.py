"""One-shot worker for an F2 arm (F2.1 here; F2.2 calls main with its own launcher; launch requires user approval
after review): preflight gates -> training -> F2's fixed checkpoint rule. Stops there: the evaluation queue (against
F2) is written and reviewed while it trains."""
import os
os.environ.setdefault('JAX_PLATFORMS', 'cpu')
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
from datetime import datetime, timezone
import importlib
import json
from pathlib import Path
import subprocess
import sys
import traceback
from saccade import train as TR
from saccade.eval.after_p1 import select_checkpoint

L = importlib.import_module('artifacts.experiments.20261104_p1f21.train_p1f21')
HERE = Path(__file__).resolve().parent


def now(): return datetime.now(timezone.utc).isoformat()


def write(path, value):
    tmp = path.with_suffix('.tmp'); tmp.write_text(json.dumps(value, indent=2) + '\n'); tmp.replace(path)


def validate_sources(here):
    pin = json.loads((here / 'SOURCE_PINS.json').read_text())
    for relative, sha in pin['files'].items():
        if TR.file_sha256(TR.ROOT / relative) != sha:
            raise RuntimeError(f'Pinned source changed: {relative}')
    return pin


def select(L, run):
    """F2's run.select: every numbered monitor record, the fixed rule, the final checkpoint's identity and chooser."""
    import torch
    records = [r for line in (run / 'log.jsonl').read_text().splitlines() if 'monitor' in (r := json.loads(line))]
    cfg = L.check_recipe(L.recipe(TR.load_cfg()))
    decision = select_checkpoint(records, cfg)
    for r in records:
        if not (run / f"ckpt_{r['checkpoint']:02d}.pt").is_file():
            raise FileNotFoundError(r)
    final = torch.load(run / 'ckpt_10.pt', map_location='cpu', weights_only=False)
    TR.validate_resume(final, cfg, L.WORLD_SEED, TR.run_provenance(cfg))
    if final['state']['frames'] < cfg['p1']['frames']:
        raise RuntimeError('Incomplete training budget')
    controller = final['fixation_controller']
    if controller['config'] != L.C.config() or controller['frames'] != final['state']['frames']:
        raise RuntimeError('Incomplete or mismatched fixation controller')
    chosen = run / f"ckpt_{decision['checkpoint']:02d}.pt"
    decision.update(checkpoint_path=str(chosen), checkpoint_sha256=TR.file_sha256(chosen), selected_at=now())
    write(run / 'selection.json', decision)
    return decision


def main(L=L, here=HERE, module='artifacts.experiments.20261104_p1f21.train_p1f21', gates=('union_checks.json', 'checks.json')):
    assert os.environ.get('MAMBA_DETERMINISTIC') == '1'
    validate_sources(here); run = here / 'run'
    for gate in gates:
        if not json.loads((here / gate).read_text()).get('passed'):
            raise RuntimeError(f'Preflight not passed: {gate}')
    status = here / 'STATUS.json'
    if status.exists():
        raise RuntimeError('One-shot worker already attempted; inspect status before manual continuation')
    state = dict(phase='training', started_at=now(), source_pins_sha256=TR.file_sha256(here / 'SOURCE_PINS.json'))
    write(status, state)
    try:
        with open(here / 'training.log', 'a') as log:
            subprocess.run([sys.executable, '-u', '-m', module, str(run)],
                           cwd=TR.ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
        validate_sources(here)
        state.update(phase='trained', training_completed_at=now(), selection=select(L, run)); write(status, state)
    except BaseException as e:
        state.update(failed_phase=state['phase'], phase='failed', failed_at=now(), error=repr(e))
        write(status, state); traceback.print_exc(); raise


if __name__ == '__main__':
    main()
