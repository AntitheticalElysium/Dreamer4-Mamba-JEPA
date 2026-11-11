"""Seal P1-HUD-Change sources after preflight passed. Never starts training."""
import importlib
from pathlib import Path

P = importlib.import_module('artifacts.experiments.20261104_p1f21.pin_sources')
HERE = Path(__file__).resolve().parent

if __name__ == '__main__':
    inputs = [
        'artifacts/experiments/20261106_determinant_curves/curves.py',
        'artifacts/experiments/20261106_decision_levels/crafting.py',
        'artifacts/experiments/20261030_actor_interface/interface.py',
        'artifacts/experiments/20261105_consequence_trace/recursion_steps.py',
        'artifacts/experiments/20261104_p1f21/run.py',
        'artifacts/experiments/20261103_p1f2/trainer.py',
        'artifacts/experiments/20261103_p1f2/checks.py',
        'artifacts/experiments/20261106_p1f4/run/ckpt_10.pt',
        'artifacts/experiments/20261106_p1f4/run/log.jsonl',
        'artifacts/eda/s0/s0_dev_labels.npz',
        'artifacts/eda/s0/s0_train400_labels.npz',
        'saccade/tests/test_m0.py',
        'saccade/tests/test_change.py',
        'artifacts/experiments/20261108_full_frame_change/natural_pair.py',
        'artifacts/saccade_full_frame_change/targets/change_targets.npy',
        'artifacts/saccade_full_frame_change/targets/change_targets.json',
    ]
    P.seal(HERE, ('PLAN.md', 'checks.json'), inputs)
