"""Seal P1-F4's sources after its preflight passed (F2.1's seal: F2's pin list + this folder + named inputs, at their
current hashes; the changed saccade/model.py and train.py are in F2's list). Never starts training."""
import importlib
from pathlib import Path
P = importlib.import_module('artifacts.experiments.20261104_p1f21.pin_sources')
HERE = Path(__file__).resolve().parent

if __name__ == '__main__':
    inputs = ['artifacts/experiments/20261106_determinant_curves/curves.py', 'artifacts/experiments/20261106_decision_levels/crafting.py',
              'artifacts/experiments/20261106_decision_levels/frozen.py', 'artifacts/experiments/20261106_determinant_routes/routes.py',
              'artifacts/experiments/20261030_actor_interface/interface.py', 'artifacts/experiments/20261105_consequence_trace/recursion_steps.py',
              'artifacts/experiments/20261104_p1f21/run.py', 'artifacts/experiments/20261103_p1f2/trainer.py',
              'artifacts/experiments/20261103_p1f2/checks.py', 'artifacts/experiments/20261105_p1f3/run/ckpt_10.pt',
              'artifacts/experiments/20261105_p1f3/run/log.jsonl', 'artifacts/eda/s0/s0_dev_labels.npz',
              'artifacts/eda/s0/s0_train400_labels.npz', 'saccade/tests/test_m0.py']
    P.seal(HERE, ('PLAN.md', 'checks.json'), inputs)
