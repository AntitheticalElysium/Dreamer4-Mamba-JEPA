"""Seal P1-F3's sources after its preflight passed (F2.1's seal unchanged: F2's pin list + this folder + named inputs).
Never starts training."""
import importlib
from pathlib import Path
P = importlib.import_module('artifacts.experiments.20261104_p1f21.pin_sources')
HERE = Path(__file__).resolve().parent
CT = 'artifacts/experiments/20261105_consequence_trace'

if __name__ == '__main__':
    inputs = [f'{CT}/{n}' for n in ('trace.py', 'sampler_steps.py', 'readings.py', 'posthoc.py', 'weighting_allocation.py',
                                    'interface_steps.py', 'recursion_steps.py', 'glimpse_check.py', 'noise_profile.py')]
    inputs += ['artifacts/experiments/20261030_actor_interface/interface.py',
               'artifacts/experiments/20261101_regression_localization/trace.py',
               'artifacts/experiments/20261029_generation_diagnosis/field_calibration.py',
               'artifacts/experiments/20261029_generation_diagnosis/catalogue.py',
               'artifacts/experiments/20261029_generation_diagnosis/information_probe.py',
               'artifacts/experiments/20261030_increment_diagnosis/increment_probe.py',
               'artifacts/experiments/20261104_p1f21/run.py', 'artifacts/experiments/20261104_p1f21/checks.py',
               'artifacts/experiments/20261103_p1f2/run/ckpt_10.pt', 'artifacts/eda/s0/s0_dev_labels.npz']
    P.seal(HERE, ('PLAN.md', 'checks.json'), inputs)
