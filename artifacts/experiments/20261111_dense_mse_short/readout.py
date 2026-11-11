"""Reuse the existing frozen hunger readout, writing only this experiment's outputs."""
import importlib,sys
from pathlib import Path
P=Path(__file__).resolve().parent
def main(arm,checkpoint):
    phase=importlib.import_module("artifacts.experiments.20261111_two_arm_audit.paired_hunger")
    phase.HERE=P
    phase.run(arm,checkpoint)
if __name__=="__main__":main(*sys.argv[1:])
