"""F6.1 dissection, check 6 (PLAN.md): is the under-drawn rise the few-step sampler or the learned conditional? The
quiet bank's own generation for F6.1 (`../20261107_danger_adversarial/sleep.py`, unchanged) with the shortcut sampler
at K = 8 evaluations in place of the trained recipe's K = 4 (k_max = 8; the only other power of two above 4 is not
allowed), read by check 3's hazard reader beside the K = 4 queue run on the same scenes and seeds.

    MAMBA_DETERMINISTIC=1 python -m artifacts.experiments.20261115_f61_dissection.sampler_steps      (GPU)
"""
import copy
import importlib
import json
from pathlib import Path

SL = importlib.import_module("artifacts.experiments.20261107_danger_adversarial.sleep")
HZ = importlib.import_module("artifacts.experiments.20261115_f61_dissection.hazard")
HERE = Path(__file__).resolve().parent
OUT = HERE / "run/sampler8/F6.1"
CKPT = "artifacts/experiments/20261114_p1f61/run/ckpt_10.pt"


def main():
    original = SL.CV.IF.load

    def load(path):
        model, cfg = original(path)
        cfg = copy.deepcopy(cfg)
        assert cfg["edm"]["sampler"]["steps"] == 4
        cfg["edm"]["sampler"]["steps"] = 8
        return model, cfg
    SL.CV.IF.load = load
    OUT.mkdir(parents=True, exist_ok=True)
    if not (OUT / "run.pt").exists():
        (OUT / "bank.pt").write_bytes((SL.HERE / "bank.pt").read_bytes())
        SL.HERE, SL.CKPT = OUT, CKPT
        SL.run()
    res = {"K8": HZ.read(OUT / "run.pt"), "K4_queue": HZ.read(HZ.BANKS["quiet_bank"]["F6.1"])}
    for k, v in res.items():
        print(k, json.dumps(v), flush=True)
    (HERE / "sampler_steps.json").write_text(json.dumps(res, indent=1) + "\n")


if __name__ == "__main__":
    main()
