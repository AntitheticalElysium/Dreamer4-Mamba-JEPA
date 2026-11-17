"""Temporal memory pass, test 2 at a saved checkpoint (PLAN.md amendment: the run stopped at 5,600, read at 4,000):
check 16b's natural catalogue on `run/record/denoiser_U.pt`.

    MAMBA_DETERMINISTIC=1 python -m artifacts.experiments.20261116_temporal_memory.record_eval 4000
"""
import importlib
import shutil
import sys

RFT = importlib.import_module("artifacts.experiments.20261116_temporal_memory.record_ft")
TF, UC = RFT.TF, RFT.UC

if __name__ == "__main__":
    u = int(sys.argv[1])
    TF.traces, TF.install = RFT.record, RFT.install
    cat = RFT.RUN / f"cat{u}"
    cat.mkdir(parents=True, exist_ok=True)
    (cat / "denoiser_2000.pt").unlink(missing_ok=True)
    (cat / "denoiser_2000.pt").symlink_to(RFT.RUN / f"denoiser_{u}.pt")  # use_check.load reads TF.OUT / denoiser_2000.pt
    TF.OUT, UC.HERE = cat, cat
    UC.catalogue()
    shutil.copy(cat / "catalogue.json", RFT.HERE / f"record_catalogue_{u}.json")
