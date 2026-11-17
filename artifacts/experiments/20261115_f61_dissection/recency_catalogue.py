"""F6.1 dissection, check 16b (PLAN.md): check 15's natural-frame catalogue on the recency fine-tune (check 16), with
its own and with zeroed recency (u = 0: no change in the last 63 frames). Same rows, seeds and readings as check 15.

    MAMBA_DETERMINISTIC=1 python -m artifacts.experiments.20261115_f61_dissection.recency_catalogue
"""
import importlib
import shutil

RF = importlib.import_module("artifacts.experiments.20261115_f61_dissection.recency_ft")
UC, TF = RF.UC, RF.TF

if __name__ == "__main__":
    TF.traces, TF.install, TF.OUT, UC.HERE = RF.recency, RF.install, RF.RUN, RF.RUN   # use_check.load / catalogue read these
    UC.catalogue()                                                      # arms f61 / traces_own / traces_zero -> RUN/catalogue.json
    shutil.copy(RF.RUN / "catalogue.json", RF.HERE / "recency_catalogue.json")
