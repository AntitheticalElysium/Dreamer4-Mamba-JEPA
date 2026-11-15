"""F6.1 dissection, check 18 (PLAN.md): does the recency use keep growing toward the cue? Check 16's fine-tune run to
6,000 updates (reads at 2,000 / 4,000 / 6,000), then check 16b's natural catalogue at 6,000.

    MAMBA_DETERMINISTIC=1 python -m artifacts.experiments.20261115_f61_dissection.recency_long
"""
import importlib
import json
import shutil

RF = importlib.import_module("artifacts.experiments.20261115_f61_dissection.recency_ft")
UC, TF = RF.UC, RF.TF
RUN = RF.HERE / "run/recency_long"

if __name__ == "__main__":
    RUN.mkdir(parents=True, exist_ok=True)
    TF.traces, TF.install, TF.OUT, TF.HERE = RF.recency, RF.install, RUN, RUN   # traces_ft.main reads these
    TF.UPDATES, TF.READS = 6000, (2000, 4000, 6000)
    TF.main()                                                           # RUN/traces_ft.json, RUN/denoiser_XXXX.pt
    res = json.loads((RUN / "traces_ft.json").read_text())
    ref = json.loads((RF.HERE / "recency_ft.json").read_text())["reads"]
    res["max_rate_change_vs_check16"] = {u: max(abs(res["reads"][u][b]["rate"][c] - ref[u][b]["rate"][c])
                                                for b in ref[u] for c in ref[u][b]["rate"]) for u in ("0", "2000")}
    (RF.HERE / "recency_long.json").write_text(json.dumps(res, indent=1) + "\n")
    cat = RUN / "cat6000"
    cat.mkdir(exist_ok=True)
    (cat / "denoiser_2000.pt").unlink(missing_ok=True)
    (cat / "denoiser_2000.pt").symlink_to(RUN / "denoiser_6000.pt")     # use_check.load reads TF.OUT / denoiser_2000.pt
    TF.OUT, UC.HERE = cat, cat
    UC.catalogue()
    shutil.copy(cat / "catalogue.json", RF.HERE / "recency_long_catalogue.json")
