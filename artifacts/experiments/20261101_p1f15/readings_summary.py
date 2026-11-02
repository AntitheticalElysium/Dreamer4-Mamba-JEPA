"""Codex's readings for P1-F1.5 against P1-F, through beta-TSP's summary (`../20261101_beta_tsp/readings_summary.py`:
Codex's own estimators, paired differences on the same scenes / contexts / draws), unchanged: its inputs point at a
directory linking P1-F's outputs (beta-TSP's readings/codex, `p1f10`) and this run's (`p1f15`), with one sampler file
holding both worlds (P1-F's from beta-TSP's). Its output keys say "beta": there, B - A = p1f15 - p1f10. CPU.
Usage: python -m artifacts.experiments.20261101_p1f15.readings_summary
"""
import importlib, json
from pathlib import Path

RS = importlib.import_module("artifacts.experiments.20261101_beta_tsp.readings_summary")
HERE = Path(__file__).resolve().parent
COMB = HERE / "readings/codex_vs_p1f"


def main():
    COMB.mkdir(parents=True, exist_ok=True)
    for src, name in ((RS.OUT, "p1f10"), (HERE / "readings/codex", "p1f15")):
        for f in src.glob(f"{name}_*.json"):
            if not (COMB / f.name).exists():
                (COMB / f.name).symlink_to(f.resolve())
    models = {"p1f10": json.loads((RS.OUT / "sampler_results.json").read_text())["models"]["p1f10"],
              "p1f15": json.loads((HERE / "readings/codex/sampler_results.json").read_text())["models"]["p1f15"]}
    (COMB / "sampler_results.json").write_text(json.dumps({"models": models}))
    RS.OUT, RS.A, RS.B, RS.HERE = COMB, "p1f10", "p1f15", HERE
    RS.main()


if __name__ == "__main__":
    main()
