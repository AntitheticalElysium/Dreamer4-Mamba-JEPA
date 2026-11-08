"""Reuse paired F1.5 estimators; A=F1.5, B=F1.6. No new metric or selection."""
import importlib
import json
from pathlib import Path

RS=importlib.import_module('artifacts.experiments.20261101_beta_tsp.readings_summary')
HERE=Path(__file__).resolve().parent
BASE=HERE.parent/'20261101_p1f15/readings/codex'


def main():
    new=HERE/'readings/codex'; combined=HERE/'readings/codex_vs_p1f15'
    combined.mkdir(parents=True,exist_ok=True)
    for source,name in ((BASE,'p1f15'),(new,'p1f16')):
        for f in source.glob(f'{name}_*.json'):
            dst=combined/f.name
            if not dst.exists():dst.symlink_to(f.resolve())
    models={name:json.loads((source/'sampler_results.json').read_text())['models'][name]
            for source,name in ((BASE,'p1f15'),(new,'p1f16'))}
    (combined/'sampler_results.json').write_text(json.dumps(dict(models=models)))
    RS.OUT,RS.A,RS.B,RS.HERE=combined,'p1f15','p1f16',HERE
    RS.main()
    file=HERE/'readings/codex_summary.json'
    result=json.loads(file.read_text())
    result['comparison_labels']=dict(A='p1f15',B='p1f16',paired_difference='B minus A; legacy metric keys retain beta/p1f wording')
    file.write_text(json.dumps(result,indent=1))


if __name__=='__main__':main()
