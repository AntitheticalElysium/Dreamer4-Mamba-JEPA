"""F1.5's exact Codex evaluators, redirected and relabeled for F1.6 only."""
import importlib
import json
import sys
from pathlib import Path
from saccade import train as TR

OLD = importlib.import_module('artifacts.experiments.20261101_p1f15.readings_codex')
HERE = Path(__file__).resolve().parent


def main():
    what, path = sys.argv[1], Path(sys.argv[2])
    out = HERE/'readings/codex'; out.mkdir(parents=True,exist_ok=True)
    OLD.HERE = HERE
    OLD.BR.OUT = out
    if what == 'sampler':
        OLD.sampler(path)
        file = out/'sampler_results.json'
        result = json.loads(file.read_text())
        result['models']['p1f16'] = result['models'].pop('p1f15')
        result['p1f16_wrapper_sha256'] = TR.file_sha256(__file__)
        file.write_text(json.dumps(result,indent=1))
    else:
        OLD.BR.WORLDS = {'p1f16':path}
        sys.argv = [sys.argv[0],what,'p1f16']
        OLD.BR.main()


if __name__=='__main__':main()
