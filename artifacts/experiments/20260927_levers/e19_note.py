"""Append completed E19 measurements to the existing EDA logs, once per matched set."""
import argparse
import datetime
import fcntl
import json
from pathlib import Path

p=argparse.ArgumentParser()
p.add_argument('--seed',type=int,required=True)
p.add_argument('--backbone',choices=['fmamba','full'],required=True)
a=p.parse_args()
here=Path(__file__).resolve().parent
summary=here.parents[2]/'artifacts/eda/levers_logs'/f'e19_{a.backbone}_s{a.seed}_summary.log'
tag=f'E19_COMPLETE_{a.backbone}_s{a.seed}'
lines=[f'\n\n## {datetime.datetime.now().astimezone().isoformat()} — {tag}\n',
       '\nAutomatically recorded exact endpoint measurements; causal interpretation remains a separate diagnosis. '
       'Reused diagnostic roots; not a sealed policy or actor test.\n',
       '\n| arm | >=2 damage caught (cut1.5) | fresh hit catch | unchanged false drops | w4/w5 drawn ratio |\n',
       '|---|---:|---:|---:|---:|\n']
notes=[]
for arm in 'ABC':
    name=f'e19_{arm}_s{a.seed}_{a.backbone}_from36000'
    r=json.loads((here/'evals'/f'{name}__e19_health.json').read_text())
    m=r['metrics']['w5_at0']['all_diagnostic_roots']
    v=m['overall']; fresh=m['fresh']
    w4=r['metrics']['w4_at0']['all_diagnostic_roots']['overall']['drawn_rate']
    ratio=w4/v['drawn_rate'] if v['drawn_rate'] else None
    fmt=lambda x: f'{x:.6f}' if x is not None else 'undefined'
    lines.append(f"| {arm} | {v['hits_drawn']}/{v['n_hit']} = {fmt(v['hit_catch'])} | "
                 f"{fresh['hits_drawn']}/{fresh['n_hit']} = {fmt(fresh['hit_catch'])} | "
                 f"{v['false_drops']}/{v['n_unchanged']} = {fmt(v['false_drop_rate'])} | {fmt(ratio)} |\n")
    notes.append(f"\n{arm} threshold readings: `{r['readings']}`. Evidence: "
                 f'`20260927_levers/evals/{name}__e19_health.json`; raw per-root `.pt` retained.\n')
lines.extend(notes)
lines.append('\nPaired parent→A, A→B, B→C intervals: '
             f'`20260927_levers/evals/e19_contrast_e19_C_s{a.seed}_{a.backbone}_from36000.json`. '
             'Half-unit damage sensitivity and self-fed errors: '
             f'`artifacts/eda/levers_logs/e19_damage_{a.backbone}_s{a.seed}.log`. '
             'Prediction-cost comparisons recorded in `20260927_levers/compare.json`; '
             'do not infer no-cost or action-selection success from health catches alone.\n')
# Experiments write raw/derived measurements to EDA logs. Agents alone maintain the
# research notebook after inspecting the evidence. Lock this summary to avoid duplicates.
summary.parent.mkdir(parents=True,exist_ok=True)
with summary.open('a+') as f:
    fcntl.flock(f,fcntl.LOCK_EX)
    f.seek(0)
    if tag not in f.read():
        f.write(''.join(lines));f.flush()
    fcntl.flock(f,fcntl.LOCK_UN)
print(f'{tag}: {summary}')
