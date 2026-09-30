"""Post-hoc decomposition of U/W native SLEEP-vs-other score margin on saved roots."""
import json
import sys
from pathlib import Path
import torch
HERE=Path(__file__).parent
ROOT=HERE.parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'artifacts/experiments/20260921_readout_ladder')]
from frozen_ladder import strata
D=ROOT/'artifacts/eda/diagnosis_dump_v1'
rows=[]
for block in ('55k','56k','57k','58k'):
    m=torch.load(D/f'{block}_meta.pt',weights_only=False)
    z=strata(m['visible'])['zombie_adjacent']
    truth=m['p_death1'][z]
    p={a:torch.load(D/f'{block}_{a}.pt',weights_only=False)['p_dead'][z] for a in ('U','W')}
    def pieces(x):
        y=x.clone();y[:,6]=float('inf')
        return x[:,6],y.min(1).values,x.argmin(1)
    us,uo,ua=pieces(p['U']);ws,wo,wa=pieces(p['W'])
    moved=(ua==6)&(wa!=6)
    rows.append({'block':block,'n':int(z.sum()),'u_sleep':float(us.mean()),'w_sleep':float(ws.mean()),
                 'u_best_other':float(uo.mean()),'w_best_other':float(wo.mean()),
                 'sleep_margin_shift':float((ws-us).mean()),'other_margin_shift':float((uo-wo).mean()),
                 'u_sleep_share':float((ua==6).float().mean()),'w_sleep_share':float((wa==6).float().mean()),
                 'u_sleep_to_w_other':int(moved.sum()),
                 'mean_safety_gain_on_switches':float(((1-truth.gather(1,wa[:,None]).squeeze(1))-(1-truth[:,6]))[moved].mean()) if moved.any() else None,
                 'switch_sleep_score_shift':float((ws-us)[moved].mean()) if moved.any() else None,
                 'switch_best_other_score_shift':float((uo-wo)[moved].mean()) if moved.any() else None,
                 'switch_u_margin':float((us-uo)[moved].mean()) if moved.any() else None,
                 'switch_w_margin':float((ws-wo)[moved].mean()) if moved.any() else None})
(HERE/'score_margin.json').write_text(json.dumps(rows,indent=2)+'\n')
print(json.dumps(rows,indent=2))
