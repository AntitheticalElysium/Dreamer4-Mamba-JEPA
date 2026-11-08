"""Report the content control and source-qualified subsets without selecting on prediction outcomes."""
import importlib,json
from pathlib import Path
import numpy as np
import torch
A=importlib.import_module('artifacts.experiments.20261104_fine_causal_design.analyze')
HERE=Path(__file__).resolve().parent

def main():
    torch.set_num_threads(2)
    x=json.loads((HERE/'RESULTS.json').read_text());raw=torch.load(HERE/'content_control.pt',map_location='cpu',weights_only=False)
    assert len(raw['rows'])==12
    valid={s['case'] for s in x['source_checks'] if s['specific_to_presence']};detail=[]
    for r in raw['rows']:
        for draw in r['draws']:
            vals={c:{k:torch.cat([a[k] for a in draw['values'] if a['condition']==c]).numpy() for k in ['arrow','score']} for c in range(4)}
            met=dict(arrow_gA_pA=float(vals[0]['arrow'].mean()),arrow_gB_pA=float(vals[1]['arrow'].mean()),
                arrow_gA_pB=float(vals[2]['arrow'].mean()),arrow_gB_pB=float(vals[3]['arrow'].mean()),
                arrow_content_gain_pA=float((vals[0]['arrow'].astype(float)-vals[1]['arrow']).mean()),
                arrow_content_gain_pB=float((vals[2]['arrow'].astype(float)-vals[3]['arrow']).mean()),
                score_content_gain_pA=float((vals[0]['score']-vals[1]['score']).mean()),
                score_content_gain_pB=float((vals[2]['score']-vals[3]['score']).mean()),
                changed_arrow_calls=int((vals[0]['arrow']!=vals[1]['arrow']).sum()+(vals[2]['arrow']!=vals[3]['arrow']).sum()),
                paired_calls=128,
                embedding_delta_pA=r['embedding_delta'][0]['rms'],predicted_delta_pA=r['predicted_delta'][0]['rms'])
            detail.append(dict(case=r['case'],episode=r['meta']['e'],source_specific=r['case'] in valid,steps=draw['steps'],metrics=met))
    out={}
    for group in ['all12','source_specific5']:
        out[group]={}
        for steps in [3,8]:
            rows=[r for r in detail if r['steps']==steps and (group=='all12' or r['source_specific'])]
            summary={k:A.interval([r['metrics'][k] for r in rows],[r['episode'] for r in rows]) for k in rows[0]['metrics']}
            summary['total_changed_arrow_calls']=sum(r['metrics']['changed_arrow_calls'] for r in rows)
            summary['total_paired_calls']=sum(r['metrics']['paired_calls'] for r in rows)
            out[group][str(steps)]=summary
    primary={}
    for steps in [3,8]:
        rows=[r for r in x['detail'] if r['case'] in valid and r['cohort']=='distant_arrows_coarse_missed' and r['steps']==steps]
        primary[str(steps)]={k:A.interval([r['metrics'][k] for r in rows],[r['episode'] for r in rows]) for k in rows[0]['metrics']}
    result=dict(content=out,genuine_gaze_source_specific5=primary,detail=detail,
        qualification='Five first-visible cases pass BOTH fixed fine readers on BOTH presence/absence alternatives. Other seven retained, not used to claim semantic fine acquisition.',
        non_claim='Internal content swaps need not be natural joint percepts. Zero observed label changes is not a population proof of exact zero influence.')
    (HERE/'CONTENT_RESULTS.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    for group in out:
        for steps in ['3','8']:
            a=out[group][steps]
            print(group,steps,{k:a[k] for k in ['arrow_gA_pA','arrow_gB_pA','arrow_content_gain_pA','arrow_content_gain_pB','score_content_gain_pA','total_changed_arrow_calls','total_paired_calls']})
    print('genuine',primary)

if __name__=='__main__':main()
