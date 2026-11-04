"""Check new descriptive instruments against preserved primary outputs."""
import importlib
import json
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent;RUN=HERE.parent/'20261103_p1f2'

def main():
    t=json.loads((HERE/'TRACE.json').read_text());old=json.loads((RUN/'readings/chooser_trace.json').read_text());out={}
    for kind,oldkind in [('any_entity','entities'),('zombies','zombies'),('cows','cows'),('skeletons','skeletons'),('arrows','arrows')]:
        for policy in ['learned','paired_uniform']:
            assert t['coverage'][kind]['all'][policy]['mean']==old['entity_contact'][oldkind+'_'+policy]['rate']
    out['original_trace_entity_rates_exact']=True
    e=importlib.import_module('artifacts.experiments.20261030_m05_guided_fovea.evaluate')
    m=json.loads((RUN/'readings/eval_m05/results.json').read_text());ctx=json.loads((HERE/'M05_CONTEXTS.json').read_text())
    for route in ['guided','periphery']:
        for kind in ['drink','wood']:
            a=dict(np.load(RUN/f'readings/eval_m05/base_{route}_{kind}.npz'))
            b=dict(np.load(RUN/f'readings/eval_m05/p1f2_1500_{route}_{kind}.npz'))
            ep=np.array([r[0] for r in ctx[kind]])[a['row_ctx']]
            diff=e.paired_prediction(a,b,ep,kind)
            expected=m['differences']['p1f2_1500-base']['prediction'][route][kind]
            for group in diff:
                for metric in diff[group]:
                    assert all(diff[group][metric][k]==expected[group][metric][k] for k in ['est','se','n_cond','rows'])
    out['counter_population_original_estimator_parity_exact']=True
    d=json.loads((HERE/'DEVELOPMENT.json').read_text());h=json.loads((HERE/'HANDOVER.json').read_text())
    for cl in ['drink','wood','stone','coal','iron']:
        for ck in [1,10]:
            z=d[str(ck)]['cohorts']['confirmation']['counter'][cl]['increment'];r=h['confirmation'][cl]['increment']
            for metric,norm in [('counter_error','normalized_error'),('counter_C','normalized_C')]:
                assert np.isclose(z[norm],r[metric]['ck'+str(ck)]['mean'],rtol=2e-6)
    out['handover_independent_aggregate_parity']=True
    first={co:dict(np.load(HERE/f'ck01/{co}.npz')) for co in ['saved','confirmation']}
    assert not set(first['saved']['ep'])&set(first['confirmation']['ep'])
    for co in first:
        for ck in range(2,11):
            z=dict(np.load(HERE/f'ck{ck:02d}/{co}.npz'))
            for k in ['ep','t','cls','counter','faced','increment','counter_before']:
                assert np.array_equal(z[k],first[co][k])
    out['all_ten_contexts_fixed_and_cohorts_episode_disjoint']=True
    out['numeric_note']='Exact checks concern preserved arrays and estimators; conclusions retain one training seed and instrument limitations.'
    (HERE/'QA.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))

if __name__=='__main__':main()
