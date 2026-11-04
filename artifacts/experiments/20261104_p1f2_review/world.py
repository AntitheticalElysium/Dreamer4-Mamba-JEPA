"""Full fixed-gaze outcome tables and granular saved counter generations. No inference."""
import csv,json,importlib
from pathlib import Path
import numpy as np
from saccade.eval import gates as G,s1 as S
HERE=Path(__file__).resolve().parent;RUN=HERE.parent/'20261103_p1f2';BASE=HERE.parent/'20261102_p1f16'

def main():
    tables=[];interfaces={}
    for policy in ['random','guided']:
        j=json.loads((RUN/f'readings/interface_{policy}.json').read_text());diff=j['differences']['p1f2-p1f16'];interfaces[policy]=j
        for cl in j['classes']:
            a=j['checkpoints']['p1f16']['classes'][cl];b=j['checkpoints']['p1f2']['classes'][cl]
            for metric,truth in [('unlock_p','event'),('neg_p','neg_p'),('term_p','term_p'),('exp_reward','exp_reward')]:
                d=diff[cl][metric]
                tables.append(dict(policy=policy,context=cl,metric=metric,histories=a['histories'],futures=a['futures'],
                    recorded=a['recorded'][truth]['est'],real_head_F16=a['rec_head'][metric]['est'],real_head_F2=b['rec_head'][metric]['est'],
                    generated_F16=a['gen'][metric]['est'],generated_F2=b['gen'][metric]['est'],difference=d['est'],
                    ci95_low=d['ci95'][0],ci95_high=d['ci95'][1]))
    with (HERE/'INTERFACE.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(tables[0]));w.writeheader();w.writerows(tables)
    s1={name:json.loads((root/'run/s1/results.json').read_text()) for name,root in [('F16',BASE),('F2',RUN)]}
    perception={};reader_counts={};belief={}
    for name,j in s1.items():
        perception[name]={}
        for light in ['day_r1_2','night_r1_2','night_r1_4']:
            perception[name][light]={k:float(np.mean([r[light][k]['est'] for r in j['s1_1']['trained']])) for k in j['s1_1']['trained'][0][light]}
        reader_counts[name]=j['s1_3']['reader_counts']
        belief[name]=dict(hud={},achievements={},departed={})
        for mode,rr in j['s1_2'].items():
            belief[name]['hud'][mode]=np.mean([[x['est'] for x in r['hud_exact']] for r in rr],axis=0).tolist()
            belief[name]['achievements'][mode]=np.mean([[a['ap'] for a in r['achievements']] for r in rr],axis=0).tolist()
            belief[name]['departed'][mode]=[r['departed'] for r in rr]
    evals=json.loads((RUN/'readings/eval_m05/results.json').read_text())
    E=importlib.import_module('artifacts.experiments.20261030_m05_guided_fovea.evaluate')
    labels=S.first_episodes(dict(np.load(S.LABELS/'s0_dev_labels.npz')),0)
    # Original selection reads only episode lengths from enc, not token values.
    # Reconstruct with those exact lengths and original RNGs, then verify the
    # saved starts/ends/recorded outcomes before using episode identities.
    dummy=[{'p':range(int(T)+1)} for T in labels['ep_T']]
    contexts,_=E.rows_of(labels,dummy,evals['provenance']['rows_per_class'])
    assert {k:len(v) for k,v in contexts.items()}==evals['rows']
    (HERE/'M05_CONTEXTS.json').write_text(json.dumps(contexts,default=lambda x:x.item())+'\n')
    # Saved success bits retain UNKNOWN as unverified, rather than replacing it
    # with a correct/copy prediction. Per-transition error composition included.
    granular={}
    for route in ['guided','periphery']:
        granular[route]={}
        for kind in ['drink','wood']:
            a=dict(np.load(RUN/f'readings/eval_m05/base_{route}_{kind}.npz'))
            b=dict(np.load(RUN/f'readings/eval_m05/p1f2_1500_{route}_{kind}.npz'))
            assert all(np.array_equal(a[k],b[k]) for k in ['row_ctx','start','end','observed','recorded'])
            # Use episode IDs from saved evaluation rows if available; do not
            # treat the2000 sampled futures as independent episodes.
            kindrows=contexts[kind];info=E.EV.row_info(kind,kindrows,labels)
            starts=np.array([r[1] for r in info])[a['row_ctx']];ends=np.array([r[2] for r in info])[a['row_ctx']]
            recorded=np.array([r[3] for r in info])[a['row_ctx']]
            assert np.array_equal(starts,a['start']) and np.array_equal(ends,a['end']) and np.array_equal(recorded,a['recorded'])
            episodes=np.array([r[0] for r in kindrows])[a['row_ctx']]
            eligible=a['eligible']&b['eligible']
            # Match the original evaluator's input-qualified population. A
            # generated token outside reader support remains UNKNOWN; it is
            # never silently dropped from this denominator.
            for z in [a,b]:
                assert np.all(z['start']!=z['end'])
                partition=sum(z[k].astype(int) for k in ['success','unknown','copy','other'])
                assert np.array_equal(partition,z['eligible'].astype(int))
            original=E.paired_prediction(a,b,episodes,kind)
            for group,mask in E.groups(kind,a['start']).items():
                check=G.estimate(b['success'].astype(float)-a['success'].astype(float),eligible&mask,episodes)
                assert check['est']==original[group]['success']['est']
            result={}
            for value in sorted(np.unique(a['start'])):
                use=eligible&(a['start']==value)
                if not use.any():continue
                rr={}
                for name,z in [('F16',a),('F2',b)]:
                    rr[name]={k:G.estimate(z[k].astype(float),use,episodes) for k in ['success','unknown','copy','other']}
                rr['difference']=G.estimate(b['success'].astype(float)-a['success'].astype(float),use,episodes)
                result[str(int(value))]=rr
            granular[route][kind]=result
    out=dict(perception=perception,reader_counts=reader_counts,belief=belief,counter_transitions=granular,
        c2st={n:j['s1_3']['c2st_generated']['accuracy'] for n,j in s1.items()},
        tsp_nll={n:j['s1_4']['tsp_nats_per_pixel'] for n,j in s1.items()},
        note='Paired one-world-seed, fixed random/guided gaze. Achievement slices are outcome-selected recall diagnostics, not unconditional calibration. Full per-class event mass in INTERFACE.csv.')
    (HERE/'WORLD.json').write_text(json.dumps(out,indent=2)+'\n')
    print('C2ST',out['c2st'],flush=True)
    print('Perception night near', {n:r['night_r1_2'] for n,r in perception.items()},flush=True)
    for route,r in granular.items():
        for k,v in r.items():
            print(route,k,{value:[round(x['success']['est'],3) for x in [r['F16'],r['F2']]] for value,r in v.items()},flush=True)

if __name__=='__main__':main()
