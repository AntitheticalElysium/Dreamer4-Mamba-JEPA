"""Paired four-arm readings and bounded runtime retry; no extra training."""
import importlib
import json
from pathlib import Path
import subprocess
import sys
import time
import numpy as np

R=importlib.import_module('artifacts.experiments.20261103_chooser_history.run')
A=importlib.import_module('artifacts.experiments.20261103_gaze_belief.analyze')
C,HERE=R.C,R.HERE
ARMS=('1C','1B','8C','8B');SEEDS=(611601,611602)


def load(arm,seed,step,cohort):
    return dict(np.load(HERE/f'{arm}_{seed}_u{step:05d}_{cohort}.npz'))


def stats(num,ep,den=None):
    return A.estimates(num,np.ones(len(ep)) if den is None else den,ep)[0]


def main():
    # Failed compaction gate is preserved. The revised mechanical check proves
    # float64 equality and bounds real-policy distortion before timing proceeds.
    cost=json.loads((HERE/'COST.json').read_text())
    if not all('shapes' in cost.get(k,{}) for k in ('eager','compiled')):
        with open(HERE/'cost_retry.log','a') as log:
            subprocess.run([sys.executable,'-u','-m','artifacts.experiments.20261103_chooser_history.bench_v2'],
                           stdout=log,stderr=subprocess.STDOUT,check=True,timeout=370)
    identity={f'{arm}_{seed}':json.loads((HERE/f'{arm}_{seed}_identity.json').read_text()) for arm in ARMS for seed in SEEDS}
    for seed in SEEDS:
        for key in ('initial_weight_sha256','initial_batch_rng_sha256','final_batch_rng_sha256'):
            assert len({identity[f'{arm}_{seed}'][key] for arm in ARMS})==1,(key,seed)
    report={};trajectories={}
    for cohort in ('saved','confirmation'):
        report[cohort]={};trajectories[cohort]={}
        for seed in SEEDS:
            z={arm:load(arm,seed,12000,cohort) for arm in ARMS};ref=z['8C'];m=A.metadata(ref);ep=ref['ep']
            for zz in z.values():
                for k in ('ep','t','cls','counter','faced','light'):assert np.array_equal(zz[k],ref[k]),k
            policies={arm:C.policies(zz['C'],zz['S']) for arm,zz in z.items()}
            groups={k:v for k,v in m['strata'].items() if k in ('all','random','day','night','action_move','health_loss',
                    'entity_present','near_entity','age_over128','counter_increment_positive') or k.startswith('class_')}
            part={}
            for group,ix in groups.items():
                if not ix.any():continue
                r=dict(n=int(ix.sum()),episodes=int(len(np.unique(ep[ix]))),arms={},contrasts={})
                cell=np.maximum(ref['counter'],0)
                remote=(C.WIN[None]==ref['counter'][:,None,None]).any(-1)
                contact=(C.WIN[None]==ref['faced'][:,None,None]).any(-1)
                for arm,zz in z.items():
                    cx=ix&(ref['counter']>=0)
                    aa=dict(mse=stats(zz['mse'][ix],ep[ix]),
                            counter_mse=stats(zz['counter_mse'][cx],ep[cx]) if cx.any() else None,
                            counter_error_over_copy=stats(zz['counter_mse'][cx],ep[cx],zz['counter_copy_mse'][cx]) if cx.any() else None,
                            policies={})
                    for pol in ('existing','C_softmax','S_softmax','probability_mixture'):
                        p=policies[arm][pol];pp={}
                        for key,mask in [('remote',remote),('contact',contact)]:
                            if key=='remote' and not (ref['counter'][ix]>=0).any():continue
                            pp[key]=stats((p*mask).sum(-1)[ix],ep[ix])
                        for key in ('any_entity','near_entity','entering_map','changed_hud','terrain_edit'):
                            sel=ix&m['masks'][key].any(-1)
                            if sel.any():pp[key]=stats((p*m['masks'][key][:,C.WIN].any(-1)).sum(-1)[sel],ep[sel])
                        aa['policies'][pol]=pp
                    r['arms'][arm]=aa
                for name,old,new in [('belief_at1','1C','1B'),('belief_at8','8C','8B'),
                                     ('replacement1B_vs8C','8C','1B'),('replacement1B_vs8B','8B','1B')]:
                    d=z[old]['mse']-z[new]['mse']
                    cc=dict(absolute_mse_reduction=stats(d[ix],ep[ix]),
                            percent_mse_reduction=stats(100*d[ix],ep[ix],z[old]['mse'][ix]),
                            counter_percent_reduction=stats(100*(z[old]['counter_mse']-z[new]['counter_mse'])[cx],ep[cx],z[old]['counter_mse'][cx]) if cx.any() else None,
                            targets={},policies={})
                    for key in ('any_entity','near_entity','terrain','terrain_edit','changed_hud','player'):
                        mask=m['masks'][key][ix]
                        olderr=(z[old]['cell_mse'][ix]*mask).sum(-1)
                        newerr=(z[new]['cell_mse'][ix]*mask).sum(-1)
                        if mask.any():cc['targets'][key]=stats(100*(olderr-newerr),ep[ix],olderr)
                    for pol in ('existing','C_softmax','S_softmax','probability_mixture'):
                        pd=policies[new][pol]-policies[old][pol];pp={}
                        for key,mask in [('remote',remote),('contact',contact)]:
                            if key=='remote' and not (ref['counter'][ix]>=0).any():continue
                            pp[key]=stats((pd*mask).sum(-1)[ix],ep[ix])
                        for key in ('any_entity','near_entity','entering_map','changed_hud','terrain_edit'):
                            sel=ix&m['masks'][key].any(-1)
                            if sel.any():pp[key]=stats((pd*m['masks'][key][:,C.WIN].any(-1)).sum(-1)[sel],ep[sel])
                        cc['policies'][pol]=pp
                    r['contrasts'][name]=cc
                inter=(z['1C']['mse']-z['1B']['mse'])-(z['8C']['mse']-z['8B']['mse'])
                r['interaction_absolute_mse']=stats(inter[ix],ep[ix]);part[group]=r
            report[cohort][str(seed)]=part
            tt={}
            for step in (0,1000,4000,8000,12000):
                st={arm:load(arm,seed,step,cohort) for arm in ARMS};sres={}
                for arm,zz in st.items():
                    p=C.policies(zz['C'],zz['S']);aa={}
                    for cl in ('drink','wood','sapling','stone','coal','iron','random'):
                        ix=zz['cls']==cl
                        if not ix.any():continue
                        remote=(C.WIN[None]==zz['counter'][:,None,None]).any(-1)
                        aa[cl]=dict(mse=float(zz['mse'][ix].mean()),
                                    counter_error_over_copy=(float(zz['counter_mse'][ix].sum()/max(zz['counter_copy_mse'][ix].sum(),1e-20)) if cl!='random' else None),
                                    remote={pol:float((prob*remote).sum(-1)[ix].mean()) for pol,prob in p.items()})
                    sres[arm]=aa
                tt[str(step)]=sres
            trajectories[cohort][str(seed)]=tt
    C.write(HERE/'COMPARISONS.json',report);C.write(HERE/'TRAJECTORIES.json',trajectories)
    C.write(HERE/'IDENTITY.json',dict(arms=identity,matching_initialization_and_batch_streams=True))
    # Keep the readable summary short; the paired JSON retains every fixed reading.
    def f(q,scale=1):
        return 'NA' if q is None else f"{q['mean']*scale:.2f} [{q['low']*scale:.2f}, {q['high']*scale:.2f}]"
    lines=['# One-frame history / native-belief ablation','',
           'All four arms completed12k updates with the same schedule, two head seeds, one frozen F1.5 world. '
           'Positive reductions mean lower factual token MSE; policy changes are probability points. '
           'Intervals are episode-clustered, conditional on these fits. No world-learning or production-adoption claim.','',
           '| Fresh-cohort slice | n | Belief at1 A / B (% MSE reduction) | Belief at8 A / B | 1B versus8C A / B |',
           '|---|---:|---|---|---|']
    for group in ('all','random','action_move','near_entity','class_drink','class_wood','class_stone','class_coal','class_iron'):
        aa,bb=[report['confirmation'][str(s)][group] for s in SEEDS]
        values=[]
        for key in ('belief_at1','belief_at8','replacement1B_vs8C'):
            values.append(f(aa['contrasts'][key]['percent_mse_reduction'])+' / '+f(bb['contrasts'][key]['percent_mse_reduction']))
        lines.append(f"| {group} | {aa['n']} | "+' | '.join(values)+' |')
    lines+=['','| Fresh combined gaze: remote counter coverage | 1C A/B | 1B A/B | 8C A/B | 8B A/B |',
            '|---|---|---|---|---|']
    for cl in ('drink','wood','sapling','stone','coal','iron'):
        vals=[]
        for arm in ARMS:
            vals.append(' / '.join(f"{report['confirmation'][str(s)]['class_'+cl]['arms'][arm]['policies']['existing']['remote']['mean']*100:.1f}%" for s in SEEDS))
        lines.append('| '+cl+' | '+' | '.join(vals)+' |')
    lines+=['','Full grounded error/coverage/interaction: COMPARISONS.json. Learning trajectories: TRAJECTORIES.json. '
            'Cost: COST_V2.json (original failed strict compaction check preserved in COST.json and detail logs). '
            'No adaptive weighting or production world run launched.']
    (HERE/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    pins={str(p.relative_to(C.ROOT)):C.sha(p) for p in (HERE/'bench_v2.py',HERE/'report.py')}
    C.write(HERE/'POSTPROCESS_PINS.json',dict(files=pins))
    print('all paired readings complete',flush=True)


if __name__=='__main__':main()
