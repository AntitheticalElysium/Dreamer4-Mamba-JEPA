"""Read-only synthesis of completed F1.6 evidence; no model fitting."""
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[3]
HERE=Path(__file__).resolve().parent
NEW=ROOT/'artifacts/experiments/20261102_p1f16'
OLD=ROOT/'artifacts/experiments/20261101_p1f15'

def read(path):return json.loads(path.read_text())
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def maxdiff(a,b):
    if isinstance(a,dict):
        assert a.keys()==b.keys()
        return max([maxdiff(a[k],b[k]) for k in a] or [0])
    if isinstance(a,list):
        assert len(a)==len(b)
        return max([maxdiff(x,y) for x,y in zip(a,b)] or [0])
    if isinstance(a,(int,float)):return abs(a-b)
    assert a==b
    return 0


def main():
    out={};pin=read(NEW/'SOURCE_PINS.json')
    changed=[f for f,h in pin['files'].items() if sha(ROOT/f)!=h]
    assert not changed,changed
    actual=sha(NEW/'run/ckpt_10.pt');assert actual==read(NEW/'run/s1/results.json')['checkpoint_sha256']
    out['audit']=dict(pinned_sources=len(pin['files']),changed=changed,checkpoint_sha256=actual,selection=read(NEW/'run/s1/selection.json'))
    oldif=read(OLD/'readings/interface_random.json')['checkpoints']['p1f15']['classes']
    newif=read(NEW/'readings/interface_random.json')['checkpoints']['p1f15']['classes']
    out['audit']['baseline_interface_rerun_max_difference']=maxdiff(oldif,newif)
    out['interface']={}
    for policy in ('random','guided'):
        d=read(NEW/f'readings/interface_{policy}.json');events={}
        for k,n in d['classes'].items():
            events[k]={'histories':n,'worlds':{w:d['checkpoints'][w]['classes'][k] for w in ('p1f15','p1f16')},
                       'paired_difference':d['differences']['p1f16-p1f15'][k]}
        out['interface'][policy]=events
    out['s1']={}
    out['decoders']={}
    for world,path in (('p1f15',OLD),('p1f16',NEW)):
        d=read(path/'run/s1/results.json');s={}
        s['C2ST']=d['s1_3']['c2st_generated']['accuracy'];s['counts']=d['s1_3']['reader_counts']
        s['calibration']=d['s1_3']['calibration'];s['tsp']=d['s1_4']['tsp_nats_per_pixel']
        s['retention']={light:{k:float(np.mean([x[light][k]['est'] for x in d['s1_1']['trained']]))
            for k in ('presence_recall','recall_zombies','recall_cows','recall_skeletons','recall_arrows')} for light in ('day_r1_2','night_r1_2','night_r1_4')}
        s['belief']={key:dict(achievement_AP=float(np.mean([v['ap'] for z in arr for v in z['achievements']])),
            HUD_exact=float(np.mean([v['est'] for z in arr for v in z['hud_exact']])),
            HUD_by_quantity=np.mean([[v['est'] for v in z['hud_exact']] for z in arr],axis=0).tolist()) for key,arr in d['s1_2'].items()}
        s['monitor']=[dict(checkpoint=x['checkpoint'],**x['monitor']) for x in map(json.loads,(path/'run/log.jsonl').read_text().splitlines()) if 'monitor' in x]
        out['s1'][world]=s
        out['decoders'][world]=read(path/f'readings/render_{world}.json')['trained']
    out['counter_transitions']={}
    p=NEW/'readings/eval_m05'
    for route in ('guided','periphery'):
        for kind in ('drink','wood'):
            arms={w:dict(np.load(p/f'{label}_{route}_{kind}.npz')) for w,label in (('p1f15','base'),('p1f16','p1f16_1500'))}
            a,b=arms.values();assert np.array_equal(a['start'],b['start'])
            keep=a['use'].astype(bool)&b['use'].astype(bool)&a['eligible'].astype(bool)&b['eligible'].astype(bool)
            rows=[]
            for v in sorted(np.unique(a['start'][keep])):
                mask=keep&(a['start']==v)
                rows.append(dict(start=int(v),samples=int(mask.sum()),histories=int(len(np.unique(a['row_ctx'][mask]))),
                    worlds={w:{k:float(z[k][mask].mean()) for k in ('success','unknown','copy')} for w,z in arms.items()}))
            out['counter_transitions'][route+'_'+kind]=rows
    out['consistency']={}
    for world,path in (('p1f15',OLD),('p1f16',NEW)):
        z=np.load(path/'readings/consistency/consistency.npz');er=z['er'];active=er.mean(1)>.05;ranges=er.max(1)-er.min(1)
        out['consistency'][world]=dict(summary=read(path/'readings/consistency/consistency.json'),
            active_contexts=int(active.sum()),active_reward_range_median=float(np.median(ranges[active])),active_reward_range_max=float(ranges[active].max()))
    out['local']=read(HERE/'local_summary.json');out['history_cost']=read(HERE/'history_cost.json')
    out['source_checks']=read(HERE/'source_checks.json');out['recipe']=read(HERE/'recipe_check.json')
    files=[NEW/'readings/interface_random.json',NEW/'readings/interface_guided.json',NEW/'run/s1/results.json',
           NEW/'readings/uptake/uptake.json',NEW/'readings/consistency/consistency.npz',NEW/'readings/codex_summary.json',
           p/'results.json',p/'readings_p1f16_1500-base.json',NEW/'readings/locus/p1f16.npz',
           HERE/'local_summary.json',HERE/'local_uptake.py',HERE/'history_cost.json',HERE/'history_cost.py',Path(__file__)]
    files.extend([HERE/'source_checks.py',HERE/'source_checks.json',HERE/'PLAN.md',HERE/'recipe_check.json'])
    out['input_sha256']={str(f.relative_to(ROOT)):sha(f) for f in files}
    (HERE/'ANALYSIS.json').write_text(json.dumps(out,indent=2))
    print('Saved ANALYSIS.json; source checks',out['audit']['pinned_sources'],'baseline max diff',out['audit']['baseline_interface_rerun_max_difference'])


if __name__=='__main__':main()
