"""Read archived matched-gaze interface and query-invariance results only."""
import hashlib
import json
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent
EXP=HERE.parent
SOURCES=[('original','20261030_p1_guided','base'),('P1-F','20261030_p1_guided','p1f'),
         ('beta-TSP','20261101_beta_tsp','beta'),('F1.5','20261101_p1f15','p1f15'),
         ('F1.6','20261103_p1f2','p1f16'),('F2','20261103_p1f2','p1f2')]
CLASSES=['risky','adjacent','ach_COLLECT_COAL','ach_COLLECT_IRON','ach_DEFEAT_ZOMBIE','ach_WAKE_UP']


def main():
    out={'comparisons':{},'invariance':{}};references={}
    for policy in ['random','guided']:
        out['comparisons'][policy]={}
        for label,folder,name in SOURCES:
            p=EXP/folder/'readings'/f'interface_{policy}.json'
            j=json.loads(p.read_text());q=j['checkpoints'][name]
            r={}
            for c in CLASSES:
                z=q['classes'][c]
                # SAME recorded target cohort, rates and history count.
                ref=(z['histories'],z['recorded'])
                if c in references:assert ref==references[c],(label,policy,c)
                else:references[c]=ref
                r[c]={'histories':z['histories'],'futures':z['futures'],
                      **{s:{k:v for k,v in z[s].items() if k in ['term_p','neg_p','unlock_p','event']}
                         for s in ['recorded','rec_head','gen']}}
            out['comparisons'][policy][label]={'source':str(p.relative_to(EXP.parent.parent)),
                'source_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),
                'checkpoint_sha256':q['sha256'],'classes':r}
    for label,folder in [('F1.5','20261101_p1f15'),('F2','20261103_p1f2')]:
        p=EXP/folder/'readings/consistency/consistency.npz';z=np.load(p)
        m=z['er'].mean(1);selected=m>.05;er=z['er'][selected]
        a=er.min(1);b=er.max(1)
        out['invariance'][label]={'contexts':len(m),'rewarding_contexts':int(selected.sum()),
            'criterion':'mean across queries > 0.05, same descriptive criterion as F1.5 review',
            'median_query_range':float(np.median(b-a)), 'maximum_query_range':float(np.max(b-a)),
            'all_continuations_one':bool((z['pc']==1).all()),
            'per_context':[dict(episode=int(z['ep'][i]),t=int(z['t'][i]),
                query_mean=float(m[i]),minimum=float(z['er'][i].min()),maximum=float(z['er'][i].max()),
                maximum_reported_mcse=float(z['er_se'][i].max())) for i in np.where(selected)[0]]}
    (HERE/'COMPARISONS.json').write_text(json.dumps(out,indent=2)+'\n')
    for pol,worlds in out['comparisons'].items():
        for label,z in worlds.items():
            r=z['classes']['risky'];print(pol,label,'death',r['gen']['term_p']['est'],
                'damage',r['gen']['neg_p']['est'],'coal',z['classes']['ach_COLLECT_COAL']['gen']['unlock_p']['est'],
                'iron',z['classes']['ach_COLLECT_IRON']['gen']['unlock_p']['est'])
    print('invariance',{k:{a:b for a,b in v.items() if a!='per_context'} for k,v in out['invariance'].items()})


if __name__=='__main__':main()
