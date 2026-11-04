"""Paired endpoint uncertainty for the saved chronological chooser replay."""
from pathlib import Path
import importlib,json
import numpy as np
P=importlib.import_module('artifacts.experiments.20261103_chooser_development.common')
A=importlib.import_module('artifacts.experiments.20261103_gaze_belief.analyze')
HERE=Path(__file__).resolve().parent

def main():
    out={}
    for cohort in ['saved','confirmation']:
        z=[dict(np.load(HERE/f'ck{c:02d}/{cohort}.npz')) for c in [1,10]]
        assert all(np.array_equal(z[0][k],z[1][k]) for k in ['ep','t','cls','counter','increment','counter_before'])
        entry={}
        for cl in np.unique(z[0]['cls']):
            valid=(z[0]['cls']==cl)&(z[0]['counter']>=0)
            if not valid.any():continue
            entry[str(cl)]={}
            for group,sel in [('attempt',valid),('increment',valid&z[0]['increment']),
                ('first_increment',valid&z[0]['increment']&(z[0]['counter_before']==0)),
                ('later_increment',valid&z[0]['increment']&(z[0]['counter_before']>0))]:
                if not sel.any():continue
                ep=z[0]['ep'][sel];ids,inv=np.unique(ep,return_inverse=True)
                # Same episode resamples for all endpoints and metrics.
                rng=np.random.default_rng(621005);bs=rng.integers(len(ids),size=(2000,len(ids)))
                boot={};means={}
                for name,v in zip(['ck1','ck10'],z):
                    dd=np.bincount(inv,weights=v['counter_copy_error'][sel])
                    for metric in ['counter_error','counter_C','counter_noop_copy_error','counter_alignment']:
                        nn=np.bincount(inv,weights=v[metric][sel]);den=dd[bs].sum(1)
                        vals=nn[bs].sum(1)/den
                        boot[name+'_'+metric]=vals;means[name+'_'+metric]=float(nn.sum()/dd.sum())
                result={}
                for metric in ['counter_error','counter_C','counter_noop_copy_error','counter_alignment']:
                    values={name:boot[name+'_'+metric] for name in ['ck1','ck10']}
                    values['difference']=values['ck10']-values['ck1']
                    result[metric]={name:dict(mean=(means['ck10_'+metric]-means['ck1_'+metric]) if name=='difference' else means[name+'_'+metric],
                        ci95=np.quantile(v,[.025,.975]).tolist()) for name,v in values.items()}
                remote=(P.WIN[None]==z[0]['counter'][:,None,None]).any(-1)
                probs=[(v['prob']*remote).sum(-1) for v in z]
                result['remote_probability']=dict(ck1=A.estimates(probs[0][sel],np.ones(sel.sum()),ep)[0],
                    ck10=A.estimates(probs[1][sel],np.ones(sel.sum()),ep)[0],
                    difference=A.estimates((probs[1]-probs[0])[sel],np.ones(sel.sum()),ep)[0])
                result.update(n=int(sel.sum()),episodes=len(ids));entry[str(cl)][group]=result
        out[cohort]=entry
    (HERE/'HANDOVER.json').write_text(json.dumps(out,indent=2)+'\n')
    for cl,v in out['confirmation'].items():
        if 'increment' not in v:continue
        r=v['increment'];print(cl,'n',r['n'],'error',r['counter_error']['ck1'],r['counter_error']['ck10'],'C',r['counter_C']['ck1'],r['counter_C']['ck10'],flush=True)

if __name__=='__main__':main()
