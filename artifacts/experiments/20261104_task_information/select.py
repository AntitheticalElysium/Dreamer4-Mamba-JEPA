"""The predeclared source-only rule, before simulator consequence outcomes."""
import json
from pathlib import Path
import numpy as np
import torch
HERE=Path(__file__).resolve().parent

def main():
    torch.set_num_threads(2);meta=dict(np.load(HERE/'query_meta.npz'));p=torch.load(HERE/'dev_probabilities.pt',map_location='cpu',weights_only=False)
    results=json.loads((HERE/'semantic_results.json').read_text());out=[];counts={};n=len(meta['query']);index=np.arange(n)
    for task in ['kind','heading']:
        truth=meta[task];keys=[f'{f}_{s}' for f in ['flat','split'] for s in [0,1]]
        for cohort,cm in [('discovery',meta['episode']<250),('confirmation',meta['episode']>=250)]:
            region=cohort+'/night';mode='fine' if results[task][region]['fine']['replicated_gain'] else 'pixels' if results[task][region]['pixels']['replicated_gain'] else None
            if mode is None:counts[task+'/'+cohort]=dict(eligible=0,reason='no replicated gain for task');continue
            a=np.stack([p[k+'_coarse'][task].numpy()[index,truth.clip(min=0)] for k in keys]);b=np.stack([p[k+'_'+mode][task].numpy()[index,truth.clip(min=0)] for k in keys])
            gain=b-a;ok=cm&(meta['light']<.5)&(meta['time']>=8)&~meta['sleep']&(meta['health']>0)
            ok&=(truth>=0) if task=='heading' else np.isin(truth,[1,3,4])
            ok&=(gain>0).all(0)&(gain.mean(0)>=.15)&(b.mean(0)>=.65)
            ids=np.flatnonzero(ok);counts[task+'/'+cohort]=dict(eligible=len(ids),episodes=len(np.unique(meta['episode'][ids])),mode=mode)
            used=set()
            for i in ids:
                e=int(meta['episode'][i])
                if e in used or len(used)>=12:continue
                used.add(e);r={k:np.asarray(v[i]).item() for k,v in meta.items()};r.update(task=task,mode=mode,cohort=cohort,
                    coarse_true_probability=a[:,i].tolist(),fine_true_probability=b[:,i].tolist(),improvement=gain[:,i].tolist(),row=int(i));out.append(r)
    (HERE/'selection.json').write_text(json.dumps(dict(counts=counts,selected=out),indent=2)+'\n')
    print(json.dumps(counts,indent=2));print('selected',len(out))

if __name__=='__main__':main()
