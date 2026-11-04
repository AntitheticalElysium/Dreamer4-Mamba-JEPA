"""Calibrated semantic proper scores, paired by episode, not fine-token loss."""
import json,sys
from pathlib import Path
import numpy as np
import torch
HERE=Path(__file__).resolve().parent

def softmax(x):
    x=x-x.max(-1,keepdims=True);p=np.exp(x);return p/p.sum(-1,keepdims=True)

def interval(diff,episode,mask):
    e=episode[mask];v=diff[mask];ids=np.unique(e)
    if not len(ids):return None
    sums=np.bincount(e,weights=v,minlength=400)[ids];counts=np.bincount(e,minlength=400)[ids]
    rng=np.random.default_rng(617800);ix=rng.integers(len(ids),size=(5000,len(ids)))
    bs=sums[ix].sum(1)/counts[ix].sum(1)
    return dict(mean=float(v.mean()),episode_bootstrap_95=np.quantile(bs,[.025,.975]).tolist(),queries=len(v),episodes=len(ids))

def main():
    torch.set_num_threads(2);q=torch.load(HERE/'dev_queries.pt',map_location='cpu',weights_only=False);meta=dict(np.load(HERE/'query_meta.npz'))
    assert np.array_equal(q['queries'].numpy(),meta['query']);e=q['episode'].numpy();y=q['kind'].numpy();h=q['heading'].numpy();night=q['night'].numpy()
    strong='--strong' in sys.argv;control=json.loads((HERE/'coarse_control.json').read_text()) if strong else None
    readers=torch.load(HERE/'readers.pt',map_location='cpu',weights_only=False);allp={};scores={}
    if strong:readers.update(torch.load(HERE/'coarse_control_readers.pt',map_location='cpu',weights_only=False))
    for key,r in readers.items():
        x=torch.load(HERE/f'dev_logits_{key}.pt',map_location='cpu',weights_only=False).float().numpy();p={}
        for task,s in [('kind',slice(0,5)),('heading',slice(5,9))]:
            c=r['calibration'][task];p[task]=softmax(x[:,s]/np.exp(c['log_temperature'])+np.array(c['bias']))
        pk=p['kind'];ph=p['heading'];pt=pk[:,[1,3,4]].sum(-1);yt=np.isin(y,[1,3,4]).astype(float)
        scores[key]=dict(kind_nll=-np.log(pk[np.arange(len(y)),y].clip(1e-12)),kind_brier=((pk-np.eye(5)[y])**2).sum(-1),
            threat_nll=-yt*np.log(pt.clip(1e-12,1))-(1-yt)*np.log((1-pt).clip(1e-12,1)),threat_brier=(pt-yt)**2,
            heading_nll=-np.log(ph[np.arange(len(h)),h.clip(min=0)].clip(1e-12)),heading_brier=((ph-np.eye(4)[h.clip(min=0)])**2).sum(-1))
        allp[key]={k:torch.from_numpy(v.astype(np.float32)) for k,v in p.items()}
    torch.save(allp,HERE/('dev_probabilities_strong.pt' if strong else 'dev_probabilities.pt'))
    rad=abs(meta['cell']//9-3)+abs(meta['cell']%9-4);answer={}
    for task in ['kind','threat','heading']:
        answer[task]={}
        for cohort,cm in [('discovery',e<250),('confirmation',e>=250)]:
            for region,rm in [('all',np.ones(len(e),bool)),('night',night),('night_far',night&(rad>=3))]:
                mask=cm&rm&((h>=0) if task=='heading' else True);k=cohort+'/'+region;res={}
                for mode in ['fine','pixels']:
                    paired={};delta={}
                    for metric in ['nll','brier']:
                        dd=[]
                        for family in ['flat','split']:
                            for seed in [0,1]:
                                base=control[f'{family}_{seed}_coarse_tsp']['chosen_by_task']['heading' if task=='heading' else 'kind'] if strong else f'{family}_{seed}_coarse'
                                test=f'{family}_{seed}_{mode}';d=scores[base][task+'_'+metric]-scores[test][task+'_'+metric]
                                paired.setdefault(family+'_'+str(seed),{})[metric]=interval(d,e,mask);dd.append(d)
                        delta[metric]=interval(np.mean(dd,axis=0),e,mask)
                    bases=[control[f'{family}_{seed}_coarse_tsp']['chosen_by_task']['heading' if task=='heading' else 'kind'] if strong else f'{family}_{seed}_coarse' for family in ['flat','split'] for seed in [0,1]]
                    mean_base={metric:float(np.mean([scores[base][task+'_'+metric][mask].mean() for base in bases])) for metric in ['nll','brier']}
                    gate=all(x['nll']['mean']>0 for x in paired.values()) and delta['nll']['episode_bootstrap_95'][0]>0 and delta['brier']['episode_bootstrap_95'][0]>0
                    res[mode]=dict(base_mean=mean_base,benefit=delta,per_reader=paired,replicated_gain=gate)
                answer[task][k]=res
    (HERE/('semantic_results_strong.json' if strong else 'semantic_results.json')).write_text(json.dumps(answer,indent=2)+'\n')
    for task,a in answer.items():
        for region in ['night','night_far']:
            for mode in ['fine','pixels']:
                r=a['confirmation/'+region][mode];print(task,region,mode,'gain',r['replicated_gain'],'NLL',r['benefit']['nll'],'Brier',r['benefit']['brier'],flush=True)

if __name__=='__main__':main()
