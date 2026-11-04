"""Reproduce the bounded native-TSP and explicitly post-result cohort summaries."""
import importlib,json
import numpy as np
A=importlib.import_module('artifacts.experiments.20261102_p1f15_readiness.analyze_uptake')

def main():
    p=A.HERE/'uptake_local';z=dict(np.load(p/'future_tsp.npz'));dd=[];bb=[];ep=[]
    for i in range(len(z['episode'])):
        w=A.weights(z['useful'][i],~z['held_mob'][i])
        if w is None:continue
        a,b=w;x=z['pixel_mse'][i].mean(-1);dd.append((a-b)@x);bb.append(b@x);ep.append(z['episode'][i])
    r={'difference':A.estimate(dd,ep),'control_loss':A.estimate(bb,ep),
       'fractional_difference':A.estimate(np.array(dd)/bb,ep),
       'scope':'native TSP predicts next crop given next real periphery and the varied post-action belief; no direct past-glimpse path'}
    (p/'future_tsp_summary.json').write_text(json.dumps(r,indent=1))
    z=dict(np.load(p/'raw.npz'));L=dict(np.load('artifacts/eda/s0/s0_dev_labels.npz'))
    so=np.r_[0,np.cumsum(L['ep_T']+1)];ao=np.r_[0,np.cumsum(L['ep_T'])];rows=[]
    for i in np.flatnonzero(z['useful'].any(1)):
        e=int(z['episode'][i]);t=int(z['t'][i]);src=np.unique(A.WIN[z['useful_cells'][i]])
        kinds=z['truth'][i,src];never=[];recent=[]
        for k in kinds:
            nm=('zombies','cows','skeletons','arrows')[int(k)-1]
            pp=L['s_pos'][so[e]:so[e]+t];op=L['s_'+nm+'_p'][so[e]:so[e]+t];mask=L['s_'+nm+'_m'][so[e]:so[e]+t]
            rel=op-pp[:,None];vis=(mask&(abs(rel[...,0])<=3)&(abs(rel[...,1])<=4)).any(-1)
            never.append(not vis.any());recent.append(not vis[-8:].any())
        a,b=A.weights(z['useful'][i],~z['held_mob'][i]);x=z['loss'][i].mean((0,1))[:,1]
        rows.append({'i':int(i),'episode':e,'t':t,'exogenous':bool(L['t_exogenous'][ao[e]+t]),
            'any_kind_never_seen':bool(any(never)),'any_kind_not_seen_last8':bool(any(recent)),
            'refit_P_also_misses':bool((~z['stage_correct'][i,0,0,src]).any()),
            'fractional_loss_difference':float(((a-b)@x)/(b@x))})
    r={}
    for k in ('all','exogenous','any_kind_never_seen','any_kind_not_seen_last8','refit_P_also_misses'):
        q=rows if k=='all' else [a for a in rows if a[k]]
        r[k]=A.estimate([a['fractional_loss_difference'] for a in q],[a['episode'] for a in q])
    (p/'cohorts_posthoc.json').write_text(json.dumps({'scope':'Post-result descriptive analysis, no new model calls; kind visibility is not identity or proof of absence from belief',
        'summary':r,'rows':rows},indent=1))

if __name__=='__main__':main()
