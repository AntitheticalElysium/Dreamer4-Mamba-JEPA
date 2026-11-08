"""Descriptive held-out chooser checks; no policy selection or fitting."""
import json,importlib
from pathlib import Path
import numpy as np
from scipy.stats import rankdata
HERE=Path(__file__).resolve().parent
CENTRES=np.array([(r,c) for r in range(1,8) for c in range(1,8)])
WIN=np.array([[(r+dr)*9+c+dc for dr in (-1,0,1) for dc in (-1,0,1)] for r,c in CENTRES])

def correlations(a,b):
    a=rankdata(a,axis=-1);b=rankdata(b,axis=-1);a-=a.mean(-1,keepdims=True);b-=b.mean(-1,keepdims=True)
    d=np.sqrt((a*a).sum(-1)*(b*b).sum(-1));return np.divide((a*b).sum(-1),d,out=np.full(len(a),np.nan),where=d>0)

def describe(v):
    v=np.asarray(v);v=v[np.isfinite(v)]
    return {'mean':float(v.mean()),'p10':float(np.quantile(v,.1)),'p90':float(np.quantile(v,.9)),'n':len(v)} if len(v) else {'n':0}

def main():
    p=HERE/'gaze_checks';out={'reliability':{},'sequential':{},'static':{}}
    for ck in (1,5,10):
        z=np.load(p/f'reliability_{ck:02d}.npz');a=z['scores'];ep=z['rows'][:,0]
        out['reliability'][ck]={}
        for split,mask in (('all',ep>=0),('heldout',ep>=8)):
            x=a[mask];r={}
            for j,key in enumerate(('contingency','surprise','mixture_probability')):
                r[key]={'rank_correlation':describe(correlations(x[:,0,j],x[:,1,j])),
                    'argmax_agreement':float((x[:,0,j].argmax(-1)==x[:,1,j].argmax(-1)).mean()),
                    'flat_fraction':float((np.ptp(x[:,0,j],axis=-1)<1e-12).mean())}
            pr=x[:,:,2];r['mixture_tv_between_draws']=describe(np.abs(pr[:,0]-pr[:,1]).sum(-1)/2)
            r['expected_hud_share']=float(pr[:,:,(WIN>=63).any(-1)].sum(-1).mean())
            r['max_probability']=describe(pr.max(-1).ravel())
            out['reliability'][ck][split]=r
        st=np.load(p/f'static_{ck:02d}.npz')['scores']
        tv=np.abs(st[:,1:,2]-st[:,:1,2]).sum(-1)/2
        out['static'][ck]={'mix_tv_same_world_different_static':describe(tv.ravel()),
            'C_max_absolute_change':float(np.abs(st[:,1:,0]-st[:,:1,0]).max()),
            'S_rank_correlation':describe(correlations(st[:,1:,1].reshape(-1,49),np.repeat(st[:,:1,1],3,axis=1).reshape(-1,49))),
            'note':'C does not read the current frame; invariance to current-frame static is structural, not learned.'}
    z=dict(np.load(p/'sequential.npz'))
    from saccade.eval import s1 as S
    LC=importlib.import_module('artifacts.experiments.20261031_learned_fixation_sweep.locus')
    L=S.first_episodes(dict(np.load(S.LABELS/'s0_dev_labels.npz')),12)
    contexts,face,direction=LC.contexts(L,np.random.default_rng(611650))
    so=np.r_[0,np.cumsum(L['ep_T']+1)]
    for policy in np.unique(z['policy']):
        out['sequential'][str(policy)]={}
        for split,mask in (('all',np.ones(len(z['episode']),bool)),('heldout',z['heldout'])):
            mask=mask&(z['policy']==policy);ind=np.where(mask)[0];maxstreak=[];repeat=[];coverage=[];returns=[]
            for e in np.unique(z['episode'][ind]):
                q=ind[z['episode'][ind]==e];ch=z['chosen'][q];streak=1;longest=1;last={}
                for j,c in enumerate(ch):
                    if j:streak=streak+1 if c==ch[j-1] else 1
                    longest=max(longest,streak)
                    if int(c) in last:returns.append(j-last[int(c)])
                    last[int(c)]=j
                maxstreak.append(longest);repeat.extend((ch[1:]==ch[:-1]).tolist());coverage.append(len(set(ch.tolist()))/49)
            pr=z['prob'][ind]
            r={'frames':len(ind),'episodes':len(maxstreak),'exact_window_repeat_rate':float(np.mean(repeat)),
                'max_streak_per_episode':describe(maxstreak),'window_coverage_fraction':describe(coverage),
                'window_return_gap':describe(returns),'realized_hud_share':float(z['hud'][ind].mean()),
                'mob_covered_share':float(z['mob_covered'][ind].mean()),
                'expected_same_window_probability':float((pr[1:]*pr[:-1]).sum(-1).mean()),
                'max_probability':describe(pr.max(-1)),'effective_windows':describe(1/(pr*pr).sum(-1))}
            for night in (False,True):
                ii=ind[z['night'][ind]==night]
                r['night' if night else 'day']={'frames':len(ii),'hud_share':float(z['hud'][ii].mean()) if len(ii) else None,
                    'mob_covered_share':float(z['mob_covered'][ii].mean()) if len(ii) else None}
            r['attempt_coverage']={}
            for cls,rows in contexts.items():
                if cls=='random':continue
                rowset=set(rows);ii=[i for i in ind if (int(z['episode'][i]),int(z['t'][i])) in rowset]
                counter=LC.ATTEMPTS[cls][1] if cls!='stone_no_tool' else LC.ATTEMPTS['stone'][1]
                if ii:
                    selected=WIN[z['chosen'][ii]]
                    src=np.array([so[z['episode'][i]]+z['t'][i]-1 for i in ii])
                    faced=(3+direction[src,0])*9+4+direction[src,1]
                    r['attempt_coverage'][cls]={'frames':len(ii),'counter':float((selected==counter).any(-1).mean()),
                        'faced':float((selected==faced[:,None]).any(-1).mean())}
                else:r['attempt_coverage'][cls]={'frames':0}
            out['sequential'][str(policy)][split]=r
    (p/'summary.json').write_text(json.dumps(out,indent=1));print(json.dumps(out,indent=1))

if __name__=='__main__':main()
