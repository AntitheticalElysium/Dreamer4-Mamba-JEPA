"""Read only the bounded campaign's saved arrays. Episode bootstrap, no fit/selection."""
import json
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent
CENTRES=np.array([(r,c) for r in range(1,8) for c in range(1,8)])
WIN=np.array([[(r+dr)*9+c+dc for dr in (-1,0,1) for dc in (-1,0,1)] for r,c in CENTRES])


def weights(info,none):
    # Controls match the informative crop's row: identical HUD overlap, no hidden row reweighting.
    usable=info & np.array([(none & (CENTRES[:,0]==r)).any() for r in CENTRES[:,0]])
    if not usable.any():return None
    a=usable/usable.sum();b=np.zeros(49)
    for r in np.unique(CENTRES[usable,0]):
        mask=none&(CENTRES[:,0]==r);b[mask]=a[CENTRES[:,0]==r].sum()/mask.sum()
    assert np.allclose(a.sum(),1) and np.allclose(b.sum(),1)
    return a,b


def estimate(v,episode):
    v=np.asarray(v,float);ep=np.asarray(episode)
    ok=np.isfinite(v);v=v[ok];ep=ep[ok]
    if not len(v):return {'n':0}
    es=np.unique(ep);gen=np.random.default_rng(611501)
    sums=np.array([v[ep==e].sum() for e in es]);counts=np.array([(ep==e).sum() for e in es])
    i=gen.integers(0,len(es),(2000,len(es)));means=sums[i].sum(1)/counts[i].sum(1)
    return {'mean':float(v.mean()),'ci95':np.quantile(means,[.025,.975]).tolist(),'n':len(v),'episodes':len(es)}


def main():
    z=dict(np.load(HERE/'uptake_local/raw.npz'));res={};N=len(z['episode'])
    valid=[];diff=[];base=[];stages=[];rms=[];strong=[];noisevar=[];stagebase=[]
    for i in range(N):
        w=weights(z['useful'][i],~z['held_mob'][i])
        if w is None:continue
        a,b=w;valid.append(i)
        loss=z['loss'][i] # two sets, eight noise levels, 49 windows, map/local
        diff.append(np.einsum('snwg,w->sng',loss,a-b));base.append(np.einsum('snwg,w->sng',loss,b))
        sources=np.unique(WIN[z['useful_cells'][i]])
        st=[];sb=[]
        for cell in sources:
            info=np.any(z['useful_cells'][i] & (WIN==cell),-1)
            wc=weights(info,~z['held_mob'][i])
            if wc is None:continue
            aa,bb=wc
            st.append(np.einsum('kw,w->k',z['stage_prob'][i,:,:,cell],aa-bb))
            sb.append(np.einsum('kw,w->k',z['stage_prob'][i,:,:,cell],bb))
        stages.append(np.mean(st,0));stagebase.append(np.mean(sb,0))
        strong.append(bool((~z['stage_correct'][i,0,0,sources]).any()))
        rr={}
        for key in ('emb','g_input','mask_input','corrected','predicted','memory'):
            x=z[key][i];d=np.einsum('wd,w->d',x,a-b)
            rr[key]=float(np.sqrt(np.mean(d*d)))
        pred=z['prediction'][i]
        # Draws are shared over gaze; raw magnitude is sensitivity, not a forecast score.
        d=np.einsum('swcd,w->scd',pred,a-b)
        rr['local_first_call_prediction']=float(np.sqrt(np.mean(d[:,z['local'][i]]**2)))
        noisevar.append(float(np.sqrt(np.mean((pred[0]-pred[1])**2))))
        rms.append(rr)
    ix=np.array(valid);diff=np.stack(diff);base=np.stack(base);stages=np.stack(stages);sb=np.stack(stagebase)
    strong=np.array(strong)
    for name,mask in (('all',np.ones(len(ix),bool)),('night',z['night'][ix]),('day',~z['night'][ix]),('refit_periphery_also_misses',strong)):
        ep=z['episode'][ix][mask];r={'frames':int(mask.sum())}
        for region,j in (('whole_map',0),('source_local',1)):
            r[region]={}
            for band,sl in (('all_noise',slice(None)),('highest_two',slice(-2,None))):
                dd=diff[mask,:,sl,j].mean((1,2));bb=base[mask,:,sl,j].mean((1,2))
                r[region][band]={'difference':estimate(dd,ep),'control_loss':estimate(bb,ep),
                    'fractional_difference':estimate(dd/bb,ep)}
        r['stage_true_kind_probability_gain']={k:estimate(stages[mask,j],ep) for j,k in enumerate(('p','embedding','corrected_belief','predicted_belief','projected_memory'))}
        r['stage_control_true_kind_probability']={k:estimate(sb[mask,j],ep) for j,k in enumerate(('p','embedding','corrected_belief','predicted_belief','projected_memory'))}
        r['rms_legitimate_crop_contrast']={k:estimate([rms[j][k] for j in np.flatnonzero(mask)],ep) for k in rms[0]}
        r['first_call_noise_draw_rms']=estimate(np.asarray(noisevar)[mask],ep)
        res[name]=r
    res['scope']={'all_frames':N,'fine_positive_control_frames':int(z['useful'].any(1).sum()),'row_matched_control_frames':len(ix),
        'source_local_definition':'radius-two source neighborhood, no future-outcome selection',
        'instrument_limit':'bounded readers measure accessibility; a misclassification is not an information-theoretic absence'}
    (HERE/'uptake_local/summary.json').write_text(json.dumps(res,indent=1));print(json.dumps(res,indent=1))

if __name__=='__main__':main()
