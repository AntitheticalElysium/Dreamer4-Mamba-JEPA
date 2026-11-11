"""Report the single declared rollout and an observed-history fatigue reference."""
import os
os.environ.setdefault('JAX_PLATFORMS','cpu')
import importlib,json
from pathlib import Path
import numpy as np
import torch

CV=importlib.import_module('artifacts.experiments.20261106_determinant_curves.curves')
DC=importlib.import_module('artifacts.experiments.20261107_f4_followups.danger_checks')
HERE=Path(__file__).resolve().parent

def interval(x):
    """x [source, draws]; preserve all sibling draws when bootstrapping sources."""
    avg=np.asarray(x,float).mean(1);r=np.random.default_rng(110711)
    boot=avg[r.integers(0,len(avg),(2000,len(avg)))].mean(1)
    return dict(mean=float(avg.mean()),ci95=np.percentile(boot,[2.5,97.5]).tolist())

def first(m,H):
    return np.where(m.any(2),m.argmax(2)+1,H+1)

def clock():
    L=dict(np.load(CV.LABELS/'s0_dev_labels.npz'));T=CV.transitions(L)
    off=np.r_[0,np.cumsum(L['ep_T']+1)];f=np.zeros(len(L['s_sleep']),np.float32)
    for e,N in enumerate(L['ep_T']):
        for t in range(1,int(N)+1):
            s=off[e]+t
            z=min(f[s-1]-1,0) if L['s_sleep'][s] else f[s-1]+1
            f[s]=0 if z>30 or z < -10 else z
    quiet=DC.quiet(L,T);en=T['body'][:,3];g=T['g']
    out={'input':'sleep observations up to and including current state, reset fatigue=0; no future energy used',
         'scope':'rule-based evaluation control with simulator-label sleep observations, whose real-frame readability is separately validated'}
    for name,m,pred,y in [
        ('asleep_energy_up',quiet&T['asleep']&(en<9),np.minimum(f[g]-1,0)<-10,T['true_body'][:,3]==en+1),
        ('awake_energy_down',quiet&~T['asleep']&(en>0)&(T['action']!=CV.IF.C.Action.SLEEP.value),f[g]+1>30,T['true_body'][:,3]==en-1)]:
        out[name]={'transitions':int(m.sum()),'positive_events':int(y[m].sum()),'errors':int((pred[m]!=y[m]).sum())}
    bank=torch.load(HERE/'bank.pt',map_location='cpu',weights_only=False)
    est=np.array([f[off[r['e']]+r['t0']] for r in bank['rows']])
    truth=np.array([r['source_fatigue'] for r in bank['rows']])
    assert np.array_equal(est,truth)
    out['independent_simulator_counter_check']={'sources':len(est),'max_fatigue_error':float(abs(est-truth).max())}
    return out

def main():
    torch.set_num_threads(1)
    z=torch.load(HERE/'run.pt',map_location='cpu',weights_only=False)
    a,b=z['generated'],z['truth'];H=z['H']
    active=np.concatenate([np.ones((*b.shape[:2],1),bool),np.cumprod(~b[:,:,:-1,3].astype(bool),2).astype(bool)],2)
    times={};result={'sources':len(z['rows']),'draws_per_source':z['K'],'horizon':H,
        'source_cohort':'asleep, energy8, health>=7, food/drink>=3; initially no close hostiles/arrows; NOOP; uniform fixation',
        'prefix_sham_rms_max':z['prefix_sham_max'],'readers_valid':z['readers_valid'],'reader_stress':z['reader_stress']}
    for name,q,ac,d in [('F4',a,a[:,:,:,3].astype(bool),a[:,:,:,2].astype(bool)),('simulator',b,active,b[:,:,:,3].astype(bool))]:
        result[name]={};times[name]={}
        for event,m in [('energy9',ac&(q[:,:,:,0]>=9)),('awake',ac&(q[:,:,:,1]<.5)&~d),('terminal',ac&d)]:
            t=first(m,H);times[name][event]=t
            result[name][event]={'probability_by_step':{str(k):interval(t<=k) for k in (12,16,32,48)},
                'median_step_or_Hplus1_if_unobserved':float(np.median(t)),
                'restricted_mean_waiting_steps':interval(np.minimum(t,H)),
                'censored_fraction':float((t>H).mean())}
    r=np.random.default_rng(110711);ix=r.integers(0,len(z['rows']),(2000,len(z['rows'])))
    for event in ('energy9','awake'):
        ga=np.minimum(times['F4'][event],H).mean(1);gb=np.minimum(times['simulator'][event],H).mean(1)
        ratio=ga[ix].mean(1)/gb[ix].mean(1)
        result[event+'_restricted_wait_ratio']={'mean':float(ga.mean()/gb.mean()),'ci95':np.percentile(ratio,[2.5,97.5]).tolist(),
            'not_an_unrestricted_duration_ratio':True}
    m=(a[:,:,:-1,0]>=9)&(a[:,:,:-1,1]>.5)&(a[:,:,:-1,3]>.5)&(a[:,:,1:,3]>.5)
    result['generated_energy9_rule']={'eligible_transitions':int(m.sum()),'next_awake_fraction':float((a[:,:,1:,1][m]<.5).mean())}
    result['clock_reference']=clock()
    (HERE/'SUMMARY.json').write_text(json.dumps(result,indent=2)+'\n')
    # Standalone scientific figure, all siblings retained in source bootstrap.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(10,3.8),sharey=True)
    x=np.arange(1,H+1)
    for ax,event,title in zip(axes,['energy9','awake'],['Energy reaches 9','Agent wakes (alive)']):
        for name,color in [('simulator','#34875b'),('F4','#486ac4')]:
            t=times[name][event];v=(t[:,:,None]<=x).mean(1);curves=v[ix].mean(1)
            ax.plot(x,v.mean(0),label=name,color=color)
            ax.fill_between(x,*np.percentile(curves,[2.5,97.5],axis=0),alpha=.17,color=color)
        ax.axvline(16,color='#777',ls='--',lw=1)
        ax.set(title=title,xlabel='Steps after real prefix',ylim=(0,1.03),xlim=(1,H));ax.grid(alpha=.2)
    axes[0].set_ylabel('Fraction of futures reaching event');axes[1].legend(loc='lower right')
    fig.suptitle('F4 native feedback: 32 sleeping energy-8 starts, 8 futures each')
    fig.tight_layout();fig.savefig(HERE/'sleep_timing.png',dpi=160);plt.close(fig)
    print(json.dumps({k:v for k,v in result.items() if k not in ('readers_valid',)},indent=2),flush=True)

if __name__=='__main__':main()
