"""Reuse actual real-path readouts: danger distinctions and naturally occurring crop exposure."""
from .common import *


def contrast(values,seen,stratum,mask,ep):
    """Standardised descriptive contrast; fixed within-stratum weights, physical-episode uncertainty."""
    wa=np.zeros(len(values));wb=np.zeros(len(values));total=int(mask.sum());covered=0
    for s in np.unique(stratum[mask]):
        m=mask&(stratum==s);na=int((m&seen).sum());nb=int((m&~seen).sum());n=int(m.sum())
        if min(na,nb)<20:continue
        wa[m&seen]=n/na;wb[m&~seen]=n/nb;covered+=n
    if covered==0:return {'coverage':0,'n_target':total,'difference':None}
    ids,inv=np.unique(ep,return_inverse=True);n=len(ids)
    a=np.bincount(inv,wa*values,n);b=np.bincount(inv,wb*values,n);da=np.bincount(inv,wa,n);db=np.bincount(inv,wb,n)
    point=a.sum()/da.sum()-b.sum()/db.sum();rng=np.random.default_rng(0);boots=[]
    for _ in range(200):
        q=rng.integers(0,n,n)
        if da[q].sum() and db[q].sum():boots.append(a[q].sum()/da[q].sum()-b[q].sum()/db[q].sum())
    return {'difference':float(point),'se':float(np.std(boots)),'coverage':covered/total,'n_target':total,
            'n_covered':covered,'exposed_error':float(a.sum()/da.sum()),'unexposed_error':float(b.sum()/db.sum())}


if __name__=='__main__':
    before=fingerprints();L=labels('dev');cache=load_cache('dev');ep=episode_rows(L);Y=hud_values(L)[:,:4]
    fixes=torch.cat([r['fix'] for r in cache]);cells=crop_cells(fixes);seen=(cells[:,:,None]==HUD_CELLS[None,None,:4]).any(1)
    terminated=np.concatenate([r['terminated'].numpy() for r in cache]);alive=(Y[:,0]>0)&~terminated
    night=L['s_light']<.5;eps=L['ep_eps'][ep];behavior=np.where(eps==1,2,np.where(eps==0,0,1));off=np.r_[0,np.cumsum(L['ep_T']+1)]
    depth=np.arange(len(ep))-off[ep];changed=np.zeros_like(Y,dtype=bool);changed[1:]=Y[1:]!=Y[:-1];changed[depth==0]=False
    gaps=[]
    for e,r in enumerate(cache):
        a,b=off[e:e+2];last=np.full(4,-1);g=np.full((b-a,4),-1)
        for t in range(b-a):
            last[seen[a+t]]=t;g[t]=np.where(last>=0,t-last,-1)
        gaps.append(g)
    gaps=np.concatenate(gaps);save('population.json',{'n_states':len(ep),'n_alive':int(alive.sum()),'n_episodes':len(cache),
          'body_prevalence_alive':{h:{str(v):int(((Y[:,j]==v)&alive).sum()) for v in range(10)} for j,h in enumerate(HUD[:4])}})
    results={}
    for rep in ('full_percept','x','b'):
        for seed in (0,1):
            ck=torch.load(OLD/f'belief_{rep}_body_body_{seed}.pt',weights_only=False,mmap=True)
            pred=ck['hud_prediction'].numpy()[:,:4].astype(np.int64);r={}
            for j,h in enumerate(HUD[:4]):
                y=Y[:,j];p=pred[:,j];err=np.abs(p-y);safe=alive
                groups={'alive':safe,'night':safe&night,'day':safe&~night,'after_change':safe&changed[:,j],
                        'deep_256':safe&(depth>=256),'random_behavior':safe&(behavior==2),'expert_behavior':safe&(behavior==0),
                        'seen_now':safe&seen[:,j],'not_seen_now':safe&~seen[:,j],
                        **{f'glimpse_age_{lo}_{hi}':safe&(gaps[:,j]>=lo)&(gaps[:,j]<=hi) for lo,hi in [(0,0),(1,4),(5,16),(17,64),(65,100000)]}}
                rr={name:{'exact':G.estimate(p==y,m,ep),'mae':G.estimate(err,m,ep),'signed_error':G.estimate(p-y,m,ep)} for name,m in groups.items() if m.any()}
                for threshold in ((2,) if j==0 else (0,2)):
                    critical=safe&(y<=threshold);normal=safe&(y>threshold)
                    rr[f'critical_le_{threshold}']={'prevalence':G.estimate(y<=threshold,safe,ep),
                       'false_safe_rate':G.estimate(p>threshold,critical,ep),'false_alarm_rate':G.estimate(p<=threshold,normal,ep),
                       'recall':G.estimate(p<=threshold,critical,ep)}
                stratum=y+10*night.astype(int)+20*behavior
                rr['natural_exposure_standardised_exact_error']=contrast((p!=y).astype(float),seen[:,j],stratum,safe,ep)
                rr['natural_exposure_standardised_mae']=contrast(err,seen[:,j],stratum,safe,ep)
                r[h]=rr
            results[f'{rep}/{seed}']=r;save('natural_state_readouts.json',results);print(rep,seed,flush=True)
    torch.save({'ep':torch.from_numpy(ep),'seen':torch.from_numpy(seen),'age':torch.from_numpy(gaps),'depth':torch.from_numpy(depth),
                'alive':torch.from_numpy(alive),'changed':torch.from_numpy(changed)},OUT/'natural_covariates.pt')
    assert before==fingerprints()
