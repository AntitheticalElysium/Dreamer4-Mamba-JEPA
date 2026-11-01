"""Read already-saved production outcomes, stratified only by preceding real context."""
from .common import *


if __name__=='__main__':
    L=labels('dev');b=torch.load(OLD/'baseline_samples.pt',weights_only=False,mmap=True)
    ep,t=b['eps'],b['ts'];off=np.r_[0,np.cumsum(L['ep_T']+1)];toff=np.r_[0,np.cumsum(L['ep_T'])]
    past=off[ep]+t-1;transition=toff[ep]+t-1;body=L['s_body'][past]
    realr=L['t_reward'][transition];realt=L['t_terminated'][transition];genr=b['generated']['r'].numpy();gent=b['generated']['term'].numpy()>.5
    zrel=L['s_zombies_p'][past]-L['s_pos'][past,None];adj=(L['s_zombies_m'][past]&(np.abs(zrel).sum(-1)<=1)).any(1)
    groups={'all':np.ones(len(ep),bool),'past_health_1_2':(body[:,0]>0)&(body[:,0]<=2),
            'past_health_3_5':(body[:,0]>=3)&(body[:,0]<=5),'past_health_6_9':body[:,0]>=6,
            'past_food_0':body[:,1]==0,'past_drink_0':body[:,2]==0,'past_energy_0':body[:,3]==0,
            'past_zombie_adjacent':adj,'past_low_health_zombie_adjacent':(body[:,0]<=2)&adj,
            'past_day':L['s_light'][past]>=.5,'past_night':L['s_light'][past]<.5,
            'random_behavior':L['ep_eps'][ep]==1,'expert_behavior':L['ep_eps'][ep]==0}
    observables={'reward':(realr,genr),'termination':(realt,gent),'negative_reward':(realr<-.05,genr<-.05),
                 'positive_reward':(realr>.05,genr>.05),'reward_ge_half':(realr>=.5,genr>=.5)}
    out={}
    for name,m in groups.items():
        if m.sum()<50:continue
        out[name]={'n_histories':int(m.sum())}
        for key,(r,g) in observables.items():
            out[name][key]={'recorded':G.estimate(r,m,ep),'generated':G.estimate(g,m,ep),
                            'paired_difference':G.estimate(g.astype(float)-r.astype(float),m,ep)}
    save('production_outcomes.json',out)
    print({name:{k:(round(v['recorded']['est'],4),round(v['generated']['est'],4)) for k,v in x.items() if isinstance(v,dict)} for name,x in out.items()},flush=True)
