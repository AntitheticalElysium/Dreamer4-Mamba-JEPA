"""Observed co-occurrence of generated threat evidence and native consequence probabilities."""
from .common import *


if __name__=='__main__':
    b=torch.load(OLD/'baseline_samples.pt',weights_only=False,mmap=True);n=torch.load(OUT/'native_observation_trace.pt',weights_only=False)
    L=labels('dev');off=np.r_[0,np.cumsum(L['ep_T']+1)];eps,ts=b['eps'],b['ts'];past=off[eps]+ts-1
    rel=L['s_zombies_p'][past]-L['s_pos'][past,None];adj=(L['s_zombies_m'][past]&(np.abs(rel).sum(-1)<=1)).any(1)
    low=L['s_body'][past,0]<=2;contexts={'all':np.ones(len(eps),bool),'past_zombie_adjacent':adj,'past_low_health_adjacent':adj&low}
    positions=[22,40,30,32];out={}
    for domain in ('real','generated'):
        evidence=(b['real_reader' if domain=='real' else 'generated_reader'][:,positions].numpy()==1).any(1)
        out[domain]={}
        for name,mask in contexts.items():
            out[domain][name]={'adjacent_zombie_reader_rate':G.estimate(evidence,mask,eps)}
            for label,m in [('reader_adjacent',mask&evidence),('reader_not_adjacent',mask&~evidence)]:
                out[domain][name][label]={'n':int(m.sum()),
                    'negative_reward_probability':G.estimate(n[domain]['negative_reward_probability'].numpy(),m,eps),
                    'death_probability':G.estimate(n[domain]['dead_probability'].numpy(),m,eps)}
    save('joint_observations.json',out);print(out)
