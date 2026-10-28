"""Native predicted risks across actual saved training snapshots; no recipe changes."""
from .common import *


class FixedTuples:
    def __init__(self,eps,ts):self.eps,self.ts,self.i=eps,ts,0
    def choice(self,*args,**kwargs):return self.eps.copy()
    def integers(self,*args,**kwargs):t=int(self.ts[self.i]);self.i+=1;return t


@torch.no_grad()
def main():
    before=fingerprints();b=torch.load(OLD/'baseline_samples.pt',weights_only=False,mmap=True);L=labels('dev');off=np.r_[0,np.cumsum(L['ep_T']+1)];toff=np.r_[0,np.cumsum(L['ep_T'])]
    es,ts=b['eps'],b['ts'];past=off[es]+ts-1;body=L['s_body'][past];rel=L['s_zombies_p'][past]-L['s_pos'][past,None]
    adj=(L['s_zombies_m'][past]&(np.abs(rel).sum(-1)<=1)).any(1);risk=(body[:,0]<=2)&adj;hungry=body[:,1]==0
    population=np.linspace(0,29999,1024,dtype=int);rng=np.random.default_rng(1102);fi=np.flatnonzero(hungry);fi=np.sort(rng.choice(fi,min(512,len(fi)),replace=False))
    panel=np.unique(np.r_[population,np.flatnonzero(risk),fi]);eps,t=es[panel],ts[panel]
    groups={'population':np.isin(panel,population),'past_low_health_zombie_adjacent':risk[panel],'past_food_zero':hungry[panel]}
    rr=L['t_reward'][toff[eps]+t-1];rt=L['t_terminated'][toff[eps]+t-1]
    save('risk_timecourse_panel.json',{'indices':panel.tolist(),'population_indices':population.tolist(),'n':len(panel),'groups':{k:int(v.sum()) for k,v in groups.items()},'draws_per_history':4})
    out={}
    for snapshot in (1,5,10):
        ck=torch.load(RUN/f'ckpt_{snapshot:02d}.pt',weights_only=False,map_location='cpu',mmap=True);cfg=ck['cfg']
        model=M.M0(cfg).cuda().eval().requires_grad_(False);model.load_state_dict(ck['model']);del ck;mh=model_hash(model)
        enc=s1.encode(model,L,cfg);print('encoded snapshot',snapshot,flush=True)
        ee=np.repeat(eps,4);tt=np.repeat(t,4);_,_,gen=s1.one_step(model,enc,cfg,len(ee),FixedTuples(ee,tt))
        prob=[];neg=[]
        for ids in torch.arange(len(ee)).split(256):
            P=M.Percept(*(gen[k][ids].cuda() for k in ('p','g','v')),torch.stack([enc[e]['fix'][z] for e,z in zip(ee[ids],tt[ids])]).cuda())
            bp=torch.stack([enc[e]['bp'][z] for e,z in zip(ee[ids],tt[ids])]).float().cuda()
            with TR.autocast(torch.device('cuda'),cfg):
                h=model.outcome.features(bp,model.belief.percept_embed(P));cl=model.outcome.cont_logit(h,gen['r'][ids].cuda()).float();lp=model.outcome.bin_logp(h).float()
            prob.append(torch.sigmoid(-cl).cpu());neg.append(lp.exp()[:,model.outcome.bin_value<-.05].sum(-1).cpu())
        pd=torch.cat(prob).numpy().reshape(-1,4).mean(1);pn=torch.cat(neg).numpy().reshape(-1,4).mean(1)
        pr=np.array([1-float(enc[e]['p_cont'][z]) for e,z in zip(eps,t)])
        outcomes={'termination':(rt,pd),'negative_reward':(rr<-.05,pn),'reward':(rr,gen['r'].reshape(-1,4).numpy().mean(1))}
        out[snapshot]={}
        for name,mask in groups.items():
            out[snapshot][name]={'n':int(mask.sum()),'teacher_forced_dead_probability':G.estimate(pr,mask,eps)}
            for key,(real,pred) in outcomes.items():
                out[snapshot][name][key]={'recorded':G.estimate(real,mask,eps),'generated':G.estimate(pred,mask,eps),'paired_difference':G.estimate(pred-real.astype(float),mask,eps)}
        save('risk_timecourse.json',out);torch.save({'panel':panel,'native_dead_probability':torch.from_numpy(pd),'native_negative_probability':torch.from_numpy(pn),
           'generated':gen},OUT/f'risk_snapshot_{snapshot:02d}.pt')
        print('snapshot',snapshot,'risk',out[snapshot]['past_low_health_zombie_adjacent']['termination'],flush=True)
        assert model_hash(model)==mh;del model,enc,gen;torch.cuda.empty_cache()
    assert before==fingerprints()


if __name__=='__main__':main()
