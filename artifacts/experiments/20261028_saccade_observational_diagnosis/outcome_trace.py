"""Trace native outcome probabilities on unmodified complete real/generated percepts."""
from scipy.spatial import cKDTree
from .common import *


def local_hud_features(p,v,cells):
    n=len(p);z=torch.zeros(n,147);z[:,:16]=p.float()
    return torch.cat([z,v.float(),torch.nn.functional.one_hot(cells.long(),81).float()],1)


@torch.no_grad()
def main():
    before=fingerprints();model,cfg=load_model();mh=model_hash(model);L=labels('dev');c=load_cache('dev')
    b=torch.load(OLD/'baseline_samples.pt',weights_only=False,mmap=True);eps,ts=b['eps'],b['ts'];off=np.r_[0,np.cumsum(L['ep_T']+1)];toff=np.r_[0,np.cumsum(L['ep_T'])]
    idx=off[eps]+ts;pre=idx-1;trans=toff[eps]+ts-1;realr=torch.from_numpy(L['t_reward'][trans]);realt=L['t_terminated'][trans]
    bp=torch.stack([c[e]['bp'][t] for e,t in zip(eps,ts)]).float();fix=torch.stack([c[e]['fix'][t] for e,t in zip(eps,ts)])
    ck=torch.load(OLD/'reader_hud_periphery_0.pt',weights_only=False);reader=s1.MLP(ck['input_dim'],ck['heads']).cuda().eval();reader.load_state_dict(ck['state'])
    observed={}
    for domain in ('real','generated'):
        p=b['real' if domain=='real' else 'generated'];r=realr if domain=='real' else b['generated']['r']
        dead=[];rneg=[];rtrue=[];hud=[]
        for ids in torch.arange(len(eps)).split(256):
            P=M.Percept(*(p[k][ids].cuda() for k in ('p','g','v')),fix[ids].cuda())
            with TR.autocast(torch.device('cuda'),cfg):
                emb=model.belief.percept_embed(P);h=model.outcome.features(bp[ids].cuda(),emb)
                lp=model.outcome.bin_logp(h);cl=model.outcome.cont_logit(h,r[ids].cuda()).float()
            dead.append(torch.sigmoid(-cl).cpu());rneg.append(lp.float().exp()[:,model.outcome.bin_value<-.05].sum(-1).cpu())
            rtrue.append(lp.float().gather(-1,model.outcome.bin_of(r[ids].cuda())[:,None])[:,0].cpu())
            ps=p['p'][ids][:,HUD_CELLS[:4]].reshape(-1,16);vs=p['v'][ids,None].expand(-1,4,-1).reshape(-1,8)
            cells=torch.as_tensor(HUD_CELLS[:4])[None].expand(len(ids),-1).reshape(-1)
            hud.append(reader(local_hud_features(ps,vs,cells).cuda())['value'].softmax(-1).reshape(-1,4,10).cpu())
        observed[domain]={'dead_probability':torch.cat(dead),'negative_reward_probability':torch.cat(rneg),
                          'reward_log_probability_of_own_value':torch.cat(rtrue),'hud_probabilities':torch.cat(hud)}
    torch.save(observed,OUT/'native_observation_trace.pt')
    # A separate real-reference instrument for HUD codes. No reward/termination is fed to it.
    train=load_cache('train');Lt=labels('train');ep=episode_rows(Lt);pt=torch.cat([r['p'] for r in train]);yt=hud_values(Lt)
    rng=np.random.default_rng(801);fit=np.flatnonzero(ep%10!=9);cal=np.flatnonzero(ep%10==9);refs={}
    for j,name in enumerate(HUD[:4]):
        ri=rng.choice(fit,min(100000,len(fit)),replace=False);ci=rng.choice(cal,min(15000,len(cal)),replace=False)
        xx=pt[ri,HUD_CELLS[j]].half().float().numpy();mean=xx.mean(0);scale=np.sqrt(xx.var(0).mean());tree=cKDTree((xx-mean)/scale)
        def read(p):
            dd,ii=tree.query((p.half().float().numpy()-mean)/scale,k=8,workers=6);lab=yt[ri,j][ii];pred=lab[:,0]
            return pred,dd[:,0],(lab==pred[:,None]).mean(1)
        cp,cd,ca=read(pt[ci,HUD_CELLS[j]]);th=np.quantile(cd,.95);refs[name]={'threshold':float(th),
              'calibration_exact':float((cp==yt[ci,j]).mean()),'calibration_zero_recall':float((cp[yt[ci,j]==0]==0).mean()) if (yt[ci,j]==0).any() else None}
        for domain in ('real','generated'):
            pr,dist,agree=read(b[domain if domain=='real' else 'generated']['p'][:,HUD_CELLS[j]])
            refs[name][domain]={'value':pr,'supported':(dist<=th)&(agree>=7/8),'distance':dist}
    torch.save(refs,OUT/'hud_reference.pt')
    body=L['s_body'][pre];zrel=L['s_zombies_p'][pre]-L['s_pos'][pre,None];adj=(L['s_zombies_m'][pre]&(np.abs(zrel).sum(-1)<=1)).any(1)
    groups={'all':np.ones(len(eps),bool),'past_low_health':body[:,0]<=2,'past_zombie_adjacent':adj,
            'past_low_health_zombie_adjacent':(body[:,0]<=2)&adj,'recorded_fatal':realt.astype(bool),'recorded_alive':~realt.astype(bool)}
    report={}
    for name,mask in groups.items():
        report[name]={'n':int(mask.sum()),'recorded_termination':G.estimate(realt,mask,eps)}
        for domain in observed:
            z=observed[domain];pr=z['hud_probabilities'][:,0].numpy();pred=pr.argmax(-1);ref=refs['health'][domain]
            report[name][domain]={'native_dead_probability':G.estimate(z['dead_probability'].numpy(),mask,eps),
                'native_negative_reward_probability':G.estimate(z['negative_reward_probability'].numpy(),mask,eps),
                'hud_reader_zero_fraction':G.estimate(pred==0,mask,eps),'hud_reader_p_zero':G.estimate(pr[:,0],mask,eps),
                'hud_nn_zero_fraction':G.estimate(ref['value']==0,mask,eps),
                'hud_nn_supported_fraction':G.estimate(ref['supported'],mask,eps),
                'hud_nn_supported_zero_fraction':G.estimate((ref['value']==0)&ref['supported'],mask,eps)}
    report['reference_calibration']={name:{k:v for k,v in a.items() if k not in ('real','generated')} for name,a in refs.items()}
    save('native_outcome_trace.json',report)
    assert before==fingerprints() and model_hash(model)==mh


if __name__=='__main__':main()
