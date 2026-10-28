"""CPU summaries of paired sampling arms and observed-target loss mass, not new model predictions."""
from .common import *
from saccade.eval.probe import KEEP


@torch.no_grad()
def read_cpu(reader,p,v):
    x=torch.cat([p[:,:63].reshape(-1,16).half(),v.repeat_interleave(63,0).half()],1)
    return torch.cat([reader(a)['kind'].argmax(-1) for a in x.split(65536)]).reshape(len(p),63).numpy()


if __name__=='__main__':
    b=torch.load(OUT/'baseline_samples.pt',weights_only=False,mmap=True)
    p=torch.load(OUT/'sampler_panel.pt',weights_only=False,mmap=True)
    reader=s1.load_reader(READER,CKPT,'cpu');L=labels('dev');off=np.r_[0,np.cumsum(L['ep_T']+1)]
    grid=np.concatenate([a['grid'] for a in grids(L)]).reshape(-1,63)
    panel=p['panel'];eps,ts=p['eps'],p['ts'];state=off[eps]+ts;true=grid[state];prev=grid[state-1]
    nights=L['s_light'][state]<.5
    selection=json.loads((OUT/'generation_panel.json').read_text());pop=np.isin(panel,selection['population_indices'])
    frames={'population':pop,'population_day':pop&~nights,'population_night':pop&nights,
            **{k:np.isin(panel,ids) for k,ids in selection['cohorts'].items()}}
    pred={}
    for arm,reps in p['arms'].items():
        pred[arm]=np.stack([read_cpu(reader,r['p'],r['v']) for r in reps])
        print('read arm',arm,flush=True)
    ref=b['real_reader'][panel].numpy();out={};se=[]
    for name,mask in frames.items():
        out[name]={}
        for k,kind in enumerate(KINDS[1:],1):
            d={'n_histories':int(mask.sum()),'simulator_real_count':G.estimate((true[:,KEEP]==k).sum(1),mask,eps),
               'reader_real_count':G.estimate((ref[:,KEEP]==k).sum(1),mask,eps)}
            for arm,a in pred.items():
                c=(a[:,:,KEEP]==k).sum(-1).mean(0);d[arm]=G.estimate(c,mask,eps)
            for a,bn in [('ode_3','production_3_churn1'),('ode_8','ode_3'),('ode_32','ode_3')]:
                ca=(pred[a][:,:,KEEP]==k).sum(-1).mean(0);cb=(pred[bn][:,:,KEEP]==k).sum(-1).mean(0)
                d[f'delta/{a}-{bn}']=G.estimate(ca-cb,mask,eps)
            out[name][kind]=d
    save('sampler_conditional_counts.json',out)
    cache=torch.load(OUT/'real_dev.pt',weights_only=False,mmap=True)
    cells=crop_cells(torch.stack([cache[e]['fix'][t] for e,t in zip(eps,ts)]))
    kg=np.pad(true,((0,0),(0,18)),constant_values=-1);gkind=kg[np.arange(len(kg))[:,None],cells]
    losses={}
    for sigma,batches in p['observed_target_denoising'].items():
        raw={k:torch.cat([a['raw_se'][k] for a in batches]).numpy() for k in ('p','g','v')}
        group={}
        for key,label in [('p',kg),('g',gkind)]:
            ss=raw[key].sum(-1);index=np.broadcast_to(np.arange(81),kg.shape) if key=='p' else cells
            masks={'entities':label>0,'map_background':(label==0)&(index!=31),'player':index==31,'HUD':index>=63}
            for name,m in masks.items():group[f'{key}/{name}']=(ss*m).sum(1)
        group['v/global']=raw['v'].sum(-1)
        total=sum(group.values());n=1296+576+8
        losses[str(sigma)]={'population_g1':float(total[pop].mean()/n),'loss_share':{k:float(v[pop].sum()/total[pop].sum()) for k,v in group.items()}}
    save('observed_target_loss_decomposition.json',losses)
    # Finite-precision CPU reader parity is checked directly against the GPU-labelled baseline.
    pp=read_cpu(reader,b['generated']['p'][:2048],b['generated']['v'][:2048]);rr=b['generated_reader'][:2048].numpy()
    save('cpu_reader_parity.json',{'cell_agreement':float((pp==rr).mean()),'n_cells':int(pp.size)})
