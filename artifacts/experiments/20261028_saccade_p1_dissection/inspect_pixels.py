"""Fixed examples, likelihood scale/mean audit and independent reader calibration."""
import time
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .common import *
from .generation import prepare,sample_prepared
from .fovea_readers import features


def reader(task,arm):
    r=torch.load(OUT/f'reader_{task}_{arm}_0.pt',weights_only=False)
    m=s1.MLP(r['input_dim'],r['heads']).cuda().eval().requires_grad_(False);m.load_state_dict(r['state']);return m


def rgb(mean):return ((mean.float()+1)*127.5).clamp(0,255)


@torch.no_grad()
def main():
    before=fingerprints();model,cfg=load_model();h=model_hash(model)
    cache=torch.load(OUT/'real_dev.pt',mmap=True,weights_only=False)
    b=torch.load(OUT/'baseline_samples.pt',weights_only=False);eps,ts=b['eps'],b['ts']
    # Exact one-batch parity, including S1's random-draw order.
    class Fixed:
        def __init__(self):self.i=0
        def choice(self,*args,**kwargs):return eps[:64].copy()
        def integers(self,*args,**kwargs):i=self.i;self.i+=1;return int(ts[i])
    _,_,ref=s1.one_step(model,half_encoding(cache),cfg,64,Fixed())
    prep=prepare(model,half_encoding(cache),eps[:64],ts[:64],cfg);rg=torch.Generator(device='cuda').manual_seed(0)
    target=prep[0];cn=[torch.randn(a[:,:-1].shape,device='cuda',generator=rg) for a in target]
    ini=[torch.randn(a[:,-1:].shape,device='cuda',generator=rg) for a in target]
    churn=[[torch.randn(a[:,-1:].shape,device='cuda',generator=rg) for a in target] for _ in range(3)]
    actual=sample_prepared(model,prep,cfg,3,1.,cn,ini,churn)
    diffs={k:float((ref[k]-actual[k]).abs().max()) for k in actual}
    save('sampler_parity.json',{'max_absolute_differences':diffs,'same_batch_same_random_draws':True})
    assert max(diffs.values())<1e-6,diffs
    L=labels('dev');off=np.r_[0,np.cumsum(L['ep_T']+1)];grid=np.concatenate([e['grid'] for e in grids(L)]).reshape(-1,63)
    fix=torch.stack([cache[e]['fix'][t] for e,t in zip(eps,ts)]);cells=M.window(fix)
    n=len(eps);kg=np.pad(grid[off[eps]+ts],((0,0),(0,18)),constant_values=-1)
    prevpx=torch.empty(n,9,147,dtype=torch.uint8);ep_objects=s1.episodes(L)
    for e in np.unique(eps):
        ix=np.flatnonzero(eps==e);ep=ep_objects[e]
        fr=D.RP.episode_fields(ep.shard,ep.slot)['observations'][ts[ix]-1]
        prevpx[ix]=M.cell_pixels(M.crop(fr,fix[ix]))
    values=np.full((n,81),-1);values[:,HUD_CELLS]=hud_values(L)[off[eps]+ts]
    kind=kg[np.arange(n)[:,None],cells.numpy()];hv=values[np.arange(n)[:,None],cells.numpy()]
    masks={'map':cells.numpy()<63,'HUD':cells.numpy()>=63,**{name:cells.numpy()==c for name,c in zip(HUD,HUD_CELLS)}}
    # Decode both real routes and generated fovea on identical sampled tuples.
    recon={d:[] for d in ('real_rec','real_tsp','generated_rec')};ls=[];mu=[];raw=[];zloss=[]
    for ids in torch.arange(n).split(256):
        pp={k:v[ids].cuda() for k,v in b['real'].items()};gg={k:b['generated'][k][ids].cuda() for k in ('p','g','v')}
        c=cells[ids].cuda();bp=torch.stack([cache[e]['bp'][t] for e,t in zip(eps[ids],ts[ids])]).float().cuda()
        # Actual targets loaded once per episode elsewhere: reuse selected crop table through global state offsets.
        tab=torch.load(OUT/'crops_dev.pt',weights_only=False,mmap=True) if len(raw)==0 else tab
        indices=torch.as_tensor((off[eps[ids]]+ts[ids])[:,None]*9+np.arange(9))
        px=tab['px'][indices].cuda();raw.append(px.cpu())
        with TR.autocast(torch.device('cuda'),cfg):
            mr,s=model.pixels(model.tsp.reconstruct(pp['g'],pp['v'],c),pp['v'],c)
            mt,st=model.pixels(model.tsp.predict(bp,pp['p'],pp['v'],c),pp['v'],c)
            mg,_=model.pixels(model.tsp.reconstruct(gg['g'],gg['v'],c),gg['v'],c)
        assert torch.equal(s,st)
        for key,x in (('real_rec',mr),('real_tsp',mt),('generated_rec',mg)):recon[key].append(rgb(x).cpu())
        ls.append(s.float().cpu());mu.append(mr.float().cpu());zloss.append(M.pixel_nll(torch.zeros_like(mr),s,px).mean(-1).cpu())
    recon={k:torch.cat(v) for k,v in recon.items()};raw=torch.cat(raw);ls=torch.cat(ls);mu=torch.cat(mu);zloss=torch.cat(zloss)
    scale=ls.clamp(min=-7).exp()*127.5;endpoint=(raw==0)|(raw==255)
    rows={}
    for name,mask in masks.items():
        if not mask.any():continue
        m=torch.as_tensor(mask);v=scale[m];y=endpoint[m];actual=raw[m]
        rows[name]={'n_cells':int(m.sum()),'endpoint_fraction':float(y.float().mean()),
                    'scale_rgb_quantiles':np.quantile(v.numpy(),[.1,.5,.9,.99]).tolist(),
                    'scale_gt_255_fraction':float((v>255).float().mean()),
                    'location_outside_rgb_fraction':float((mu[m].abs()>1).float().mean()),
                    'zero_location_same_scale_nll':float(zloss[m].mean()),
                    'copy_previous_same_scale_nll':float(M.pixel_nll(prevpx[m].float()/127.5-1,ls[m],actual).mean()),
                    'rec_nll':float(M.pixel_nll(mu[m],ls[m],actual).mean()),
                    'rec_mae':float((recon['real_rec'][m]-actual.float()).abs().mean())}
    save('pixel_density.json',rows)
    # Fresh independently fitted readers, including exact positive controls on REAL raw cells.
    all_preds={};summaries={}
    for arm in ('periphery','fovea','joint','pixels_full'):
        readers={task:reader(task,arm) for task in ('map','hud')}
        domains=('real','generated') if arm!='pixels_full' else ('raw','real_rec','real_tsp','generated_rec')
        for domain in domains:
            parts=b['real'] if domain in ('real','raw','real_rec','real_tsp') else b['generated']
            pred={task:[] for task in readers}
            for ids in torch.arange(n).split(256):
                c=cells[ids];p=parts['p'][ids][torch.arange(len(ids))[:,None],c]
                px=raw[ids].float() if domain=='raw' else recon[domain][ids] if arm=='pixels_full' else raw[ids].float()
                tt={'p':p.reshape(-1,16),'g':parts['g'][ids].reshape(-1,64),
                    'v':parts['v'][ids,None].expand(-1,9,-1).reshape(-1,8),'px':px.reshape(-1,147),'cell':c.flatten()}
                for task,r in readers.items():
                    o=r(features(tt,torch.arange(len(c)*9),arm,task).cuda());pred[task].append(next(iter(o.values())).argmax(-1).reshape(-1,9).cpu())
            pred={k:torch.cat(v).numpy() for k,v in pred.items()};all_preds[f'{arm}/{domain}']=pred
            m=(cells.numpy()<63)&(cells.numpy()!=31);hm=hv>=0
            summaries[f'{arm}/{domain}']={'crop_entity_counts_per_frame':{name:float(((pred['map']==k)&m).sum(1).mean()) for k,name in enumerate(KINDS[1:],1)}}
            if domain in ('real','raw','real_rec','real_tsp'):
                summaries[f'{arm}/{domain}']['real_label_kind_recall']={name:float((pred['map'][m&(kind==k)]==k).mean()) for k,name in enumerate(KINDS[1:],1)}
                summaries[f'{arm}/{domain}']['hud_exact']={name:float((pred['hud'][cells.numpy()==c]==hv[cells.numpy()==c]).mean()) for name,c in zip(HUD,HUD_CELLS)}
        del readers
    save('independent_readers.json',summaries);torch.save(all_preds,OUT/'independent_predictions.pt')
    # Preselected conditions depend on real labels only; choose earliest eligible sampled tuple.
    conditions={f'{name}/'+light:(kind==k).any(1)&((L['s_light'][off[eps]+ts]<.5)==night)
                for k,name in ((1,'zombie'),(4,'arrow')) for light,night in (('day',False),('night',True))}
    for j,name in enumerate(('health','food','drink','energy')):
        conditions[name]=((cells.numpy()==HUD_CELLS[j])&(hv<=3)&(hv>=0)).any(1)
    chosen={name:int(np.flatnonzero(m)[0]) for name,m in conditions.items() if m.any()}
    save('example_selection.json',{'rule':'first eligible tuple in fixed S1 tuple sequence; chosen from real labels only','indices':chosen})
    fig,axes=plt.subplots(len(chosen),4,figsize=(10,2.1*len(chosen)))
    def tile(a):return a.reshape(3,3,7,7,3).permute(0,2,1,3,4).reshape(21,21,3).byte().numpy()
    for row,(name,i) in enumerate(chosen.items()):
        for col,(title,x) in enumerate((('Real crop',raw[i]),('TSP before glimpse',recon['real_tsp'][i]),('Fovea reconstruction',recon['real_rec'][i]),('Generated fovea',recon['generated_rec'][i]))):
            axes[row,col].imshow(tile(x),interpolation='nearest');axes[row,col].set_xticks([]);axes[row,col].set_yticks([])
            if row==0:axes[row,col].set_title(title)
            if col==0:axes[row,col].set_ylabel(name)
    fig.tight_layout();fig.savefig(OUT/'crops_examples.png',dpi=170);plt.close(fig)
    assert model_hash(model)==h and fingerprints()==before
    save('pixels_run.json',{'checkpoint_and_sources_unchanged':True,'examples':len(chosen)})


if __name__=='__main__':main()
