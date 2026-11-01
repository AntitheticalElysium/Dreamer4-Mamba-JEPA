"""Exact additive accounting of existing loss derivatives; no altered learning or optimizer step."""
from .common import *
from .density import templates


def stats(a,b):
    na,nb=a.norm(),b.norm()
    return {'a_norm':float(na),'b_norm':float(nb),'dot':float(a@b),'cosine':float((a@b)/(na*nb)) if na*nb>0 else None}


if __name__=='__main__':
    before=fingerprints();model,cfg=load_model();mh=model_hash(model);model.pixels.requires_grad_(True)
    proto=templates();varmask=torch.zeros(81,147,dtype=torch.bool)
    for j,h in enumerate(HUD[:4]):px=proto[h]['pixels'];varmask[HUD_CELLS[j]]=px.max(0).values!=px.min(0).values
    cache=load_cache('train');tab=load_crops('train');L=labels('train');off=np.r_[0,np.cumsum(L['ep_T']+1)];ep=episode_rows(L)
    rng=np.random.default_rng(1901);sample=np.sort(rng.choice(np.flatnonzero(ep%10!=9),1024,replace=False));es=np.searchsorted(off[1:],sample,side='right');ts=sample-off[es]
    assert np.all((ts>=0)&(ts<L['ep_T'][es]+1)) and np.array_equal(off[es]+ts,sample)
    params=list(model.pixels.parameters());nmean=sum(p.numel() for p in model.pixels.mean.parameters());mean_ids=[];scale_ids=[];pos=0
    for name,p in model.pixels.named_parameters():
        ids=list(range(pos,pos+p.numel()));(mean_ids if name.startswith('mean.') else scale_ids).extend(ids);pos+=p.numel()
    grads={key:torch.zeros(pos,device='cuda') for key in ('tsp','rec','tsp_body','rec_body')};losses={k:0. for k in grads}
    for batch in np.array_split(np.arange(len(sample)),8):
        P=M.Percept(*(torch.stack([cache[e][k][t] for e,t in zip(es[batch],ts[batch])]).cuda() for k in ('p','g','v')),
                    torch.stack([cache[e]['fix'][t] for e,t in zip(es[batch],ts[batch])]).cuda())
        bp=torch.stack([cache[e]['bp'][t] for e,t in zip(es[batch],ts[batch])]).float().cuda();cells=M.window(P.fix)
        raw=tab['px'][torch.as_tensor(sample[batch,None]*9+np.arange(9))].cuda();vm=varmask[cells.cpu()].cuda()
        assert torch.equal(cells.cpu(),tab['cell'][torch.as_tensor(sample[batch,None]*9+np.arange(9))])
        with TR.autocast(torch.device('cuda'),cfg):
            mt,st=model.pixels(model.tsp.predict(bp,P.p,P.v,cells),P.v,cells);mr,sr=model.pixels(model.tsp.reconstruct(P.g,P.v,cells),P.v,cells)
        lt,lr=M.pixel_nll(mt,st,raw),M.pixel_nll(mr,sr,raw)
        terms={'tsp':lt.mean(),'rec':lr.mean(),'tsp_body':(lt*vm).mean(),'rec_body':(lr*vm).mean()}
        for key,loss in terms.items():
            gg=torch.autograd.grad(loss,params,retain_graph=True,allow_unused=True)
            vector=torch.cat([torch.zeros_like(p).flatten() if g is None else g.float().flatten() for p,g in zip(params,gg)])
            grads[key]+=vector*len(batch)/len(sample);losses[key]+=float(loss)*len(batch)/len(sample)
    out={'n_natural_frames':len(sample),'losses':losses,'parameters':{}}
    for name,inds in [('all',range(pos)),('mean',mean_ids),('scale_and_position',scale_ids)]:
        inds=torch.as_tensor(list(inds),device='cuda');g={k:v[inds] for k,v in grads.items()};full=g['tsp']+g['rec'];body=g['tsp_body']+g['rec_body'];rest=full-body
        out['parameters'][name]={'tsp_vs_rec':stats(g['tsp'],g['rec']),'body_vs_rest':stats(body,rest),
            'total_norm':float(full.norm()),'body_norm':float(body.norm()),'body_projection_share':float((body@full)/(full@full))}
    save('decoder_gradient_accounting.json',out);torch.save({k:v.cpu() for k,v in grads.items()},OUT/'decoder_gradients.pt')
    assert before==fingerprints() and model_hash(model)==mh
