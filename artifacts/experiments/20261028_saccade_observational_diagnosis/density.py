"""Observe full HUD densities, not only their clipped location images. No model/input changes."""
import time
import torch.nn.functional as F
from .common import *


def reference_nll(mu,log_s,pixels,dtype=torch.float64):
    """Same mathematical distribution. Bin width is explicit, not a subtraction of rounded CDF arguments."""
    m=mu.to(dtype);s=log_s.to(dtype).clamp(min=-7);inv=(-s).exp();y=pixels.to(dtype)/127.5-1
    a=(y+1/255-m)*inv;b=(y-1/255-m)*inv
    middle=F.logsigmoid(a)+F.logsigmoid(-b)+torch.log(-torch.expm1(-(2/255)*inv))
    return -torch.where(pixels==0,F.logsigmoid(a),torch.where(pixels==255,F.logsigmoid(-b),middle))


def templates():
    tab=load_crops('train');out={};meta={}
    for j,h in enumerate(HUD[:4]):
        use=(tab['cell']==HUD_CELLS[j])&(tab['ep']%10!=9)
        px=tab['px'][use].numpy();y=tab['value'][use].numpy();vals=np.unique(y);proto=[];info={}
        for v in vals:
            unique,count=np.unique(px[y==v],axis=0,return_counts=True);proto.append(unique[np.argmax(count)])
            info[int(v)]={'n':int(count.sum()),'n_unique':len(count),'mode_fraction':float(count.max()/count.sum())}
        out[h]={'pixels':torch.from_numpy(np.stack(proto)),'classes':vals}
        meta[h]=info
    save('template_training.json',meta)
    return out


def semantics(mu,ls,raw,truth,classes,proto,ep):
    # Exact raw-image control, not a neural prediction on decoded pixels.
    raw_dist=(raw[:,None].float()-proto[None].float()).square().sum(-1)
    raw_pred=classes[raw_dist.argmin(-1).numpy()]
    # Only coordinates that vary among candidates in the renderer's number region.
    region=torch.zeros(7,7,3,dtype=torch.bool);region[2:,2:]=True
    varying=(proto.max(0).values!=proto.min(0).values)&region.flatten()
    assert varying.any()
    scores=[];refs=[]
    for batch in torch.arange(len(mu)).split(512):
        x=proto[None,:,varying];m=mu[batch][:,None,varying];s=ls[batch][:,None,varying]
        scores.append(-M.pixel_nll(m,s,x).sum(-1));refs.append(-reference_nll(m,s,x).sum(-1).float())
    score=torch.cat(scores);ref=torch.cat(refs);pred=classes[score.argmax(-1).numpy()]
    refpred=classes[ref.argmax(-1).numpy()];cond=score.softmax(-1);llegal=score.logsumexp(-1)
    reflegal=ref.logsumexp(-1);allm=np.ones(len(truth),bool)
    result={'n':len(mu),'n_variable_rgb_coordinates':int(varying.sum()),'classes':classes.tolist(),
            'raw_template_accuracy':G.estimate(raw_pred==truth,allm,ep),
            'canonical_digit_likelihood_accuracy':G.estimate(pred==truth,allm,ep),
            'reference_digit_likelihood_accuracy':G.estimate(refpred==truth,allm,ep),
            'conditional_confidence_median':float(cond.max(-1).values.median()),
            'canonical_log_legal_mass_quantiles':torch.quantile(llegal,torch.tensor([.1,.5,.9])).tolist(),
            'reference_log_legal_mass_quantiles':torch.quantile(reflegal,torch.tensor([.1,.5,.9])).tolist(),
            'canonical_prediction_histogram':np.bincount(pred,minlength=10).tolist(),
            'truth_histogram':np.bincount(truth,minlength=10).tolist(),
            'by_value':{int(v):{'n':int((truth==v).sum()),'canonical_accuracy':float((pred[truth==v]==v).mean()),
                             'reference_accuracy':float((refpred[truth==v]==v).mean())} for v in np.unique(truth)}}
    return result


def numerics(mu,ls,raw,ep):
    a=mu.float().clone().requires_grad_();b=ls.float().clone().requires_grad_()
    L=M.pixel_nll(a,b,raw);g=torch.autograd.grad(L.sum(),(a,b))
    rmu=mu.double().clone().requires_grad_();rls=ls.double().clone().requires_grad_()
    R=reference_nll(rmu,rls,raw);gr=torch.autograd.grad(R.sum(),(rmu,rls))
    out={};masks={'all':torch.ones_like(raw,dtype=torch.bool),'interior':(raw>0)&(raw<255),'endpoint':(raw==0)|(raw==255)}
    for name,mask in masks.items():
        err=(L.detach().double()-R.detach())[mask];sg=g[1][mask].detach().double();rg=gr[1][mask].detach()
        active=rg.abs()>.01
        out[name]={'n_rgb':int(mask.sum()),'nll_mean_canonical':float(L.detach()[mask].mean()),'nll_mean_reference':float(R.detach()[mask].mean()),
                  'nll_error_quantiles':torch.quantile(err,torch.tensor([.01,.5,.99],dtype=torch.float64)).tolist(),
                  'fraction_abs_nll_error_gt_01':float((err.abs()>.1).double().mean()),
                  'scale_gradient_sign_disagreement_when_ref_gt_001':float(((sg[active]*rg[active])<0).double().mean()) if active.any() else None,
                  'canonical_scale_gradient_mean':float(sg.mean()),'reference_scale_gradient_mean':float(rg.mean()),
                  'canonical_negative_ref_positive_count':int(((sg<0)&(rg>.01)).sum()),
                  'mean_abs_location_gradient_canonical':float(g[0][mask].abs().mean()),
                  'mean_abs_location_gradient_reference':float(gr[0][mask].abs().mean())}
    return out


@torch.no_grad()
def capture_latest(model,cfg):
    b=torch.load(OLD/'baseline_samples.pt',weights_only=False,mmap=True);cache=load_cache('dev')
    L=labels('dev');off=np.r_[0,np.cumsum(L['ep_T']+1)];tab=load_crops('dev')
    eps,ts=b['eps'],b['ts'];fix=torch.stack([cache[e]['fix'][t] for e,t in zip(eps,ts)]);cells=M.window(fix)
    ix=torch.as_tensor((off[eps]+ts)[:,None]*9+np.arange(9))
    out={'cell':cells,'raw':tab['px'][ix],'value':tab['value'][ix],'ep':torch.as_tensor(eps)}
    mu=[];logs=[]
    for ids in torch.arange(len(eps)).split(256):
        g,v=b['real']['g'][ids].cuda(),b['real']['v'][ids].cuda();c=cells[ids].cuda()
        with TR.autocast(torch.device('cuda'),cfg):m,s=model.pixels(model.tsp.reconstruct(g,v,c),v,c)
        mu.append(m.float().cpu());logs.append(s.float().cpu())
    out['mu']=torch.cat(mu);out['log_s']=torch.cat(logs)
    torch.save(out,OUT/'captured_real_density.pt');return out


if __name__=='__main__':
    start=time.time();before=fingerprints();model,cfg=load_model();mh=model_hash(model)
    proto=templates();z=capture_latest(model,cfg);out={};math={}
    for j,h in enumerate(HUD[:4]):
        m=z['cell']==HUD_CELLS[j];ep=z['ep'][:,None].expand_as(m)[m].numpy()
        mm,ss,raw,y=z['mu'][m],z['log_s'][m],z['raw'][m],z['value'][m].numpy()
        out[h]=semantics(mm,ss,raw,y,proto[h]['classes'],proto[h]['pixels'],ep)
        math[h]=numerics(mm,ss,raw,ep)
        save('density_semantics.json',out);save('likelihood_numerics.json',math)
        print(h,'raw',out[h]['raw_template_accuracy']['est'],'density',out[h]['canonical_digit_likelihood_accuracy']['est'],
              'reference',out[h]['reference_digit_likelihood_accuracy']['est'],'interior',math[h]['interior'],flush=True)
    assert before==fingerprints() and model_hash(model)==mh
    save('density_run.json',{'seconds':time.time()-start,'checkpoint_and_sources_unchanged':True,'peak_gpu_gib':torch.cuda.max_memory_allocated()/2**30})
