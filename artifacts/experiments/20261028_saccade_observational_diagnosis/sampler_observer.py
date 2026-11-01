"""Read-only observation of the existing production sampler. Hook returns no value and consumes no RNG."""
from .common import *
from .risk_timecourse import FixedTuples


@torch.no_grad()
def main():
    before=fingerprints();model,cfg=load_model();mh=model_hash(model);cache=previous.half_encoding(load_cache('dev'))
    b=torch.load(OLD/'baseline_samples.pt',weights_only=False,mmap=True)
    target=torch.cat([b['real'][k].flatten(1) for k in ('p','g','v')],1);mean=target.mean(0).cuda();direction=mean/mean.norm()
    panel=json.loads((OUT/'risk_timecourse_panel.json').read_text());indices=np.array(panel['indices']);eps,ts=b['eps'][indices],b['ts'][indices]
    rows=[];call=0
    def hook(module,args,output):
        nonlocal call
        scaled,cn=args[:2];sigma=(4*cn[:,-1]).exp();cs,co,ci,_=M.precondition(sigma,cfg['edm']['sigma_data'])
        x=torch.cat([a[:,-1].flatten(1) for a in scaled],1)/ci[:,None]
        f=torch.cat([a[:,-1].flatten(1).float() for a in output],1);D=(cs[:,None]*x+co[:,None]*f).clamp(-3,3)
        rows.append({'stage':call%3,'sigma':sigma.cpu(),'input_projection':(x*direction[None]).sum(-1).cpu(),'clean_projection':(D*direction[None]).sum(-1).cpu(),
                     'input_rms':x.square().mean(-1).sqrt().cpu(),'clean_rms':D.square().mean(-1).sqrt().cpu()});call+=1
    # Exact instrumentation control.
    ee,tt=eps[:64],ts[:64];_,_,plain=s1.one_step(model,cache,cfg,len(ee),FixedTuples(ee,tt))
    handle=model.denoiser.register_forward_hook(hook);_,_,observed=s1.one_step(model,cache,cfg,len(ee),FixedTuples(ee,tt));handle.remove()
    diff={k:float((plain[k]-observed[k]).abs().max()) for k in plain};assert max(diff.values())==0,diff
    rows=[];call=0;handle=model.denoiser.register_forward_hook(hook)
    s1.one_step(model,cache,cfg,len(eps),FixedTuples(eps,ts));handle.remove()
    out={'instrumentation_max_differences':diff,'target_dimension':target.shape[1],'real_mean_norm':float(mean.norm()),
         'real_projection_mean':float((target@direction.cpu()).mean()),'real_projection_sd':float((target@direction.cpu()).std()),
         'real_rms':float(target.square().mean().sqrt()),'stages':{}}
    for stage in range(3):
        selected=[r for r in rows if r['stage']==stage];s=float(selected[0]['sigma'][0]);r={}
        for key in ['input_projection','clean_projection','input_rms','clean_rms']:
            a=torch.cat([x[key] for x in selected]).float();r[key]={'mean':float(a.mean()),'sd':float(a.std()),'q10_50_90':torch.quantile(a,torch.tensor([.1,.5,.9])).tolist()}
        r['sigma']=s;r['real_plus_noise_projection_mean']=out['real_projection_mean'];r['real_plus_noise_projection_sd']=float(np.sqrt(out['real_projection_sd']**2+s**2))
        r['gaussian_mean_shift_kl_term']=out['real_mean_norm']**2/(2*s*s);out['stages'][stage]=r
    save('sampler_observation.json',out);torch.save(rows,OUT/'sampler_observation.pt');assert before==fingerprints() and model_hash(model)==mh


if __name__=='__main__':main()
