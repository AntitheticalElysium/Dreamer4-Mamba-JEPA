"""Likelihood geometry and natural checkpoint trajectory. Derivatives only; no updates/perturbations."""
from .common import *
from .density import templates,numerics,reference_nll


if __name__=='__main__':
    before=fingerprints();proto=templates();b=torch.load(OLD/'baseline_samples.pt',weights_only=False,mmap=True);cache=load_cache('dev')
    ix=np.linspace(0,29999,8192,dtype=int);eps,ts=b['eps'][ix],b['ts'][ix];L=labels('dev');episodes=s1.episodes(L)
    frames=torch.empty(len(ix),63,63,3,dtype=torch.uint8);fix=torch.stack([cache[e]['fix'][t] for e,t in zip(eps,ts)])
    for e in np.unique(eps):
        ids=np.flatnonzero(eps==e);E=episodes[e];frames[ids]=D.RP.episode_fields(E.shard,E.slot)['observations'][ts[ids]]
    varmask=torch.zeros(81,147,dtype=torch.bool)
    for j,h in enumerate(HUD[:4]):
        px=proto[h]['pixels'];varmask[HUD_CELLS[j]]=px.max(0).values!=px.min(0).values
    cfg=torch.load(CKPT,map_location='cpu',weights_only=False,mmap=True)['cfg'];out={};trajectory={}
    for snapshot in (1,5,10):
        ck=torch.load(RUN/f'ckpt_{snapshot:02d}.pt',map_location='cpu',weights_only=False,mmap=True)
        model=M.M0(cfg).cuda().eval().requires_grad_(False);model.load_state_dict(ck['model']);del ck;mh=model_hash(model)
        rows={k:[] for k in ['mu','ls','raw','cell','g_gradient','varying_g_gradient','nll','g']}
        for ids in torch.arange(len(ix)).split(256):
            f,fi=frames[ids].cuda(),fix[ids].cuda();cells=M.window(fi)
            with torch.no_grad(),TR.autocast(torch.device('cuda'),cfg):P=model.perception(f,fi)
            g=P.g.detach().requires_grad_()
            with TR.autocast(torch.device('cuda'),cfg):mu,ls=model.pixels(model.tsp.reconstruct(g,P.v,cells),P.v,cells)
            raw=M.cell_pixels(M.crop(f,fi));loss=M.pixel_nll(mu,ls,raw)
            grad=torch.autograd.grad(loss.mean(-1).sum(),g,retain_graph=True)[0]
            varying=varmask[cells.cpu()].cuda();gv=torch.autograd.grad((loss*varying).mean(-1).sum(),g)[0]
            fields={'mu':mu.detach().float(),'ls':ls.detach().float(),'raw':raw,'cell':cells,'g_gradient':grad.float().norm(dim=-1),
                    'varying_g_gradient':gv.float().norm(dim=-1),'nll':loss.detach().float().mean(-1),'g':g.detach().float()}
            for key,z in fields.items():rows[key].append(z.cpu())
        rows={k:torch.cat(v) for k,v in rows.items()};trajectory[snapshot]=rows;out[snapshot]={}
        for j,h in enumerate(HUD[:4]):
            m=rows['cell']==HUD_CELLS[j];mu,ls,raw=rows['mu'][m].double(),rows['ls'][m].double(),rows['raw'][m]
            scale=ls.clamp(min=-7).exp();p0=torch.sigmoid((-1+1/255-mu)/scale);p255=torch.sigmoid((mu-1+1/255)/scale)
            px=proto[h]['pixels'];endpoint=((px==0)|(px==255)).all(0);vary=px.max(0).values!=px.min(0).values
            # Exact derivative of endpoint-bin NLL in radial coordinates (d_mu=mu, d_log_scale=1).
            mm=mu.clone().requires_grad_();ss=ls.clone().requires_grad_();nll=reference_nll(mm,ss,raw)
            gm,gs=torch.autograd.grad(nll.sum(),(mm,ss));rad=gm*mu+gs
            r={}
            for name,cols in [('all',torch.ones(147,dtype=torch.bool)),('binary_variable',endpoint&vary),('constant_endpoint',endpoint&~vary),('has_interior',~endpoint)]:
                if not cols.any():continue
                r[name]={'n_coordinates':int(cols.sum()),'scale_rgb_quantiles':torch.quantile((scale[:,cols]*127.5).flatten(),torch.tensor([.1,.5,.9],dtype=torch.float64)).tolist(),
                         'mean_endpoint_mass':float((p0[:,cols]+p255[:,cols]).mean()),'median_abs_location_scale_ratio':float((mu[:,cols]/scale[:,cols]).abs().median()),
                         'mean_radial_derivative':float(rad[:,cols].mean()),'mean_abs_location_derivative':float(gm[:,cols].abs().mean()),
                         'mean_bernoulli_limit_error':float((p255[:,cols]-torch.sigmoid(mu[:,cols]/scale[:,cols])).abs().mean())}
            r['glimpse_gradient_mean']=float(rows['g_gradient'][m].mean());r['glyph_varying_glimpse_gradient_mean']=float(rows['varying_g_gradient'][m].mean());r['nll']=float(rows['nll'][m].mean())
            out[snapshot][h]=r
        assert model_hash(model)==mh;save('scale_geometry.json',out);print('snapshot',snapshot,{h:(v['glimpse_gradient_mean'],v.get('binary_variable',{}).get('mean_endpoint_mass')) for h,v in out[snapshot].items()},flush=True)
        del model;torch.cuda.empty_cache()
    torch.save(trajectory,OUT/'scale_trajectory.pt');assert before==fingerprints()
