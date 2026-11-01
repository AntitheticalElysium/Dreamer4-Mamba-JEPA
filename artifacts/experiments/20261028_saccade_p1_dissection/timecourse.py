"""Re-encode identical real frames through each snapshot's OWN perception/decoder."""
from .common import *


@torch.no_grad()
def main():
    before=fingerprints();cache=torch.load(OUT/'real_dev.pt',weights_only=False,mmap=True)
    b=torch.load(OUT/'baseline_samples.pt',weights_only=False,mmap=True);inds=np.linspace(0,len(b['eps'])-1,8192,dtype=int)
    es,ts=b['eps'][inds],b['ts'][inds];L=labels('dev');ep=s1.episodes(L)
    frames=torch.empty(len(es),63,63,3,dtype=torch.uint8);fix=torch.stack([cache[e]['fix'][t] for e,t in zip(es,ts)])
    for e in np.unique(es):
        ids=np.flatnonzero(es==e);fr=D.RP.episode_fields(ep[e].shard,ep[e].slot)['observations'];frames[ids]=fr[ts[ids]]
    cells=M.window(fix).numpy();out={}
    cfg=torch.load(CKPT,weights_only=False,map_location='cpu')['cfg']
    for snapshot in range(11):
        torch.manual_seed(1);model=M.M0(cfg).cuda().eval().requires_grad_(False)
        if snapshot:
            file=RUN/f'ckpt_{snapshot:02d}.pt';ck=torch.load(file,weights_only=False,map_location='cpu');model.load_state_dict(ck['model']);del ck
        obs=[]
        for ids in torch.arange(len(es)).split(256):
            f=frames[ids].cuda();fi=fix[ids].cuda();c=M.window(fi);pix=M.cell_pixels(M.crop(f,fi))
            with TR.autocast(torch.device('cuda'),cfg):
                P=model.perception(f,fi);mean,ls=model.pixels(model.tsp.reconstruct(P.g,P.v,c),P.v,c)
            obs.append({'nll':M.pixel_nll(mean,ls,pix).mean(-1).cpu().numpy(),
                        'mae':((((mean.float()+1)*127.5).clamp(0,255)-pix.float()).abs().mean(-1)).cpu().numpy(),
                        'scale':(ls.float().clamp(min=-7).exp()*127.5).mean(-1).cpu().numpy()})
        S={k:np.concatenate([o[k] for o in obs]) for k in obs[0]};out[snapshot]={}
        for name,m in {'map':cells<63,'HUD':cells>=63,**{h:cells==c for h,c in zip(HUD,HUD_CELLS)}}.items():
            if m.any():out[snapshot][name]={'n':int(m.sum()),**{k:G.estimate(v.ravel(),m.ravel(),np.repeat(es,9)) for k,v in S.items()}}
        save('decoder_timecourse.json',out);print('snapshot',snapshot,{k:out[snapshot][k]['scale']['est'] for k in ['map','HUD','food']},flush=True)
        del model;torch.cuda.empty_cache()
    assert before==fingerprints()


if __name__=='__main__':main()
