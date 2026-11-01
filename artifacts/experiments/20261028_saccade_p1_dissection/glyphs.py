"""Renderer-grounded glyph-template check, only on REAL crop reconstructions."""
from .common import *


def white(x,threshold):return (x.reshape(-1,7,7,3)[:,2:,2:].amin(-1)>threshold).reshape(-1,25).float()


@torch.no_grad()
def main():
    before=fingerprints();tr=torch.load(OUT/'crops_train.pt',weights_only=False,mmap=True)
    dv=torch.load(OUT/'crops_dev.pt',weights_only=False,mmap=True);L=labels('dev');off=np.r_[0,np.cumsum(L['ep_T']+1)]
    b=torch.load(OUT/'baseline_samples.pt',weights_only=False,mmap=True);cache=torch.load(OUT/'real_dev.pt',weights_only=False,mmap=True)
    eps,ts=b['eps'],b['ts'];idx=torch.as_tensor(((off[eps]+ts)[:,None]*9+np.arange(9)).ravel())
    raw=dv['px'][idx].float();cell=dv['cell'][idx].numpy();truth=dv['value'][idx].numpy();dec=[]
    model,cfg=load_model();mh=model_hash(model)
    for ids in torch.arange(len(eps)).split(256):
        g,v=b['real']['g'][ids].cuda(),b['real']['v'][ids].cuda();fix=torch.stack([cache[e]['fix'][t] for e,t in zip(eps[ids],ts[ids])]).cuda();cells=M.window(fix)
        with TR.autocast(torch.device('cuda'),cfg):mu,_=model.pixels(model.tsp.reconstruct(g,v,cells),v,cells)
        dec.append(((mu.float()+1)*127.5).clamp(0,255).reshape(-1,147).cpu())
    dec=torch.cat(dec);out={}
    for threshold in (210,240,250):
        out[threshold]={}
        for name,c in zip(HUD,HUD_CELLS):
            m=tr['cell']==c;tx=white(tr['px'][m],threshold);y=tr['value'][m].numpy();classes=np.unique(y)
            templates=torch.stack([tx[y==v].mean(0) for v in classes]);dm=cell==c
            results={}
            for domain,x in [('raw',raw[dm]),('reconstructed',dec[dm])]:
                binary=white(x,threshold);dist=(binary[:,None]-templates[None]).square().sum(-1)
                pred=classes[dist.argmin(-1).numpy()];yy=truth[dm]
                results[domain]={'n':len(pred),'exact':float((pred==yy).mean()),'prediction_histogram':np.bincount(pred,minlength=10).tolist(),
                                 'truth_histogram':np.bincount(yy,minlength=10).tolist()}
            out[threshold][name]=results
    save('glyphs.json',out)
    assert before==fingerprints() and model_hash(model)==mh


if __name__=='__main__':main()
