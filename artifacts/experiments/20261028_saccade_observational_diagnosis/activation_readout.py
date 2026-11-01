"""Read-only activations along the fovea-to-pixel path, then train measurement readers only."""
import torch.nn.functional as F
from .common import *
fm=__import__('importlib').import_module('artifacts.experiments.20261028_saccade_p1_dissection.fovea_readers')


@torch.no_grad()
def extract(model,cfg,split):
    cache=load_cache(split);tab=load_crops(split);result={k:[] for k in ['g','adapter','hidden','mu','display','params','cdf_logits','scale','v','cell','value','ep','night']};offset=0
    for e,r in enumerate(cache):
        n=len(r['g'])
        for start in range(0,n,512):
            stop=min(start+512,n);g=r['g'][start:stop].cuda();v=r['v'][start:stop].cuda();cells=M.window(r['fix'][start:stop].cuda())
            with TR.autocast(torch.device('cuda'),cfg):
                h=model.tsp.reconstruct(g,v,cells);mu,ls=model.pixels(h,v,cells);hidden=model.pixels.mean[:-1](h)
                assert torch.equal(model.pixels.mean[-1](hidden),mu)
            rr=torch.arange((offset+start)*9,(offset+stop)*9);mask=tab['value'][rr]>=0;vv=v[:,None].expand(-1,9,-1).reshape(-1,8).cpu()[mask]
            m,s=mu.float(),ls.float().clamp(min=-7);inv=(-s).exp();a=(-1+1/255-m)*inv;b=(m-1+1/255)*inv
            fields={'g':g,'adapter':h,'hidden':hidden,'mu':m,'display':m.clamp(-1,1),'params':torch.cat([m,s],-1),'cdf_logits':torch.cat([a,b],-1),'scale':s}
            for key,z in fields.items():result[key].append(z.float().reshape(len(rr),-1).cpu()[mask])
            result['v'].append(vv)
            for key in ('cell','value','ep','night'):result[key].append(tab[key][rr][mask])
        offset+=n
    result={k:torch.cat(z) for k,z in result.items()};torch.save(result,OUT/f'activation_{split}.pt');print('captured',split,len(result['cell']),flush=True)


if __name__=='__main__':
    before=fingerprints();model,cfg=load_model();mh=model_hash(model)
    for split in ('train','dev'):extract(model,cfg,split)
    assert model_hash(model)==mh;del model;torch.cuda.empty_cache()
    tr=torch.load(OUT/'activation_train.pt',weights_only=False,mmap=True);dv=torch.load(OUT/'activation_dev.pt',weights_only=False,mmap=True)
    ti=torch.where(tr['ep']%10!=9)[0];vi=torch.where(tr['ep']%10==9)[0];di=torch.arange(len(dv['cell']));y=tr['value'].long().cuda()
    out={}
    for arm in ['g','adapter','hidden','mu','display','params','cdf_logits','scale']:
        mean=tr[arm][ti].mean(0);std=tr[arm][ti].std(0).clamp(min=1e-6)
        def feature(t,ids):
            z=(t[arm][ids]-mean)/std;return torch.cat([F.pad(z,(0,294-z.shape[1])),t['v'][ids],F.one_hot(t['cell'][ids].long(),81).float()],1)
        X=feature(tr,torch.arange(len(tr['cell']))).cuda();it,iv=ti.cuda(),vi.cuda()
        mx=lambda idx,val:X[(iv if val else it)[idx]];lf=lambda o,idx,val:F.cross_entropy(o['value'],y[(iv if val else it)[idx]])
        for seed in (0,1):
            rd,log=s1.fit(mx,lf,len(ti),len(vi),{'value':10},seed)
            with torch.no_grad():p=torch.cat([rd(feature(dv,q).cuda())['value'].cpu() for q in di.split(16384)])
            metrics=fm.metrics({'value':p},dv,di,'hud');out[f'{arm}/{seed}']={'metrics':metrics,'validation_loss':log,'best_epoch':int(np.argmin(log))}
            save('activation_readout.json',out);print(arm,seed,[round(metrics[h]['exact']['est'],3) for h in HUD[:4]],flush=True)
        del X;torch.cuda.empty_cache()
    assert before==fingerprints()
