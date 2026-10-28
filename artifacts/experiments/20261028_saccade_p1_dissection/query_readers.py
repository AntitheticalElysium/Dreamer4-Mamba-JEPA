"""Shared cell-query readers of distributed fovea content, with raw-pixel power control."""
import torch.nn.functional as F
from .common import *
from .fovea_readers import metrics


def prepare(t,arm,device):
    n=len(t['cell'])//9
    return {'z':t['px'].reshape(n,9,147).to(device) if arm=='raw_query' else t['g'].reshape(n,9,64).to(device),
            'v':t['v'].reshape(n,9,8)[:,0].to(device)}


def feature(q,ids,arm):
    fr,lo=ids//9,ids%9
    if arm=='g_context':
        order=(lo[:,None]+torch.arange(9,device=lo.device))%9;z=q['z'][fr[:,None],order].flatten(1).float()
    else:z=q['z'][fr,lo].float()
    if arm=='raw_query':z=z/127.5-1
    z=F.pad(z,(0,1323-z.shape[1]))
    return torch.cat([z,q['v'][fr].float(),F.one_hot(lo,9).float()],1)


if __name__=='__main__':
    before=fingerprints();tr=torch.load(OUT/'crops_train.pt',weights_only=False,mmap=True);dv=torch.load(OUT/'crops_dev.pt',weights_only=False,mmap=True)
    valid=lambda t:(t['cell']<63)&(t['cell']!=31)
    ti=torch.where(valid(tr)&(tr['ep']%10!=9))[0].cuda();vi=torch.where(valid(tr)&(tr['ep']%10==9))[0].cuda()
    di=torch.where(valid(dv))[0];kind=tr['kind'].long().cuda();terr=tr['terr'].long().cuda();terr[kind>0]=-1
    out={}
    for arm in ('g_query','g_context','raw_query'):
        q=prepare(tr,arm,'cuda');qd=prepare(dv,arm,'cpu')
        mx=lambda ids,val:feature(q,(vi if val else ti)[ids],arm)
        def loss(o,ids,val):
            r=(vi if val else ti)[ids]
            return F.cross_entropy(o['kind'],kind[r])+F.cross_entropy(o['terr'],terr[r],ignore_index=-1)
        for seed in (0,1):
            rd,log=s1.fit(mx,loss,len(ti),len(vi),{'kind':5,'terr':17},seed)
            with torch.no_grad():
                pred=torch.cat([rd(feature(qd,ix,arm).cuda())['kind'].cpu() for ix in di.split(8192)])
            res=metrics({'kind':pred},dv,di,'map');out[f'{arm}/{seed}']={'metrics':res,'validation_loss':log,'best_epoch':int(np.argmin(log))}
            save('query_readers.json',out);print(arm,seed,np.argmin(log),res['night/r1_2']['arrows_recall'],flush=True)
        del q,qd;torch.cuda.empty_cache()
    assert before==fingerprints()
