"""Whole-crop diagnostic readers: check distributed fovea information before claiming loss."""
import torch.nn.functional as F
from .common import *


def data(split):
    t=torch.load(OUT/f'crops_{split}.pt',weights_only=False,mmap=True);n=len(t['ep'])//9
    q={k:v.reshape(n,9,*v.shape[1:]) for k,v in t.items()};q['ep']=q['ep'][:,0]
    q['v']=q['v'][:,0];q['kind']=q['kind'].long();q['kind'][q['cell']==31]=-1
    q['presence']=torch.stack([(q['kind']==k).any(-1) for k in range(1,5)],-1).float()
    q['fix']=((q['cell'][:,4]//9-1)*7+q['cell'][:,4]%9-1).long()
    return q


def features(q,arm):
    if arm=='periphery':z=q['p'].flatten(1).float()
    elif arm=='fovea':z=q['g'].flatten(1).float()
    elif arm=='joint':z=torch.cat([q['p'],q['g']],-1).flatten(1).float()
    elif arm=='pixels_full':z=q['px'].flatten(1).float()/127.5-1
    else:raise ValueError(arm)
    x=torch.zeros(len(z),1323);x[:,:z.shape[1]]=z
    return torch.cat([x,q['v'].float(),F.one_hot(q['fix'],49).float()],1)


if __name__=='__main__':
    before=fingerprints();tr,dv=data('train'),data('dev');held=tr['ep']%10==9;out={}
    for task in ('presence','spatial'):
        heads={'presence':4} if task=='presence' else {'kind':45}
        y=tr['presence'].cuda() if task=='presence' else tr['kind'].cuda();yy=dv['presence'].numpy() if task=='presence' else dv['kind'].numpy()
        ti=torch.where(~held)[0].cuda();vi=torch.where(held)[0].cuda()
        for arm in ('periphery','fovea','joint','pixels_full'):
            X=features(tr,arm).cuda();Xd=features(dv,arm)
            mx=lambda idx,val:X[(vi if val else ti)[idx]]
            def loss(o,idx,val):
                rows=(vi if val else ti)[idx]
                return F.binary_cross_entropy_with_logits(o['presence'],y[rows]) if task=='presence' else F.cross_entropy(o['kind'].reshape(-1,5),y[rows].reshape(-1),ignore_index=-1)
            for seed in (0,1):
                rd,log=s1.fit(mx,loss,len(ti),len(vi),heads,seed)
                with torch.no_grad():o=torch.cat([next(iter(rd(x.cuda()).values())).cpu() for x in Xd.split(8192)])
                if task=='presence':
                    pr=o.sigmoid().numpy();pred=pr>.5
                    res={name:{'ap':G.average_precision(yy[:,j],pr[:,j]),'recall':G.estimate(pred[:,j],yy[:,j]>0,dv['ep'].numpy()),
                          'precision':G.estimate(yy[:,j],pred[:,j],dv['ep'].numpy())} for j,name in enumerate(KINDS[1:])}
                else:
                    pr=o.reshape(-1,9,5).softmax(-1).numpy();pred=pr.argmax(-1);ep=np.repeat(dv['ep'].numpy(),9)
                    res={}
                    for ln,lm in [('day',~dv['night'].numpy()),('night',dv['night'].numpy())]:
                        res[ln]={name:G.estimate((pred==k).ravel(),((yy==k)&lm).ravel(),ep) for k,name in enumerate(KINDS[1:],1)}
                out[f'{task}/{arm}/{seed}']={'metrics':res,'validation_loss':log,'best_epoch':int(np.argmin(log)),
                     'input_dim':int(X.shape[1]),'n_train_crops':len(ti),'n_val_crops':len(vi),'n_dev_crops':len(Xd)}
                save('crop_readers.json',out);print(task,arm,seed,np.argmin(log),flush=True)
            del X,Xd;torch.cuda.empty_cache()
    assert before==fingerprints()
