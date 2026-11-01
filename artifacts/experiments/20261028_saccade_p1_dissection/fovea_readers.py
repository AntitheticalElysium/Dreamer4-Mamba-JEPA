"""Matched post-hoc readers of the SAME selected cells; never train the world model."""
import time
import torch.nn.functional as F
from .common import *
from saccade.eval.probe import RING


def features(tab, ids, arm, task):
    n=len(ids); x=torch.zeros(n,147)
    if arm=='periphery': z=tab['p'][ids].float()
    elif arm=='fovea': z=tab['g'][ids].float()
    elif arm=='joint': z=torch.cat([tab['p'][ids],tab['g'][ids]],1).float()
    elif arm=='pixels_low': z=tab['low'][ids].float()/127.5-1
    elif arm=='pixels_full': z=tab['px'][ids].float()/127.5-1
    else: raise ValueError(arm)
    x[:,:z.shape[1]]=z
    x=torch.cat([x,tab['v'][ids].float()],1)
    if task=='hud':x=torch.cat([x,F.one_hot(tab['cell'][ids].long(),81).float()],1)
    return x


@torch.no_grad()
def predict(reader,tab,ids,arm,task):
    out={k:[] for k in reader.heads}
    for q in ids.split(16384):
        o=reader(features(tab,q,arm,task).cuda())
        for k,z in o.items():out[k].append(z.cpu())
    return {k:torch.cat(z) for k,z in out.items()}


def metrics(o,tab,ids,task):
    ep=tab['ep'][ids].numpy(); night=tab['night'][ids].numpy(); cell=tab['cell'][ids].numpy()
    if task=='map':
        y=tab['kind'][ids].numpy();pred=o['kind'].argmax(-1).numpy()
        ring=RING[cell];res={}
        for ln,lm in (('day',~night),('night',night)):
            for rn,rm in (('all',True),('r1_2',ring<=2)):
                m=lm&rm
                res[f'{ln}/{rn}']={f'{name}_{measure}':G.estimate((pred==k) if measure=='recall' else (y==k),
                         m&((y==k) if measure=='recall' else (pred==k)),ep)
                         for k,name in enumerate(KINDS[1:],1) for measure in ('recall','precision')}
        return res
    y=tab['value'][ids].numpy();pr=o['value'].softmax(-1).numpy();pred=pr.argmax(-1)
    res={}
    for name,c in zip(HUD,HUD_CELLS):
        m=cell==c
        if not m.any():continue
        classes=np.unique(y[m]);err=np.abs(pred-y)
        res[name]={'n':int(m.sum()),'exact':G.estimate(pred==y,m,ep),'mae':G.estimate(err,m,ep),
                   'within_one':G.estimate(err<=1,m,ep),
                   'balanced_recall':float(np.mean([(pred[m&(y==k)]==k).mean() for k in classes])),
                   'positive_ap':G.average_precision((y[m]>0).astype(int),1-pr[m,0]),
                   'majority':float(np.bincount(y[m],minlength=10).max()/m.sum()),
                   'confusion':np.bincount((y[m]*10+pred[m]),minlength=100).reshape(10,10).tolist()}
    return res


if __name__=='__main__':
    start=time.time();before=fingerprints();out={}
    tr=torch.load(OUT/'crops_train.pt',weights_only=False,mmap=True)
    dv=torch.load(OUT/'crops_dev.pt',weights_only=False,mmap=True)
    for task in ('map','hud'):
        # Identical instances across all representations; player sprite and blank HUD cells excluded.
        valid=lambda t: ((t['cell']<63)&(t['cell']!=31)) if task=='map' else t['value']>=0
        ti=torch.where(valid(tr)&(tr['ep']%10!=9))[0];vi=torch.where(valid(tr)&(tr['ep']%10==9))[0]
        di=torch.where(valid(dv))[0]
        heads={'kind':5,'terr':17} if task=='map' else {'value':10}
        for arm in ('periphery','fovea','joint','pixels_low','pixels_full'):
            X=[features(tr,q,arm,task).cuda() for q in (ti,vi)]
            yy={k:[tr['kind' if k=='kind' else 'terr' if k=='terr' else 'value'][q].long().cuda() for q in (ti,vi)] for k in heads}
            if task=='map':
                for j,q in enumerate((ti,vi)):yy['terr'][j][yy['kind'][j]>0]=-1
            mx=lambda idx,val:X[int(val)][idx]
            def loss(o,idx,val):
                return sum(F.cross_entropy(o[k],yy[k][int(val)][idx],ignore_index=-1) for k in heads)
            for seed in (0,1):
                reader,log=s1.fit(mx,loss,len(ti),len(vi),heads,seed)
                o=predict(reader,dv,di,arm,task);res=metrics(o,dv,di,task)
                key=f'{task}/{arm}/{seed}';out[key]={'metrics':res,'validation_loss':log,'best_epoch':int(np.argmin(log)),
                     'n_train':len(ti),'n_val':len(vi),'n_dev':len(di),'input_dim':X[0].shape[1]}
                torch.save({'state':{k:v.cpu() for k,v in reader.state_dict().items()},'input_dim':X[0].shape[1],'heads':heads,
                            'dev_indices':di,'predictions':{k:v.argmax(-1).byte() for k,v in o.items()}},OUT/f'reader_{task}_{arm}_{seed}.pt')
                save('fovea_readers.json',out); print(key, 'best',np.argmin(log),flush=True)
                del reader,o
            del X,yy;torch.cuda.empty_cache()
    assert fingerprints()==before
    save('fovea_readers_run.json',{'seconds':time.time()-start,'world_sources_unchanged':True,
          'peak_gpu_gib':torch.cuda.max_memory_allocated()/2**30})
