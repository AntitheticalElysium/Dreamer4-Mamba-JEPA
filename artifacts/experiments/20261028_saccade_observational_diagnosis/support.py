"""Real-token reference support, calibrated before reading existing generated samples."""
import time
from scipy.spatial import cKDTree
from saccade.eval.probe import KEEP
from .common import *


def all_real(split):
    cache=load_cache(split);L=labels(split);P=torch.cat([r['p'] for r in cache])[:,:63].half().numpy()
    V=torch.cat([r['v'] for r in cache]).half().numpy();ep=episode_rows(L)
    y=np.concatenate([e['grid'] for e in previous.grids(L)]).reshape(-1,63)
    return P,V,ep,y


def sample_ids(ep,held,n,rng):
    frames=np.flatnonzero((ep%10==9)==held);cells=np.flatnonzero(KEEP)
    q=rng.choice(len(frames)*len(cells),n,replace=False)
    return frames[q//len(cells)],cells[q%len(cells)]


def summarize(pred,agreement,dist,threshold,true,ep,frame):
    valid=(dist<=threshold[pred])&(agreement>=7/8);out={'n':len(pred),'support_fraction':float(valid.mean()),
       'distance_quantiles':np.quantile(dist,[.1,.5,.9,.99]).tolist(),'class':{}}
    for k,name in enumerate(previous.KINDS):
        m=pred==k;v=m&valid
        out['class'][name]={'n_predicted':int(m.sum()),'n_supported':int(v.sum()),'mean_distance':float(dist[m].mean()) if m.any() else None}
        if true is not None:
            actual=true==k
            out['class'][name].update({'precision':G.estimate(true==k,m,ep),'recall':G.estimate(pred==k,actual,ep),
                 'supported_precision':G.estimate(true==k,v,ep),'supported_recall':G.estimate((pred==k)&valid,actual,ep)})
    if frame is not None:
        out['counts_per_frame']={name:{'all_nn':float((pred==k).sum()/len(np.unique(frame))),
             'supported_nn':float(((pred==k)&valid).sum()/len(np.unique(frame)))} for k,name in enumerate(previous.KINDS[1:],1)}
    return out,valid


if __name__=='__main__':
    before=fingerprints();start=time.time();rng=np.random.default_rng(447);P,V,ep,y=all_real('train')
    rf,rc=sample_ids(ep,False,300000,rng);cf,cc=sample_ids(ep,True,100000,rng)
    rp,rv=P[rf,rc].astype(np.float32),V[rf].astype(np.float32);ry=y[rf,rc]
    cp,cv=P[cf,cc].astype(np.float32),V[cf].astype(np.float32);cy=y[cf,cc];ce=ep[cf]
    b=torch.load(OLD/'baseline_samples.pt',weights_only=False,mmap=True);ix=np.linspace(0,29999,4096,dtype=int);cells=np.flatnonzero(KEEP)
    query={d:{'p':b['real' if d=='real' else 'generated']['p'][ix,:63][:,KEEP].half().float().reshape(-1,16).numpy(),
              'v':b['real' if d=='real' else 'generated']['v'][ix].repeat_interleave(len(cells),0).half().float().numpy()} for d in ('real','generated')}
    L=labels('dev');off=np.r_[0,np.cumsum(L['ep_T']+1)]
    grid=np.concatenate([e['grid'] for e in previous.grids(L)]).reshape(-1,63)
    qy=grid[off[b['eps'][ix]]+b['ts'][ix]][:,KEEP].ravel();qe=np.repeat(b['eps'][ix],len(cells));frame=np.repeat(np.arange(len(ix)),len(cells))
    out={};saved={}
    for space in ('p','pv'):
        specs=[('p',rp,cp)]+([('v',rv,cv)] if space=='pv' else [])
        centers={};scales={}
        for name,a,_ in specs:
            centers[name]=a.mean(0);scales[name]=float(np.sqrt(a.var(0).mean())*np.sqrt(a.shape[1]))
        transform=lambda d:np.concatenate([(d[name]-centers[name])/max(scales[name],1e-6) for name,_,_ in specs],1)
        R=transform({'p':rp,'v':rv});C=transform({'p':cp,'v':cv});tree=cKDTree(R,leafsize=32)
        def read(X):
            ds=[];ps=[];ag=[]
            for i in range(0,len(X),8192):
                dist,idx=tree.query(X[i:i+8192],k=8,workers=6);lbl=ry[idx];pred=lbl[:,0]
                ds.append(dist[:,0]);ps.append(pred);ag.append((lbl==pred[:,None]).mean(1))
            return np.concatenate(ds),np.concatenate(ps),np.concatenate(ag)
        dc,pc,ac=read(C);glob=np.quantile(dc,.95);threshold=np.array([np.quantile(dc[pc==k],.95) if (pc==k).sum()>=50 else glob for k in range(5)])
        out[space]={'scale':scales,'calibration_thresholds':threshold.tolist(),'calibration_predicted_counts':np.bincount(pc,minlength=5).tolist(),
                    'calibration':summarize(pc,ac,dc,threshold,cy,ce,None)[0]}
        for domain in ('real','generated'):
            d,p,a=read(transform(query[domain]));s,valid=summarize(p,a,d,threshold,qy if domain=='real' else None,qe,frame)
            original=b['real_reader' if domain=='real' else 'generated_reader'][ix][:,KEEP].numpy().ravel()
            s['neural_nn_agreement']=float((original==p).mean());s['supported_neural_nn_agreement']=float((original[valid]==p[valid]).mean()) if valid.any() else None
            s['neural_entity_nn_supported_background_fraction']=float(((original>0)&(p==0)&valid).mean())
            out[space][domain]=s;saved[f'{space}/{domain}']={'nn':p,'distance':d,'agreement':a,'supported':valid,'neural':original}
            print(space,domain,'support',s['support_fraction'],'agreement',s['neural_nn_agreement'],flush=True)
        save('reference_support.json',out);torch.save({'tuple_indices':ix,'observations':saved},OUT/'reference_support.pt')
        del tree
    assert before==fingerprints()
    save('support_run.json',{'seconds':time.time()-start,'n_reference':300000,'n_calibration':100000,'n_population_tuples':4096,'source_hashes_unchanged':True})
