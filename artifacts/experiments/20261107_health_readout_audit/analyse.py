"""CPU interpretation of the declared frozen readout audit. No learned-world changes."""
import os
os.environ.setdefault('JAX_PLATFORMS','cpu')
import importlib,json
from pathlib import Path
import numpy as np
import torch
RT=importlib.import_module('artifacts.experiments.20261106_determinant_routes.routes')
HERE=Path(__file__).resolve().parent

def boot_mean(x,ep,w=None,n=1000):
    x=np.asarray(x,float);w=np.ones(len(x)) if w is None else np.asarray(w,float)
    _,ee=np.unique(ep,return_inverse=True);a=np.bincount(ee,weights=w*x);b=np.bincount(ee,weights=w)
    rng=np.random.default_rng(110716);ii=rng.integers(0,len(a),(n,len(a)))
    v=a[ii].sum(1)/b[ii].sum(1)
    return {'mean':float((w*x).sum()/w.sum()),'ci95':np.quantile(v,[.025,.975]).tolist()}

def probe(X,y,fold):
    X=X.double();s=np.zeros(len(y))
    for f in range(5):
        m=fold==f;s[m]=RT.logistic(X[~m],y[~m],X[m])
    return s

def main():
    torch.set_num_threads(2)
    z=torch.load(HERE/'capture.pt',map_location='cpu',weights_only=False)
    s,T,rows,ep,fold,w=z['states'],z['T'],z['rows'],z['ep'],z['fold'],z['w']
    y=(T['true_body'][rows,0]<T['body'][rows,0]).astype(float)
    prev=T['body'][rows,0].clip(0,9)
    report={'rows':len(rows),'reader_validity':z['reader_validity'],'checkpoint_sha256':z['checkpoint_sha256'],'populations':{}}
    W=z['old_weight'][:-1];Q=torch.linalg.qr(W).Q
    feat={**{k:s['native0/'+k] for k in ('raw','norm','mod','out')},
          'output_current_digit_interaction':torch.cat([s['native0/out'][:,None,:]*F_onehot(prev)[:,:,None],F_onehot(prev)[:,:,None]],-1).flatten(1),
          'output_nullspace':s['native0/mod'].double()-(s['native0/mod'].double()@Q)@Q.T}
    for name,mask in [('awake_adjacent',z['groups']==0),('asleep_adjacent',z['groups']==1)]:
        yy=y[mask];ee=ep[mask];ff=fold[mask];out={'n':int(mask.sum()),'true':float(yy.mean()),'ap':{}}
        for key,X in feat.items():
            sc=probe(X[mask],yy,ff)
            out['ap'][key]={'ap':float(RT.G.average_precision(yy,sc)),
                           'brier':float(np.mean((sc-yy)**2))}
        d=s['truth'][mask]-s['copy'][mask];dd=d.square().sum(1).clamp(min=1e-8);hit=yy>0
        out['lean_on_recorded_hit']={}
        for key in ('native0/out','fine0/out','teacher','native3/out'):
            v=(((s[key][mask]-s['copy'][mask])*d).sum(1)/dd).numpy()
            out['lean_on_recorded_hit'][key]=boot_mean(v[hit],ee[hit])
        for key in ('native0/out','fine0/out','teacher'):
            err=(s[key][mask]-s['truth'][mask]).square().mean(1).numpy()
            out.setdefault('token_mse',{})[key]=float(err.mean())
        report['populations'][name]=out;print(name,json.dumps(out),flush=True)
    # Shared-head exact frozen squared-loss derivatives; no Adam/clipping claims.
    g={}
    for name,st in z['stats'].items():
        xx=sum(a[0] for a in st);xy=sum(a[1] for a in st);n=sum(a[3] for a in st)
        g[name]=(xx@z['old_weight']-xy)/n
    g['other']=(81*g['shared']-g['health'])/80
    hitw=w*(.1+.9*s['flow_tau'].numpy())*y
    X=torch.cat([s['flow_x'].double(),torch.ones(len(rows),1,dtype=torch.float64)],1)
    Y=s['flow_y'].double();ww=torch.as_tensor(hitw)[:,None]
    g['health_hit']=X.T@(ww*(X@z['old_weight']-Y))/ww.sum()
    report['gradient_geometry']={'norms':{k:float(v.norm()) for k,v in g.items()},
                                'cosines':{a+'__'+b:float((g[a]*g[b]).sum()/(g[a].norm()*g[b].norm()))
                                           for a,b in [('health','shared'),('health','other'),('health_hit','shared'),('health_hit','health')]}}
    eig=torch.linalg.svdvals(W)
    report['linear_head']={'rank':int(torch.linalg.matrix_rank(W)),'singular_values':eig.tolist()}
    (HERE/'STAGES.json').write_text(json.dumps(report,indent=2))

def F_onehot(y):
    return torch.nn.functional.one_hot(torch.as_tensor(y).long(),10).float()

def native():
    z=torch.load(HERE/'capture.pt',map_location='cpu',weights_only=False)
    n=torch.load(HERE/'native.pt',map_location='cpu',weights_only=False)
    T,rows,ep,w=z['T'],z['rows'],z['ep'],z['w']
    y=(T['true_body'][rows,0]<T['body'][rows,0]).astype(float);prev=T['body'][rows,0]
    result={}
    for name,mask in [('natural_all',np.ones(len(rows),bool)),('awake_adjacent',z['groups']==0),('asleep_adjacent',z['groups']==1),
                      ('awake_nonadjacent',z['groups']==2),('asleep_nonadjacent',z['groups']==3)]:
        r={'n':int(mask.sum()),'true_drop':boot_mean(y[mask],ep[mask],w[mask]),'arms':{}}
        for arm in ('old','shared','health'):
            dig=n[arm+'/digit'].numpy();k=(dig<prev[:,None]).sum(1);p=k/4
            fair=k*(k-1)/12-2*p*y+y
            r['arms'][arm]={'drop':boot_mean(p[mask],ep[mask],w[mask]),
                            'fair_brier':boot_mean(fair[mask],ep[mask],w[mask]),
                            'rise':boot_mean((dig>prev[:,None]).mean(1)[mask],ep[mask],w[mask]),
                            'entity_count':boot_mean(n[arm+'/entity_count'].float().mean(1).numpy()[mask],ep[mask],w[mask]),
                            'sample_mse_descriptive':boot_mean(n[arm+'/sample_mse'].mean(1).numpy()[mask],ep[mask],w[mask])}
            for lab,mm in [('recorded_hit',mask&(y>0)),('recorded_nohit',mask&(y==0))]:
                if mm.any():r['arms'][arm][lab]=boot_mean(p[mm],ep[mm],w[mm])
        for arm in ('shared','health'):
            kk=(n[arm+'/digit'].numpy()<prev[:,None]).sum(1);ko=(n['old/digit'].numpy()<prev[:,None]).sum(1)
            df=(kk*(kk-1)-ko*(ko-1))/12-2*(kk-ko)/4*y
            r['arms'][arm]['paired_fair_brier_change']=boot_mean(df[mask],ep[mask],w[mask])
        result[name]=r;print(name,json.dumps(r),flush=True)
    (HERE/'NATIVE.json').write_text(json.dumps(result,indent=2))

if __name__=='__main__':
    import sys
    (native if '--native' in sys.argv else main)()
