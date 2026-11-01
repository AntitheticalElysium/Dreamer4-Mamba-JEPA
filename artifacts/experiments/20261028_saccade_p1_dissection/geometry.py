"""CPU geometry-only domain tests, with physical-episode splits and a real-data null."""
from .common import *
import lightgbm as lgb


def stats(parts):
    feats=[];names=[]
    for name in ('p','g','v'):
        z=parts[name].half().float().numpy().reshape(len(parts[name]),-1,parts[name].shape[-1])
        vals={'rms':np.sqrt((z*z).mean(-1)),'mean':z.mean(-1),'absmax':np.abs(z).max(-1)}
        for measure,a in vals.items():
            for q in (.0,.1,.5,.9,1.):
                feats.append(np.quantile(a,q,axis=1));names.append(f'{name}/{measure}/q{q}')
    return np.stack(feats,1),names


if __name__=='__main__':
    b=torch.load(OUT/'baseline_samples.pt',weights_only=False,mmap=True)
    real,names=stats(b['real']);gen,_=stats(b['generated']);eps=b['eps']
    X=np.concatenate([real,gen]);y=np.repeat([0,1],len(eps));ee=np.tile(eps,2)
    tr=ee%10>=3;va=ee%10==2;te=ee%10<=1
    out={'n_test_pairs':int((eps%10<=1).sum()),'features':names,'precision':'both domains rounded to float16 before measurement'}
    for key in ('p','g','v','all','real_null'):
        cols=np.arange(len(names)) if key in ('all','real_null') else np.array([j for j,n in enumerate(names) if n.startswith(key+'/')])
        if key=='real_null':
            # Identical paired real features under opposite labels: exact null, no domain fingerprint.
            xx=np.concatenate([real,real])[:,cols]
        else:xx=X[:,cols]
        m=lgb.train({'objective':'binary','metric':'binary_logloss','num_leaves':15,'learning_rate':.03,
                     'verbosity':-1,'num_threads':6,'seed':0},lgb.Dataset(xx[tr],label=y[tr]),num_boost_round=400,
                    valid_sets=[lgb.Dataset(xx[va],label=y[va])],callbacks=[lgb.early_stopping(30,verbose=False)])
        pr=m.predict(xx[te]);good=(pr>=.5)==y[te]
        out[key]={'accuracy':G.estimate(good,np.ones(len(good),bool),ee[te]),'best_iteration':m.best_iteration,
                  'features':cols.tolist()}
    save('geometry_c2st.json',out)
    inverse={}
    for domain in ('real','generated'):
        inverse[domain]={}
        for key in ('p','g','v'):
            z=b[domain][key].half().float()
            q=3*torch.atanh((z/3).clamp(-.999999,.999999))
            inverse[domain][key]={'inverse_mean_absolute_quantiles':torch.quantile(q.mean(-1).abs().flatten(),torch.tensor([.5,.9,.99])).tolist(),
                                  'inverse_variance_quantiles':torch.quantile(q.var(-1,unbiased=False).flatten(),torch.tensor([.01,.5,.99])).tolist()}
    save('inverse_encoder_geometry.json',inverse)
    print({k:v['accuracy']['est'] for k,v in out.items() if isinstance(v,dict) and 'accuracy' in v},flush=True)
