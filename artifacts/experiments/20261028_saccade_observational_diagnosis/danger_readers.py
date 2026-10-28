"""Direct danger readers, not a new world model or actor."""
import torch.nn.functional as F
from .common import *


if __name__=='__main__':
    before=fingerprints();cache={s:load_cache(s) for s in ('train','dev')};L={s:labels(s) for s in cache}
    ep={s:episode_rows(L[s]) for s in cache};alive={s:hud_values(L[s])[:,0]>0 for s in cache}
    yi={s:hud_values(L[s])[:,:4] for s in cache};target={s:np.column_stack([yi[s][:,0]<=2,*[yi[s][:,j]==0 for j in range(1,4)]]).astype(np.float32) for s in cache}
    ids={False:torch.from_numpy(np.flatnonzero(alive['train']&(ep['train']%10!=9))).cuda(),
         True:torch.from_numpy(np.flatnonzero(alive['train']&(ep['train']%10==9))).cuda()}
    di=torch.from_numpy(np.flatnonzero(alive['dev']));Y=torch.from_numpy(target['train']).cuda();out={}
    for rep in ('full_percept','x','b'):
        features={s:s1.frame_features(previous.half_encoding(cache[s]))[:,:-17] if rep=='full_percept' else torch.cat([r[rep] for r in cache[s]]) for s in cache}
        X=features['train'].cuda();Xd=features['dev'];del features
        for j,name in enumerate(HUD[:4]):
            mx=lambda q,val:X[ids[val][q]]
            lf=lambda o,q,val:F.binary_cross_entropy_with_logits(o['critical'][:,0],Y[ids[val][q],j])
            for seed in (0,1):
                rd,log=s1.fit(mx,lf,len(ids[False]),len(ids[True]),{'critical':1},seed)
                with torch.no_grad():
                    pv=torch.cat([rd(X[q])['critical'][:,0].sigmoid().cpu() for q in ids[True].split(16384)]).numpy()
                    pd=torch.cat([rd(Xd[q].cuda())['critical'][:,0].sigmoid().cpu() for q in di.split(16384)]).numpy()
                vv=target['train'][ids[True].cpu().numpy(),j];yy=target['dev'][di.numpy(),j].astype(bool);ee=ep['dev'][di.numpy()]
                pos=pv[vv>0];th=float(np.sort(pos)[int(np.floor(.05*len(pos)))]) if len(pos) else None
                result={'ap':G.average_precision(yy,pd),'prevalence':float(yy.mean()),'n_dev':len(yy),'validation_loss':log,
                        'n_validation_positive':len(pos),'threshold_for_val_recall95':th,'thresholds':{}}
                for label,t in [('half',.5),('val_recall95',th)]:
                    if t is None:continue
                    pp=pd>=t;result['thresholds'][label]={'recall':G.estimate(pp,yy,ee),'false_alarm':G.estimate(pp,~yy,ee),
                                                           'precision':G.estimate(yy,pp,ee)}
                key=f'{rep}/{name}/{seed}';out[key]=result;save('danger_readers.json',out)
                torch.save({'probabilities':torch.from_numpy(pd).half(),'dev_indices':di,'reader_state':{k:v.cpu() for k,v in rd.state_dict().items()}},OUT/f'danger_{rep}_{name}_{seed}.pt')
                print(key,'AP',result['ap'],'recall95 threshold',result['thresholds'].get('val_recall95'),flush=True)
        del X,Xd;torch.cuda.empty_cache()
    assert before==fingerprints()
