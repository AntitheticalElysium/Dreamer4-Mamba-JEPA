"""Read existing body-head probabilities; do not refit any network."""
from .common import *


@torch.no_grad()
def main():
    before=fingerprints();cache={s:load_cache(s) for s in ('train','dev')};L={s:labels(s) for s in cache};ep={s:episode_rows(L[s]) for s in cache}
    Y={s:hud_values(L[s])[:,:4] for s in cache};alive={s:Y[s][:,0]>0 for s in cache};val=(ep['train']%10==9)&alive['train'];dev=alive['dev']
    out={}
    for rep in ('full_percept','x','b'):
        X={s:s1.frame_features(previous.half_encoding(cache[s]))[:,:-17] if rep=='full_percept' else torch.cat([r[rep] for r in cache[s]]) for s in cache}
        for seed in (0,1):
            ck=torch.load(OLD/f'belief_{rep}_body_body_{seed}.pt',weights_only=False,mmap=True);rd=s1.MLP(ck['input_dim'],ck['heads']).cuda().eval();rd.load_state_dict(ck['state'])
            probs={};parity=None
            for split,mask in [('train',val),('dev',dev)]:
                ids=torch.from_numpy(np.flatnonzero(mask));pp=[]
                for batch in ids.split(16384):pp.append(rd(X[split][batch].cuda())['hud'].reshape(-1,16,10)[:,:4].softmax(-1).cpu())
                p=torch.cat(pp);probs[split]=torch.stack([p[:,0,:3].sum(-1),*[p[:,j,0] for j in range(1,4)]],-1).numpy()
                if split=='dev':parity=float((p.argmax(-1)==ck['hud_prediction'][ids,:4]).float().mean())
            for j,h in enumerate(HUD[:4]):
                y=(Y['dev'][dev,j]<=2) if j==0 else (Y['dev'][dev,j]==0);vy=(Y['train'][val,j]<=2) if j==0 else (Y['train'][val,j]==0)
                pd,pv=probs['dev'][:,j],probs['train'][:,j];positive=pv[vy];t=float(np.sort(positive)[int(.05*len(positive))])
                result={'ap':G.average_precision(y,pd),'prevalence':float(y.mean()),'argmax_parity':parity,'threshold_val95':t,'thresholds':{}}
                for label,threshold in [('half',.5),('val_recall95',t)]:
                    pp=pd>=threshold;result['thresholds'][label]={'recall':G.estimate(pp,y,ep['dev'][dev]),'false_alarm':G.estimate(pp,~y,ep['dev'][dev]),'precision':G.estimate(y,pp,ep['dev'][dev])}
                out[f'{rep}/{h}/{seed}']=result
            save('multiclass_risk.json',out);print(rep,seed,{h:out[f'{rep}/{h}/{seed}']['ap'] for h in HUD[:4]},flush=True)
        del X;torch.cuda.empty_cache()
    assert before==fingerprints()


if __name__=='__main__':main()
