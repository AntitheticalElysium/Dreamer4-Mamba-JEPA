"""Add generic full-frame sparse-change and exact-identity examples to the observation-derived A9b training data.
No simulator labels. Full-frame templates drawn only from the train split. Old map data is retained for night noise.
This is a candidate teacher; audit on disjoint dev episodes before adopting it."""
import os
os.environ.setdefault('JAX_PLATFORMS','cpu')
import json,sys
import numpy as np
import lightgbm as lgb
from saccade import data as D,train as TR
from .audit import HERE,A,full_cells

def main(clean_only=False):
    rng=np.random.default_rng(110818);eps=D.load_split(TR.load_cfg())['train']
    picks=rng.choice(len(eps),64,replace=False);templates=[]
    for i in picks:
        e=eps[i];f=D.RP.episode_fields(e.shard,e.slot)['observations'].numpy()
        take=rng.choice(len(f),min(8,len(f)),replace=False)
        templates.append(full_cells(f[take]).reshape(-1,7,7,3))
    bank=np.concatenate(templates);n=120000
    a=bank[rng.integers(len(bank),size=n)].copy();b=a.copy()
    # Every screen cell uses the same sampling rule. Sparse edits prevent small changes being absent from positives.
    mode=rng.integers(0,4,size=n)
    different=mode==1;b[different]=bank[rng.integers(len(bank),size=different.sum())]
    sparse=np.flatnonzero(mode>=2)
    for count in (1,2,4,8,16):
        ids=sparse[np.arange(len(sparse))%5==int(np.log2(count))]
        pp=rng.integers(0,49,size=(len(ids),count));source=bank[rng.integers(len(bank),size=len(ids))].reshape(-1,49,3)
        bb=b[ids].reshape(-1,49,3);bb[np.arange(len(ids))[:,None],pp]=source[np.arange(len(ids))[:,None],pp]
        b[ids]=bb.reshape(-1,7,7,3)
    y=np.any(a!=b,axis=(1,2,3)).astype(np.int8)
    # Paired view corruption is generic and full-cell, not restricted to any screen region.
    def corrupt(x):
        original=x.copy();x=x.astype(np.float32);z=rng.random((n,1,1,1));strength=rng.random((n,1,1,1))*.8
        active=rng.random((n,1,1,1))<.5;strength*=active
        lum=(x*np.array([.299,.587,.114],np.float32)).sum(-1,keepdims=True)
        x=(z*lum+(1-z)*x)*rng.uniform(.35,1,(n,1,1,1))
        noise=rng.uniform(32,127,(n,7,7,1));x=(1-strength)*x+strength*noise
        # Half of examples remain entirely clean.
        return np.where(active,np.clip(x,0,255),original).astype(np.float32)
    aa,bb=(a,b) if clean_only else (corrupt(a),corrupt(b))
    Xnew=A.feats(aa,bb)
    old=np.load(D.A9B/'a9b_cells.npz');gr=old['real_g'];perm=np.random.default_rng(0).permutation(np.unique(gr))
    tr=perm[:int(.7*len(perm))];tau=A.otsu(old['real_lf'][(~old['real_night'])&np.isin(gr,tr)])
    keep=np.isin(old['corrupt_g'],tr)
    xo=old['corrupt_X'][keep];yo=(old['corrupt_lf'][keep]>tau).astype(np.int8)
    X=np.concatenate([xo,Xnew]);Y=np.r_[yo,y]
    order=rng.permutation(len(Y));v=order[:len(order)//10];t=order[len(order)//10:]
    model=lgb.train({'objective':'binary','learning_rate':.1,'num_leaves':63,'min_data_in_leaf':50,'verbose':-1,'num_threads':4,'seed':110818},lgb.Dataset(X[t],Y[t]),400,valid_sets=[lgb.Dataset(X[v],Y[v])],callbacks=[lgb.early_stopping(30,verbose=False)])
    tag='candidate_clean_teacher' if clean_only else 'candidate_teacher'
    model.save_model(str(HERE/f'{tag}.txt'))
    (HERE/f'{tag}_REFIT.json').write_text(json.dumps({'template_episodes':[eps[int(i)].key for i in picks],'template_cells':len(bank),'synthetic_pairs':n,'old_observation_derived_pairs':len(yo),'trees':model.best_iteration,'seed':110818,'clean_only':clean_only},indent=2))
    print('fit',len(Y),model.best_iteration,flush=True)

if __name__=='__main__':main('--clean' in sys.argv)
