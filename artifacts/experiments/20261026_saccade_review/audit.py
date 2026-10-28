"""Targeted independent checks; the reviewed package is imported without edits."""
import json
from pathlib import Path
import sys
import numpy as np
from saccade.eval import gates as G

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]
SOURCE = ROOT / 'artifacts/eda/s0'


def rejecting(a, b):
    result = []
    for k, x in a.items():
        y = b.get(k)
        if x is None or y is None:
            continue
        se = np.hypot(x['se'], y['se'])
        if se > 0 and abs(y['est'] - x['est']) > 3 * se:
            result.append(k)
    return result


def structural():
    L = dict(np.load(SOURCE / 's0_dev_labels.npz'))
    P = G.build_pairs(L)
    A, B = G.subset(P, P['ep'] % 2 == 0), G.subset(P, P['ep'] % 2 == 1)
    Q = {k: v.copy() for k, v in B.items()}
    Q['a'] = np.array([{1: 2, 2: 1, 3: 4, 4: 3}.get(int(x), int(x)) for x in Q['a']])
    ra, rb, rq = map(G.all_estimates, (A, B, Q))
    key = 'dir_ok_given_moved_move'
    result = {'opposite_move_actions': {'reference': ra[key], 'corrupt': rq[key],
               'real_B_rejections': rejecting(ra, rb), 'corrupt_rejections': rejecting(ra, rq)}}

    # Reproduce c2st's split arithmetic on paired real/generated data, without fitting a classifier.
    paired = G.subset(P, P['ep'] < 128)
    rng = np.random.default_rng(20261026)
    n = min(len(paired['a']), 60000)
    ia, ib = rng.choice(len(paired['a']), n, replace=False), rng.choice(len(paired['a']), n, replace=False)
    ids = np.r_[paired['ep'][np.sort(ia)], paired['ep'][np.sort(ib)] + 10**6]
    unique = np.unique(ids); te = np.isin(ids, rng.choice(unique, len(unique)//3, replace=False))
    train_physical, test_physical = set(ids[~te] % 10**6), set(ids[te] % 10**6)
    result['paired_c2st_split'] = {'physical_episodes': len(set(ids % 10**6)),
                                  'physical_episodes_in_both_train_and_test': len(train_physical & test_physical)}

    # A joint score on summaries can be identical for physically different trajectories.
    H = 4
    good = np.zeros((H+1, 7, 9), np.int8)
    good[:, 2, 3] = 1; good[:, 4, 5] = 1
    bad = good.copy()
    bad[1::2] = 0; bad[1::2, 3, 2] = 1; bad[1::2, 3, 6] = 1
    pos = np.full((H+1, 2), 10); body = np.full((H+1, 4), 9)
    rewards, done = np.zeros(H), np.zeros(H, bool)
    vg = G.summary_vector(good, body, rewards, done, pos)
    vb = G.summary_vector(bad, body, rewards, done, pos)
    def pairs(grid):
        return {'g0': grid[:-1], 'g1': grid[1:], 'd': np.zeros((H,2), int), 'h0': body[:-1,0],
                'h1': body[1:,0], 'a': np.zeros(H, int), 'night': np.zeros(H,bool),
                'since': np.full(H,10), 'ep': np.zeros(H,int)}
    eg, eb = G.events(pairs(good)), G.events(pairs(bad))
    result['impossible_diagonal_moves'] = {'different_grid_cells': int((good != bad).sum()),
        'identical_summary': bool(np.array_equal(vg,vb)),
        'energy_score': G.energy_score(np.stack([vb,vb]),vg),
        'same_persistence': all(np.array_equal(x,y) for x,y in zip(G.persistence(good,pos),G.persistence(bad,pos))),
        'all_one_step_event_arrays_equal': all(np.array_equal(eg[k],eb[k]) for k in eg)}

    # Last sighting of this terrain is occluded; the stored value is from an older clear frame.
    positions = np.array([[10,10],[10,10],[10,11],[10,10]])
    grid = np.zeros((4,7,9),np.int8); grid[1,3,0]=1
    view = np.zeros((4,7,9),np.int8); view[0,3,0]=2; view[1,3,0]=3; view[3,3,0]=3
    rows = G.revisit_events([{'pos':positions,'grid':grid,'view':view}])
    result['occluded_last_sighting'] = {'disagreement_rows': rows[rows[:,2]==0].tolist(),
         'last_clear_terrain_age':3, 'last_in_view_age':2,
         'documented_both_sightings_unoccluded':False}
    (OUT/'structural_checks.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))


def calibration():
    import lightgbm as lgb
    from saccade.eval import calib as C
    tr = C.transitions(dict(np.load(SOURCE/'s0_train400_labels.npz')))
    dv = C.transitions(dict(np.load(SOURCE/'s0_dev_labels.npz')))
    held = tr['ep'] % 10 == 9
    out = {}
    for leaves in (31,127):
        params = {'objective':'binary','verbose':-1,'num_threads':12,'seed':0,'num_leaves':leaves,
                  'learning_rate':0.05,'lambda_l2':1.0,'min_sum_hessian_in_leaf':1.0}
        model = lgb.train(params,lgb.Dataset(tr['X2'][~held],tr['unlock'][~held]),2000,
             valid_sets=[lgb.Dataset(tr['X2'][held],tr['unlock'][held])],
             callbacks=[lgb.early_stopping(50,verbose=False)])
        p = model.predict(dv['X2'],num_iteration=model.best_iteration)
        y = dv['unlock'].astype(float)
        out[str(leaves)] = C.score(np.where(y==1,p,1-p),p,y,dv['ep']) | {'best_iteration':model.best_iteration}
        np.savez_compressed(OUT/f'capacity_{leaves}_predictions.npz',p=p,y=y,ep=dv['ep'])
        print(leaves,json.dumps(out[str(leaves)]),flush=True)
        (OUT/'memory_free_capacity.json').write_text(json.dumps(out,indent=2))


def calibration_context():
    import lightgbm as lgb
    from saccade.eval import calib as C
    def data(path):
        L = dict(np.load(path)); d = C.transitions(L)
        T = L['ep_T']; off = np.r_[0,np.cumsum(T+1)]
        i0 = np.concatenate([np.arange(off[e],off[e]+T[e]) for e in range(len(T))])
        grid = np.concatenate([e['grid'] for e in G.episode_grids(L)])
        feats = []
        for idx in (i0,i0+1):
            vis = grid[idx] == 0; vis[:,3,4] = False
            feats += [np.stack([(grid[idx]==k).sum((1,2)) for k in range(1,5)],1),
                      np.stack([((L['s_view'][idx]==k)&vis).sum((1,2)) for k in range(17)],1)]
        d['X'] = np.concatenate([d['X2'],*feats],1)
        return d
    tr,dv = data(SOURCE/'s0_train400_labels.npz'),data(SOURCE/'s0_dev_labels.npz')
    held = tr['ep']%10 == 9
    params = {'objective':'binary','verbose':-1,'num_threads':12,'seed':0,'num_leaves':31,
              'learning_rate':0.05,'lambda_l2':1.0,'min_sum_hessian_in_leaf':1.0}
    model = lgb.train(params,lgb.Dataset(tr['X'][~held],tr['unlock'][~held]),2000,
         valid_sets=[lgb.Dataset(tr['X'][held],tr['unlock'][held])],
         callbacks=[lgb.early_stopping(50,verbose=False)])
    p=model.predict(dv['X'],num_iteration=model.best_iteration); y=dv['unlock'].astype(float)
    result=C.score(np.where(y==1,p,1-p),p,y,dv['ep'])|{'best_iteration':model.best_iteration,'older_history':False}
    np.savez_compressed(OUT/'visible_context_predictions.npz',p=p,y=y,ep=dv['ep'])
    baseline = np.load(OUT/'capacity_31_predictions.npz')
    assert np.array_equal(baseline['ep'],dv['ep']) and np.array_equal(baseline['y'],y)
    loss = lambda q: -np.log(np.maximum(np.where(y==1,q,1-q),1e-12))
    delta = loss(p)-loss(baseline['p'])
    sums = np.bincount(dv['ep'],weights=delta); counts=np.bincount(dv['ep'])
    rng=np.random.default_rng(20261026)
    boots=[]
    for _ in range(2000):
        idx=rng.integers(0,len(counts),len(counts)); boots.append(sums[idx].sum()/counts[idx].sum())
    result['paired_nll_difference_to_R2']={'est':float(delta.mean()),
        'ci95':np.quantile(boots,[0.025,0.975]).tolist(),'bootstrap_episodes':len(counts),'bootstrap_samples':2000}
    (OUT/'memory_free_visible_context.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2),flush=True)


def revisit_age():
    L=dict(np.load(SOURCE/'s0_dev_labels.npz'))
    rr,cc=np.meshgrid(np.arange(7)-3,np.arange(9)-4,indexing='ij')
    offsets=list(zip(rr.ravel().tolist(),cc.ravel().tolist()))
    total=affected=changed_stratum=0
    ages=[]
    stratum=lambda x: sum(x>b for b in (16,64,256))
    for ep in G.episode_grids(L):
        seen={}; clear={}; prev=set()
        for t,(pos,grid,view) in enumerate(zip(ep['pos'],ep['grid'],ep['view'])):
            cur=set()
            for (dr,dc),owner,value in zip(offsets,grid.ravel(),view.ravel()):
                w=(int(pos[0])+dr,int(pos[1])+dc); cur.add(w)
                if t>0 and w not in prev and w in seen and owner==0 and w in clear:
                    total+=1
                    if clear[w][0]!=seen[w]:
                        affected+=1; ages.append(seen[w]-clear[w][0])
                        changed_stratum+=stratum(t-seen[w])!=stratum(t-clear[w][0])
                seen[w]=t
                if owner==0: clear[w]=(t,int(value))
            prev=cur
    result={'revisits':total,'last_sighting_occluded':affected,
            'fraction_affected':affected/total,'different_gap_stratum':int(changed_stratum),
            'extra_value_age_quantiles':np.quantile(ages,[0,.5,.9,1]).tolist()}
    (OUT/'revisit_age.json').write_text(json.dumps(result,indent=2)); print(json.dumps(result),flush=True)


if __name__ == '__main__':
    {'structural':structural,'calibration':calibration,'calibration_context':calibration_context,
     'revisit_age':revisit_age}[sys.argv[1]]()
