"""Paired descriptive measurements; no fitting and no policy selection."""
import ast
import importlib
import json
from functools import lru_cache
from pathlib import Path
import numpy as np
import torch
from saccade.eval import s1 as S, gates as G

HERE = Path(__file__).resolve().parent
C = importlib.import_module('artifacts.experiments.20261103_chooser_development.common')
POLICIES = ('uniform', 'existing', 'C_softmax', 'S_softmax', 'probability_mixture')
SEEDS = (611601, 611602)
L = dict(np.load(S.LABELS/'s0_dev_labels.npz'))
E = G.episode_grids(L)
OFF = np.r_[0, np.cumsum(L['ep_T']+1)]
AOFF = np.r_[0, np.cumsum(L['ep_T'])]
KINDS = G.KINDS
HUD_SRC = ast.literal_eval(next(n.value for n in ast.parse(
    (HERE.parent/'20261029_generation_diagnosis/loss_accounting.py').read_text()).body
    if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'HUD_SRC' for t in n.targets)))


def load(path):
    with np.load(path) as z:
        return dict(z)


def write(name, obj):
    C.write(HERE/name, obj)


@lru_cache(maxsize=16)
def bootstrap_weights(n):
    gen = np.random.default_rng(611920)
    ids = gen.integers(n, size=(2000, n))
    return np.bincount((np.arange(2000)[:,None]*n+ids).ravel(), minlength=2000*n).reshape(2000,n).astype(float)


def estimates(num, den, ep):
    """Paired episode-cluster ratio bootstrap. Zero denominators are not data."""
    num, den = np.asarray(num, float), np.asarray(den, float)
    if num.ndim == 1:
        num = num[:, None]
    if den.ndim == 1:
        den = den[:, None]
    den = np.broadcast_to(den, num.shape)
    _, inv = np.unique(ep, return_inverse=True)
    u, v = np.zeros((inv.max()+1, num.shape[1])), np.zeros((inv.max()+1, num.shape[1]))
    np.add.at(u, inv, num); np.add.at(v, inv, den)
    weights = bootstrap_weights(len(u))
    bnum, bden = weights@u, weights@v
    boot = np.divide(bnum, bden, out=np.full_like(bnum, np.nan), where=bden > 0)
    result = []
    for j in range(num.shape[1]):
        if den[:, j].sum() <= 0:
            result.append(None); continue
        good = np.isfinite(boot[:, j])
        low, high = np.quantile(boot[good, j], [.025, .975])
        result.append(dict(mean=float(num[:, j].sum()/den[:, j].sum()), low=float(low), high=float(high),
                           n=int((den[:, j] > 0).sum()), episodes=int((v[:, j] > 0).sum()),
                           denominator=float(den[:, j].sum()), valid_bootstraps=int(good.sum())))
    return result


def metadata(z):
    ep, t = z['ep'], z['t']; n = len(t); ix = OFF[ep]+t
    grid = np.stack([E[e]['grid'][tt] for e, tt in zip(ep, t)])
    oldgrid = np.stack([E[e]['grid'][tt-1] for e, tt in zip(ep, t)])
    view = L['s_view'][ix]; p = L['s_pos'][ix]; oldp = L['s_pos'][ix-1]
    masks = {}
    empty = np.ones((n, 81), bool)
    for k, kind in enumerate(KINDS, 1):
        a = np.zeros((n, 81), bool); a[:, :63] = (grid == k).reshape(n, -1)
        masks[kind] = a; empty &= ~a
    a = np.zeros((n, 81), bool); a[:, 31] = True
    masks['player'] = a; empty &= ~a
    terra = np.zeros((n, 81), bool); terra[:, :63] = True; terra &= empty
    for name, values in [('terrain_resources', [3,4,5,8,9,10,15,16]),
                         ('terrain_structures', [11,12]), ('terrain_common', [2,7,13]),
                         ('terrain_other', [0,1,6,14])]:
        a = np.zeros((n, 81), bool); a[:, :63] = np.isin(view, values).reshape(n, -1)
        masks[name] = a & terra
    for name, cells in [('health', [63]), ('food', [64]), ('drink', [65]), ('energy', [66]),
                        ('inventory', list(range(67,79))), ('blank_hud', [79,80])]:
        a = np.zeros((n, 81), bool); a[:, cells] = True; masks[name] = a
    assert np.all(sum(masks.values()) == 1), 'Semantic categories must partition all81 cells'
    mob = sum(masks[k] for k in KINDS) > 0
    rr, cc = np.divmod(np.arange(81), 9)
    near = mob & (np.maximum(abs(rr-3), abs(cc-4)) <= 2)[None]
    entering = np.zeros((n, 81), bool); edits = entering.copy(); entity_union = mob.copy()
    for i, (e, tt) in enumerate(zip(ep, t)):
        r, c = np.indices((7, 9)); d = p[i]-oldp[i]
        pr, pc = r+d[0], c+d[1]
        valid = (pr >= 0)&(pr < 7)&(pc >= 0)&(pc < 9)
        entering[i, :63] = (~valid).ravel()
        pr, pc = pr.clip(0,6), pc.clip(0,8)
        clear_before = (oldgrid[i,pr,pc] == 0)&~((pr == 3)&(pc == 4))
        diff = (view[i] != E[e]['view'][tt-1,pr,pc]) & valid & clear_before
        edits[i,:63] = diff.ravel() & terra[i,:63]
        entity_union[i,:63] |= (valid & (oldgrid[i,pr,pc] > 0)).ravel()
    changed_hud = np.zeros((n,81), bool)
    for cell, (key,col) in HUD_SRC.items():
        changed_hud[:,cell] = (L[key][ix,col].clip(0,9) != L[key][ix-1,col].clip(0,9))
    lighting = L['s_light'][ix]; action = L['t_action'][AOFF[ep]+t-1]
    num_mob = mob.sum(-1)
    previous_sighting = np.array([(E[e]['grid'][tt-8:tt] > 0).any() for e,tt in zip(ep,t)])
    strata = dict(all=np.ones(n,bool), random=z['cls']=='random', night=lighting<.5, day=lighting>=.5,
                  age_8_32=t<=32, age_33_128=(t>32)&(t<=128), age_over128=t>128,
                  no_entity=num_mob==0, one_entity=num_mob==1, multiple_entities=num_mob>=2,
                  entity_present=num_mob>0, near_entity=near.any(-1),
                  entity_after_8_clear_frames=(num_mob>0)&~previous_sighting,
                  action_noop=action==0, action_move=(action>=1)&(action<=4), action_do=action==5,
                  action_other=action>=6, sleeping=L['s_sleep'][ix-1],
                  health_loss=L['s_body'][ix,0]<L['s_body'][ix-1,0],
                  achievement_unlock=(L['s_ach'][ix]&~L['s_ach'][ix-1]).any(-1),
                  body_change=(L['s_body'][ix,:]!=L['s_body'][ix-1,:]).any(-1))
    for k in KINDS:
        strata[k+'_present'] = masks[k].any(-1)
    for cl in np.unique(z['cls']):
        strata['class_'+str(cl)] = z['cls']==cl
    increment = np.zeros(n, bool); counter_before = np.full(n,-1)
    for cell,(key,col) in HUD_SRC.items():
        sel = z['counter']==cell
        a,b = L[key][ix-1,col].clip(0,9),L[key][ix,col].clip(0,9)
        increment[sel] = b[sel]==a[sel]+1
        counter_before[sel] = a[sel]
    strata.update(counter_increment=increment, counter_increment_positive=increment&(counter_before>0),
                  counter_no_increment=(z['counter']>=0)&~increment)
    covers = {**masks, 'any_entity':mob, 'near_entity':near, 'terrain':terra,
              'entering_map':entering, 'terrain_edit':edits, 'changed_hud':changed_hud,
              'entity_outcome_union':entity_union}
    return dict(categories=masks, masks=covers, strata=strata, ep=ep, t=t, light=lighting,
                action=action, entity_count=num_mob, increment=increment, counter_before=counter_before)


def models(cohort):
    root = C.HERE
    names = [(f'E1_{s}', root/f'e1_{s}_u4000_{cohort}.npz') for s in SEEDS]
    names += [(f'E2_carry_{s}',root/f'e2_carry_{s}_ck10_s4000_{cohort}.npz') for s in SEEDS]
    names += [(f'E2_fresh_{s}',root/f'e2_fresh_{s}_ck10_endpoint_{cohort}.npz') for s in SEEDS]
    names += [(f'E3_{k}_{s}',root/f'e3_{k}_{s}_endpoint_{cohort}.npz')
              for k in ('coarse','rich') for s in SEEDS]
    out = {k:load(p) for k,p in names}
    if cohort == 'saved':
        z = load(QREF)
        out['native'] = {**z, 'C':z['contingency'], 'S':z['surprise']}
    else:
        z = load(C.OLD/'native_confirmation_and_noise.npz')
        out['native'] = {k:v[:978] for k,v in z.items()}
    ref = next(iter(out.values()))
    for name,z in out.items():
        for k in ('ep','t','cls','counter','faced'):
            assert np.array_equal(ref[k],z[k]), (name,k)
    return out


QREF = HERE.parent/'20261101_p1f15/readings/locus/p1f15.npz'


def gaze():
    report = {}
    for cohort in ('saved','confirmation'):
        mm = models(cohort); ref = mm['native']; meta = metadata(ref)
        natural = ref['cls']=='random'; part = {}
        strata = {k: v&natural for k,v in meta['strata'].items() if not k.startswith('class_')}
        strata['all_random'] = natural
        for name,z in mm.items():
            part[name] = {}
            pols = C.policies(z['C'],z['S'])
            unif = pols['uniform']
            for pol in POLICIES:
                p = pols[pol]
                native_p = C.policies(ref['C'],ref['S'])[pol]
                assert np.allclose(p.sum(-1),1)
                # Expected crop-cell shares are disjoint; cover-any probabilities are not.
                shares = {k:(p*(v[:,C.WIN].sum(-1)/9)).sum(-1) for k,v in meta['categories'].items()}
                assert np.allclose(sum(shares.values()),1)
                centres = C.CENTRES[:,0]*9+C.CENTRES[:,1]
                center_shares = {k:(p*v[:,centres]).sum(-1) for k,v in meta['categories'].items()}
                assert np.allclose(sum(center_shares.values()),1)
                cov = {k:(p*v[:,C.WIN].any(-1)).sum(-1) for k,v in meta['masks'].items()}
                ucov = {k:(unif*v[:,C.WIN].any(-1)).sum(-1) for k,v in meta['masks'].items()}
                ncov = {k:(native_p*v[:,C.WIN].any(-1)).sum(-1) for k,v in meta['masks'].items()}
                metrics = {**{'cell_share_'+k:v for k,v in shares.items()},
                           **{'center_share_'+k:v for k,v in center_shares.items()},
                           'touch_hud':(p*(C.WIN>=63).any(-1)).sum(-1),
                           'center_in_hud':p[:,C.CENTRES[:,0]>=7].sum(-1),
                           'entropy':-(p*np.log(p)).sum(-1)}
                keys=list(metrics); vals=np.stack(list(metrics.values()),-1)
                result = {}
                for group,ix in strata.items():
                    if not ix.any():continue
                    stats=estimates(vals[ix],np.ones(ix.sum()),ref['ep'][ix])
                    result[group]={k:v for k,v in zip(keys,stats)}
                for k in meta['masks']:
                    ix=natural&meta['masks'][k].any(-1)
                    if not ix.any():continue
                    values=np.stack([cov[k],ucov[k],cov[k]-ucov[k],cov[k]-ncov[k]],-1)
                    stats=estimates(values[ix],np.ones(ix.sum()),ref['ep'][ix])
                    result['coverage_'+k]=dict(zip(('policy','uniform','minus_uniform','minus_native'),stats))
                    for day,light in [('day',meta['light']>=.5),('night',meta['light']<.5)]:
                        j=ix&light
                        if j.any():
                            stats=estimates(values[j],np.ones(j.sum()),ref['ep'][j])
                            result['coverage_'+k+'_'+day]=dict(zip(('policy','uniform','minus_uniform','minus_native'),stats))
                part[name][pol]=result
        report[cohort]=part
        # Raw expected-cell heatmaps are directly reproducible, no gaze draws.
        heat = {}
        for name,z in mm.items():
            p=C.policies(z['C'],z['S'])['existing'][natural]
            cell=np.zeros((len(p),81))
            for j,w in enumerate(C.WIN):cell[:,w]+=p[:,j,None]
            assert np.allclose(cell.sum(-1),9)
            heat[name]=cell.mean(0).reshape(9,9)
        np.savez_compressed(HERE/f'heatmaps_{cohort}.npz',**heat)
    write('gaze.json',report)


def belief():
    report={}; audits={}
    for cohort in ('saved','confirmation'):
        report[cohort]={}
        for seed in SEEDS:
            co=load(HERE/f'e3_coarse_{seed}_{cohort}_cells.npz')
            ri=load(HERE/f'e3_rich_{seed}_{cohort}_cells.npz')
            for k in ('ep','t','cls'):assert np.array_equal(co[k],ri[k])
            m=metadata(co); ep=co['ep']
            ce,re=co['cell_S']/16,ri['cell_S']/16
            diff=ce-re
            assert np.allclose(sum((diff*v).sum(-1) for v in m['categories'].values()),diff.sum(-1),atol=2e-6)
            pco=C.policies(co['cell_C'][:,C.WIN].sum(-1),co['cell_S'][:,C.WIN].sum(-1))
            pri=C.policies(ri['cell_C'][:,C.WIN].sum(-1),ri['cell_S'][:,C.WIN].sum(-1))
            out={}
            for group,ix in m['strata'].items():
                if not ix.any():continue
                x=diff[ix].mean(-1); old=ce[ix].mean(-1); new=re[ix].mean(-1)
                values=np.stack([old,new,x,100*x],-1)
                den=np.stack([np.ones(ix.sum())]*3+[old],-1)
                r=dict(zip(('coarse_mse','rich_mse','absolute_reduction','percent_reduction'),
                           estimates(values,den,ep[ix])))
                r['unique_frames']=int(len(np.unique(np.stack([ep[ix],co['t'][ix]],-1),axis=0)))
                r['targets']={}
                for cat,mask in m['masks'].items():
                    count=mask[ix].sum(-1); d=(diff[ix]*mask[ix]).sum(-1)
                    old=(ce[ix]*mask[ix]).sum(-1); new=(re[ix]*mask[ix]).sum(-1)
                    if count.sum()==0:continue
                    nums=np.stack([old,new,d,100*d,d/81],-1)
                    dens=np.stack([count,count,count,old,np.ones(len(count))],-1)
                    stats=estimates(nums,dens,ep[ix])
                    r['targets'][cat]=dict(zip(('coarse_per_cell','rich_per_cell','per_cell_reduction',
                                                'percent_reduction','contribution_to_global_reduction'),stats))
                for pol in ('existing','C_softmax','S_softmax','probability_mixture'):
                    r[pol]={}
                    for cat in ('any_entity','near_entity','terrain','changed_hud','entering_map','terrain_edit'):
                        touch=m['masks'][cat][:,C.WIN].any(-1)
                        exposed=ix&m['masks'][cat].any(-1)
                        if exposed.any():
                            delta=((pri[pol]-pco[pol])*touch).sum(-1)
                            r[pol][cat]=estimates(delta[exposed],np.ones(exposed.sum()),ep[exposed])[0]
                out[group]=r
            report[cohort][str(seed)]=out
            audits[f'{cohort}_{seed}']={
                'global_reduction':float(diff.mean()),
                'sum_disjoint_category_contributions':float(sum((diff*v).mean() for v in m['categories'].values())),
                'paired_identical_errors_max':float(np.abs(ce-ce).max())}
    write('belief.json',report);write('decomposition_audit.json',audits)


def counter_breakdown():
    out={}
    for cohort in ('saved','confirmation'):
        out[cohort]={}
        for seed in SEEDS:
            co=load(HERE/f'e3_coarse_{seed}_{cohort}_cells.npz')
            ri=load(HERE/f'e3_rich_{seed}_{cohort}_cells.npz')
            m=metadata(co);ar=np.arange(len(co['ep']));cell=np.maximum(co['counter'],0)
            old=co['cell_S'][ar,cell]/16; new=ri['cell_S'][ar,cell]/16
            part={}
            for cl in np.unique(co['cls']):
                cls=(co['cls']==cl)&(co['counter']>=0)
                if not cls.any():continue
                statuses={'all':np.ones(len(cls),bool), 'increment':m['increment'],
                          'first_increment':m['increment']&(m['counter_before']==0),
                          'positive_increment':m['increment']&(m['counter_before']>0),
                          'no_increment':~m['increment']}
                part[str(cl)]={}
                for status,sel in statuses.items():
                    ix=cls&sel
                    if not ix.any():continue
                    vals=np.stack([old[ix],new[ix],old[ix]-new[ix],100*(old[ix]-new[ix])],-1)
                    den=np.stack([np.ones(ix.sum())]*3+[old[ix]],-1)
                    stats=estimates(vals,den,co['ep'][ix])
                    part[str(cl)][status]=dict(zip(('coarse_mse','rich_mse','absolute_reduction','percent_reduction'),stats))
            out[cohort][str(seed)]=part
    write('belief_counters.json',out)


def noise():
    rows=load(HERE/'e1_611601_static.npz')['rows']
    native=load(HERE.parent/'20261102_p1f15_readiness/gaze_checks/static_10.npz')
    assert np.array_equal(rows,native['rows'])
    bank={p.stem[:-7]:load(p) for p in HERE.glob('*_static.npz')}
    bank['native']={'C':native['scores'][:,:,0], 'S':native['scores'][:,:,1]}
    report={}
    current = load(HERE/'static_percepts.npz')['p'].astype(float)
    variance = current.var(1).sum(-1)  # finite-bank exact variance, denominator4
    variance_windows = variance[:,C.WIN].sum(-1)
    for name,z in bank.items():
        c,s=z['C'],z['S']
        assert np.array_equal(c[:,0],c[:,-1]),name
        pols=C.policies(c.reshape(-1,49),s.reshape(-1,49))
        report[name]={}
        # Exact decomposition across the four fixed-state render draws:
        # mean squared residual = squared residual of mean percept + empirical variance.
        # Uniform-window averaging is identical for cheap and archived native scores.
        v = variance_windows.mean(-1); total = s.mean((1,2))
        assert np.all(total-v > -1e-4), (name,total,v)
        report[name]['finite_repeat_noise_component'] = dict(
            per_scene_fraction=(v/total).tolist(),
            fraction_of_pooled_surprise=float(v.sum()/total.sum()),
            empirical_noise_score=v.tolist(), total_surprise_score=total.tolist(),
            convention='four-draw empirical variance; not the total irreducible-noise fraction')
        for pol in POLICIES:
            p=pols[pol].reshape(8,4,49)
            tv=[];switch=[];hud=[]
            for a in range(4):
                for b in range(a+1,4):
                    tv.append(.5*np.abs(p[:,a]-p[:,b]).sum(-1))
                    switch.append(p[:,a].argmax(-1)!=p[:,b].argmax(-1))
                    hu=(C.WIN>=63).any(-1)
                    hud.append(np.abs(p[:,a,hu].sum(-1)-p[:,b,hu].sum(-1)))
            # One value per physical scene: six repeat pairs are not independent scenes.
            vals=np.stack([np.mean(tv,0),np.mean(switch,0),np.mean(hud,0)],-1)
            report[name][pol]=dict(zip(('TV','argmax_switch','abs_hud_coverage_change'),
                                      estimates(vals,np.ones(8),rows[:,0])))
            report[name][pol]['per_scene']=vals.tolist()
    write('noise.json',dict(rows=rows.tolist(),selected_night_scenes=8,models=report))


def main():
    torch.set_num_threads(2)
    gaze();print('gaze complete',flush=True)
    belief();print('belief complete',flush=True)
    counter_breakdown();print('counter breakdown complete',flush=True)
    noise();print('noise complete',flush=True)


if __name__=='__main__':main()
