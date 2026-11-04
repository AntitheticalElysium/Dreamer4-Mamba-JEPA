"""Post-result paired/grounded dissection; archived forecasts only, no model fitting."""
import ast
import importlib
import json
from pathlib import Path
import numpy as np
import torch

C=importlib.import_module('artifacts.experiments.20261103_chooser_development.common')
CLASSES=('drink','stone','coal','iron')


def estimate(v,ep):
    v=np.asarray(v,float); ep=np.asarray(ep)
    _,inv=np.unique(ep,return_inverse=True); count=np.bincount(inv)
    sums=np.bincount(inv,weights=v)
    rng=np.random.default_rng(611803);bs=rng.integers(len(count),size=(2000,len(count)))
    b=sums[bs].sum(1)/count[bs].sum(1)
    lo,hi=np.quantile(b,[.025,.975])
    return dict(mean=float(v.mean()),low=float(lo),high=float(hi),n=len(v),episodes=len(count))


def archived(stem,cohort):
    with np.load(C.HERE/f'{stem}_{cohort}.npz') as f:
        return dict(f)


def pair(a,b):
    for k in ('ep','t','cls','counter','faced','light'):
        assert np.array_equal(a[k],b[k]),k
    pa,pb=C.policies(a['C'],a['S']),C.policies(b['C'],b['S'])
    remote=(C.WIN[None]==a['counter'][:,None,None]).any(-1)
    contact=(C.WIN[None]==a['faced'][:,None,None]).any(-1)
    out={}
    for cl in CLASSES:
        ix=a['cls']==cl; ep=a['ep'][ix];r={}
        for pol in ('existing','C_softmax','S_softmax','probability_mixture','direct_mixture'):
            r[pol]={}
            for k,mask in (('remote',remote),('contact',contact),('union',remote|contact)):
                v=((pa[pol]-pb[pol])*mask).sum(-1)
                r[pol][k]=estimate(v[ix],ep)
        for key in ('counter_mse','counter_C','factual_mse'):
            r[key]=estimate(a[key][ix]-b[key][ix],ep)
        out[cl]=r
    return out


def fields():
    source=C.HERE.parent/'20261029_generation_diagnosis/loss_accounting.py'
    node=next(n for n in ast.parse(source.read_text()).body if isinstance(n,ast.Assign)
              and any(isinstance(t,ast.Name) and t.id=='HUD_SRC' for t in n.targets))
    return ast.literal_eval(node.value)


def ground(z,labels):
    offset=np.r_[0,np.cumsum(labels['ep_T']+1)];g=offset[z['ep']]+z['t']
    before=np.full(len(g),-1);after=before.copy()
    for cell,(key,col) in fields().items():
        ix=z['counter']==cell
        before[ix]=labels[key][g[ix]-1,col].clip(0,9)
        after[ix]=labels[key][g[ix],col].clip(0,9)
    return dict(before=before,after=after,increment=after==before+1)


def grounded_reading(z,cache,labels):
    q=ground(z,labels);ids=cache['offsets'][z['ep']]+z['t']
    target=cache['p'][ids].float().numpy();previous=cache['p'][ids-1].float().numpy()
    taken,null=z['prediction'].astype(float),z['noop'].astype(float)
    cell=np.maximum(z['counter'],0); ar=np.arange(len(cell))
    d=(target-previous)[ar,cell]
    u=(taken-previous)[ar,cell];v=(null-previous)[ar,cell]
    ccell=((taken-null)**2).sum(-1)
    cover=np.zeros((49,81));cover[np.arange(49)[:,None],C.WIN]=1
    remote=(C.WIN[None]==z['counter'][:,None,None]).any(-1)
    multiplicity=remote.astype(float)@cover
    own=ccell[ar,cell]*remote.sum(-1)
    total=(ccell*multiplicity).sum(-1)
    pp=C.policies(z['C'],z['S'])
    result={}
    def ratio(a,b):return float(a/b) if b>1e-15 else None
    for cl in (*CLASSES,'wood','sapling','stone_no_tool'):
        base=z['cls']==cl
        if not base.any():continue
        strata=dict(all=base,increment=base&q['increment'],
                    increment_from0=base&q['increment']&(q['before']==0),
                    increment_from_positive=base&q['increment']&(q['before']>0),
                    no_increment=base&~q['increment'],
                    day_increment=base&q['increment']&(z['light']>=.5),
                    night_increment=base&q['increment']&(z['light']<.5))
        result[cl]={}
        for label,ix in strata.items():
            if not ix.any():continue
            energy=(d[ix]**2).sum()
            r=dict(n=int(ix.sum()),episodes=len(np.unique(z['ep'][ix])),
                   actual_increment_rate=float(q['increment'][ix].mean()),
                   mean_before=float(q['before'][ix].mean()),
                   counter_mse_vs_copy=ratio(z['counter_mse'][ix].sum(),z['counter_copy_mse'][ix].sum()),
                   counter_C_over_recorded_step=ratio(z['counter_C'][ix].sum(),z['counter_target_step'][ix].sum()),
                   forecast_delta_alignment=ratio((u[ix]*d[ix]).sum(),energy),
                   noop_delta_alignment=ratio((v[ix]*d[ix]).sum(),energy),
                   target_cell_share_of_C_in_remote_windows=ratio(own[ix].sum(),total[ix].sum()),
                   fp16_C_vs_stored_float32=ratio(ccell[ar,cell][ix].sum(),z['counter_C'][ix].sum()))
            r['remote_mass']={p:float((prob[ix]*remote[ix]).sum(-1).mean()) for p,prob in pp.items()}
            r['top_C_cell_is_target_counter']=float((ccell[ix].argmax(-1)==z['counter'][ix]).mean())
            r['top_C_cells']={str(k):int(n) for k,n in zip(*np.unique(ccell[ix].argmax(-1),return_counts=True))}
            result[cl][label]=r
    return result


def audit_metadata():
    out={}
    for split,n in (('train',400),('dev',600)):
        base=torch.load(C.OLD/f'{split}_{n}_coarse.pt',weights_only=False,map_location='cpu')
        for ck in (1,5):
            z=torch.load(C.HERE/f'cache_ck{ck:02d}/{split}_{n}_coarse.pt',weights_only=False,map_location='cpu')
            for k in ('reward','terminated','prev_action','episode','t','eps','monitor'):
                assert torch.equal(base[k],z[k]),(split,ck,k)
            for k in ('offsets','episode_keys'):assert np.array_equal(base[k],z[k]),(split,ck,k)
            out[f'{split}_ck{ck}']=dict(rows=len(z['p']),identical_nontoken_fields=True,
                                      checkpoint_sha256=z['checkpoint_sha256'])
    for seed in (611601,611602):
        for ck,step in ((1,4000),(5,8000),(10,12000)):
            z=torch.load(C.HERE/f'e2_carry_{seed}_ck{ck:02d}.pt',weights_only=False,map_location='cpu')
            steps={int(v['step']) for v in z['optimizer']['state'].values()}
            assert steps=={step},(seed,ck,steps)
            out[f'optimizer_{seed}_ck{ck}']=dict(steps=list(steps))
    return out


def main():
    torch.set_num_threads(2)
    C.write(C.HERE/'metadata_optimizer_audit.json',audit_metadata())
    cache=torch.load(C.OLD/'dev_600_coarse.pt',weights_only=False,map_location='cpu')
    with np.load(C.ROOT/'artifacts/eda/s0/s0_dev_labels.npz') as f:
        labels={k:f[k] for k in ('ep_T','s_body','s_inv')}
    pairs={};grounded={}
    for cohort in ('saved','confirmation'):
        pairs[cohort]={};grounded[cohort]={}
        for seed in (611601,611602):
            for label,aa,bb in [
                ('e1_final_minus_init',f'e1_{seed}_u4000',f'e1_{seed}_u0000'),
                ('e2_ck5_carry_minus_fresh',f'e2_carry_{seed}_ck05_s4000',f'e2_fresh_{seed}_ck05_endpoint'),
                ('e2_ck10_carry_minus_fresh',f'e2_carry_{seed}_ck10_s4000',f'e2_fresh_{seed}_ck10_endpoint'),
                ('e3_rich_minus_coarse',f'e3_rich_{seed}_endpoint',f'e3_coarse_{seed}_endpoint')]:
                pairs[cohort][f'{label}_{seed}']=pair(archived(aa,cohort),archived(bb,cohort))
            stems=[f'e1_{seed}_u4000',f'e2_carry_{seed}_ck10_s4000',
                   f'e2_fresh_{seed}_ck10_endpoint',f'e3_coarse_{seed}_endpoint',f'e3_rich_{seed}_endpoint']
            for stem in stems:
                grounded[cohort][stem]=grounded_reading(archived(stem,cohort),cache,labels)
    C.write(C.HERE/'paired_dissection.json',pairs)
    C.write(C.HERE/'grounded_dissection.json',grounded)
    print('archived dissection complete',flush=True)


if __name__=='__main__':main()
