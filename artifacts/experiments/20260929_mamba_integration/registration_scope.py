"""Exploratory measured registration burden for per-screen-slot temporal Mamba.

Uses only previously inspected diagnostic roots. Derives a high-confidence scroll label from
visible terrain, independent of the patch-token distance being measured. Ambiguous move attempts
are reported rather than assigned an invented label.
"""
import hashlib
import json
import sys
from pathlib import Path
import torch

ROOT=Path('/home/antithetical/EPITA/PERSO/DynamicHorizons-Mamba-JEPA')
HERE=Path(__file__).resolve().parent
sys.path[:0]=[str(ROOT),str(ROOT/'artifacts/experiments/20260927_levers'),
              str(ROOT/'artifacts/experiments/20260926_diagnosis')]
import teval as E  # noqa: E402
from onestep import classify  # noqa: E402
from scroll import SHIFTS, MOVE_SHIFT  # noqa: E402


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def shifted(a,b,shift):
    dr,dc=SHIFTS[shift]
    rs,cs=slice(max(0,-dr),7-max(0,dr)),slice(max(0,-dc),9-max(0,dc))
    rt,ct=slice(max(0,dr),7+min(0,dr)),slice(max(0,dc),9+min(0,dc))
    return a[rt,ct],b[rs,cs],a[rs,cs]


def terrain(v):return v[:1071].reshape(7,9,17).argmax(-1)
def zombie(v):return v[1071:1512].reshape(7,9,7)[...,0]>0

def terrain_decision(v0,v1,action):
    t0,t1=terrain(v0),terrain(v1)
    shift=MOVE_SHIFT[int(action)]
    x,y,same=shifted(t0,t1,shift)
    aligned=float((x==y).float().mean())
    fixed=float((same==y).float().mean())
    # Indecisive if the terrain is homogeneous or the action modifies the map.
    if aligned>=.98 and aligned-fixed>=.10:return shift,aligned,fixed
    if fixed>=.98 and fixed-aligned>=.10:return 0,aligned,fixed
    return -1,aligned,fixed


def compare_tokens(t0,t1,shift,z0,z1):
    a,b,fixed=shifted(t0[:63].float().reshape(7,9,192),
                      t1[:63].float().reshape(7,9,192),shift)
    za,zb,zf=shifted(z0,z1,shift)
    fixed_err=(fixed-b).square().mean(-1)
    aligned_err=(a-b).square().mean(-1)
    zchange=za!=zb
    return float(fixed_err.mean()),float(aligned_err.mean()),float((aligned_err<fixed_err).float().mean()),\
           int(zchange.sum()),int(zchange.numel()),\
           float(aligned_err[zchange].mean()) if zchange.any() else None,\
           float(aligned_err[~zchange].mean()) if (~zchange).any() else None


def main():
    torch.set_num_threads(8)
    meta,train,_=E.split();cls,_=classify(meta)
    cache=torch.load(Path(str(E.CACHE).format('raw')),map_location='cpu',weights_only=False,mmap=True)
    assert len(cache['ctx'])==len(meta['seed'])==1002
    cal={'moved':{'classified':0,'correct':0,'ambiguous':0},'blocked':{'classified':0,'correct':0,'ambiguous':0}}
    for i in range(len(meta['seed'])):
        root=meta['root_visible'][i]
        for act in range(1,5):
            truth='moved' if int(cls[i,act])==0 else 'blocked' if int(cls[i,act])==1 else None
            if truth is None:continue
            label,_,_=terrain_decision(root,meta['onestep_visible'][i,0,act],act)
            cell=cal[truth]
            if label<0:cell['ambiguous']+=1
            else:
                cell['classified']+=1
                cell['correct']+=int((label!=0)==(truth=='moved'))
    R,H=cache['fut'].shape[:2]
    alive=~meta['future_dead'][:,0].cumsum(1).bool()
    out={'moving_actions':0,'high_confidence_scrolled':0,'high_confidence_blocked':0,
         'ambiguous_move_attempts':0,'all_alive_transitions':int(alive.sum()),
         'scrolled':{'fixed_err_sum':0.,'aligned_err_sum':0.,'aligned_lower_share_sum':0.,
                     'changed_zombie_cells':0,'overlap_cells':0,
                     'zombie_change_err_sum':0.,'zombie_change_pairs':0,
                     'zombie_stable_err_sum':0.,'zombie_stable_pairs':0},
         'per_seed':{}}
    for i in range(R):
        prior_v=meta['root_visible'][i]
        prior_t=cache['ctx'][i,-1]
        seed=int(meta['seed'][i])
        bucket=out['per_seed'].setdefault(seed,{'scrolled':0,'fixed_err_sum':0.,'aligned_err_sum':0.})
        for k in range(H):
            if not bool(alive[i,k]):break
            act=int(meta['future_actions'][i,k])
            next_v=meta['future_visible'][i,0,k]
            next_t=cache['fut'][i,k]
            if 1<=act<=4:
                out['moving_actions']+=1
                label,_,_=terrain_decision(prior_v,next_v,act)
                if label<0:out['ambiguous_move_attempts']+=1
                elif label==0:out['high_confidence_blocked']+=1
                else:
                    out['high_confidence_scrolled']+=1
                    f,a,better,zc,allc,ze,se=compare_tokens(prior_t,next_t,label,
                                                               zombie(prior_v),zombie(next_v))
                    d=out['scrolled'];d['fixed_err_sum']+=f;d['aligned_err_sum']+=a
                    d['aligned_lower_share_sum']+=better;d['changed_zombie_cells']+=zc;d['overlap_cells']+=allc
                    if ze is not None:d['zombie_change_err_sum']+=ze;d['zombie_change_pairs']+=1
                    if se is not None:d['zombie_stable_err_sum']+=se;d['zombie_stable_pairs']+=1
                    bucket['scrolled']+=1;bucket['fixed_err_sum']+=f;bucket['aligned_err_sum']+=a
            prior_v,prior_t=next_v,next_t
    d=out['scrolled'];n=out['high_confidence_scrolled']
    d['mean_fixed_mse']=d['fixed_err_sum']/n if n else None
    d['mean_aligned_mse']=d['aligned_err_sum']/n if n else None
    d['relative_mse_reduction']=1-d['aligned_err_sum']/d['fixed_err_sum'] if n else None
    d['mean_fraction_cells_better']=d['aligned_lower_share_sum']/n if n else None
    d['zombie_changed_fraction']=d['changed_zombie_cells']/d['overlap_cells'] if n else None
    d['mean_aligned_err_zombie_changed_pair']=d['zombie_change_err_sum']/d['zombie_change_pairs'] if d['zombie_change_pairs'] else None
    d['mean_aligned_err_zombie_stable_pair']=d['zombie_stable_err_sum']/d['zombie_stable_pairs'] if d['zombie_stable_pairs'] else None
    rows=list(out['per_seed'].values());g=torch.Generator().manual_seed(20260929);vals=[]
    for _ in range(1000):
        idx=torch.randint(len(rows),(len(rows),),generator=g)
        f=sum(rows[int(j)]['fixed_err_sum'] for j in idx)
        a=sum(rows[int(j)]['aligned_err_sum'] for j in idx)
        if f>0:vals.append(1-a/f)
    d['relative_mse_reduction_seed_bootstrap_ci95']=torch.tensor(vals).quantile(torch.tensor([.025,.975])).tolist()
    out['calibration_on_one_step_simulator_moved_blocked']=cal
    out['source_sha256']=digest(__file__)
    out['raw_cache_sha256']=digest(str(E.CACHE).format('raw'))
    del out['per_seed']
    HERE.joinpath('registration_scope.json').write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps(out,indent=2),flush=True)

if __name__=='__main__':main()
