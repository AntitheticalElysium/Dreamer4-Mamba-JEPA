"""Exploratory causal corrt move-gate reliability on true roots and generated prefixes.

No treatment training. Threshold fitted on TRAIN-seed one-step forks, then held fixed on TEST-seed
one-step forks and the same TEST roots' 16-step factual action sequences in imagination.
The true-future movement label uses the independently calibrated visible-terrain rule and abstains.
"""
import hashlib,json,sys
from pathlib import Path
import torch
ROOT=Path('/home/antithetical/EPITA/PERSO/DynamicHorizons-Mamba-JEPA')
HERE=Path(__file__).resolve().parent
sys.path[:0]=[str(ROOT),str(ROOT/'artifacts/experiments/20260927_levers'),
              str(ROOT/'artifacts/experiments/20260926_diagnosis'),str(HERE)]
import teval as E
import tworld as T
from onestep import classify
from compound import auc
from registration_scope import terrain_decision
from scroll import estimate,MOVE_SHIFT
import spatial as S
from d4mj.config import config_from_dict

CKPT=ROOT/'artifacts/eda/levers_tworlds_v1/corrt_raw_suffix_s7_fcanvas.pt'

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def threshold_fit(x,y):
    z,order=x.sort(); ys=y[order].long(); pos=int(ys.sum());neg=len(y)-pos
    cp=ys.cumsum(0);cn=torch.arange(1,len(y)+1)-cp
    score=(pos-cp).float()/(2*pos)+cn.float()/(2*neg)
    idx=int(score.argmax());return float((z[idx]+z[min(idx+1,len(z)-1)])/2)

def summary(x,y,threshold):
    if not len(x):return {'n':0}
    pred=x>threshold
    return {'n':len(x),'moved_share':float(y.float().mean()),
            'auc':float(auc(x,y)) if y.any() and (~y).any() else None,
            'accuracy':float((pred==y).float().mean()),
            'balanced_accuracy':float((((pred&y).sum()/y.sum())+((~pred&~y).sum()/(~y).sum()))/2) if y.any() and (~y).any() else None,
            'tpr':float((pred&y).sum()/y.sum()) if y.any() else None,
            'tnr':float((~pred&~y).sum()/(~y).sum()) if (~y).any() else None}

@torch.no_grad()
def main():
    device=torch.device('cuda')
    meta,train_roots,_=E.split();cls,_=classify(meta)
    cache=torch.load(Path(str(E.CACHE).format('raw')),map_location='cpu',weights_only=False,mmap=True)
    world,payload=E.load_world(CKPT,device)
    assert payload['args']['head']=='corrt' and payload['args']['backbone']=='fcanvas'
    source=Path(T.__file__).read_text()
    now='POOLS = {"raw": ROOT / "artifacts/eda/spatial_pool_v1", "tc": ROOT / "artifacts/eda/spatial_pool_tc_v1",\n         "ldad10": ROOT / "artifacts/eda/spatial_pool_ldad10_v1",\n         "ldad1": ROOT / "artifacts/eda/spatial_pool_ldad1_v1"}'
    old='POOLS = {"raw": ROOT / "artifacts/eda/spatial_pool_v1", "tc": ROOT / "artifacts/eda/spatial_pool_tc_v1",\n         "ldad10": ROOT / "artifacts/eda/spatial_pool_ldad10_v1"}'
    assert now in source and hashlib.sha256(source.replace(now,old).encode()).hexdigest()==payload['script_sha256']
    cfg=config_from_dict(torch.load(S.CHECKPOINT,map_location='cpu',weights_only=False)['config'])
    captured=[];original=world.backbone_full
    def tap(frames,actions):
        h,ha=original(frames,actions)
        target=torch.tensor([31,30,32,22,40],device=actions.device)[actions.clamp(max=4)]
        ht=h.gather(2,target[:,:,None,None].expand(-1,-1,1,h.shape[-1]))[:,:,0]
        moved=world.frame(ha).float()+((actions>=1)&(actions<=4))[:,:,None]*world.target_gate(ht).float()
        captured.append(moved[:,-1,0].cpu())
        return h,ha
    world.backbone_full=tap
    # Threshold/calibration: one-step true contexts, all four move actions.
    rows=[(i,a) for i in range(len(cache['ctx'])) for a in (1,2,3,4) if int(cls[i,a]) in (0,1)]
    logits=[]
    for start in range(0,len(rows),16):
        chunk=rows[start:start+16]
        frames=torch.stack([cache['ctx'][i] for i,_ in chunk]).float()
        actions=torch.stack([torch.cat([cache['ctx_a'][i],torch.tensor([a])]) for i,a in chunk])
        E.step(world,frames,actions,device,cfg)
        logits.append(captured.pop())
    logits=torch.cat(logits)
    labels=torch.tensor([int(cls[i,a])==0 for i,a in rows])
    tr=torch.tensor([bool(train_roots[i]) for i,a in rows]);te=~tr
    threshold=threshold_fit(logits[tr],labels[tr])
    factual={'train':summary(logits[tr],labels[tr],threshold),
             'test':summary(logits[te],labels[te],threshold)}
    print(json.dumps({'stage':'factual_gate','threshold':threshold,'test':factual['test']}),flush=True)
    # True future movement: visible terrain label; ambiguous and non-move rows are excluded.
    test_ix=torch.where(~train_roots)[0]
    truth=torch.full((len(test_ix),E.H),-1,dtype=torch.long)
    for q,i in enumerate(test_ix):
        prev=meta['root_visible'][i]
        for k in range(E.H):
            action=int(meta['future_actions'][i,k]);cur=meta['future_visible'][i,0,k]
            if 1<=action<=4:truth[q,k]=terrain_decision(prev,cur,action)[0]
            prev=cur
    alive=~meta['future_dead'][test_ix,0].cumsum(1).bool()
    gate=torch.empty(len(test_ix),E.H)
    pair_shift=torch.empty(len(test_ix),E.H,dtype=torch.long)
    for start in range(0,len(test_ix),8):
        sl=test_ix[start:start+8];ctx=cache['ctx'][sl].float()
        ca,fa=cache['ctx_a'][sl],cache['fut_a'][sl]
        frames=[ctx[:,j] for j in range(4)];actions=[ca[:,j] for j in range(3)]
        for k in range(E.H):
            w=4 if k==0 else 5
            a=torch.stack(actions[-(w-1):]+[fa[:,k]],1)
            nxt=E.step(world,torch.stack(frames[-w:],1),a,device,cfg)
            gate[start:start+len(sl),k]=captured.pop()
            pair_shift[start:start+len(sl),k]=estimate(frames[-1],nxt)
            frames.append(nxt);actions.append(fa[:,k])
        if (start//8+1)%10==0:print(json.dumps({'stage':'rollout','roots':start+len(sl)}),flush=True)
    assert not captured
    out={'status':'complete','world_sha256':sha(CKPT),'world_source_sha256':payload['script_sha256'],
         'raw_cache_sha256':sha(str(E.CACHE).format('raw')),'script_sha256':sha(__file__),
         'threshold_train_only':threshold,'one_step':factual,'by_depth':{}}
    pred=gate>threshold
    test_seed=meta['seed'][test_ix]
    groups=[torch.where(test_seed==u)[0] for u in test_seed.unique()]
    torch.save({'gate':gate,'pair_shift':pair_shift,'truth':truth,'alive':alive,'seed':test_seed,
                'test_root_index':test_ix,'threshold_train_only':threshold},HERE/'learned_gate_rollout_rows.pt')
    out['rows_sha256']=sha(HERE/'learned_gate_rollout_rows.pt')
    for name,lo,hi in [('1',0,1),('2-4',1,4),('5-8',4,8),('9-16',8,16),('all',0,16)]:
        m=(truth[:,lo:hi]>=0)&alive[:,lo:hi]
        x=gate[:,lo:hi][m];y=(truth[:,lo:hi][m]>0);p=pred[:,lo:hi][m]
        shift=pair_shift[:,lo:hi][m]
        # Action-derived direction is known in advance to either gate; compare movement detection only.
        q=(shift!=0)
        row=summary(x,y,threshold)
        row['pair_estimator_accuracy']=float((q==y).float().mean()) if len(y) else None
        row['gate_minus_pair_accuracy']=float((p==y).float().mean()-(q==y).float().mean()) if len(y) else None
        row['abstained_move_rows']=int((((truth[:,lo:hi]<0)&alive[:,lo:hi])&((meta['future_actions'][test_ix,lo:hi]>=1)&(meta['future_actions'][test_ix,lo:hi]<=4))).sum())
        row['roots']=int(m.any(1).sum());row['seeds']=int(test_seed[m.any(1)].unique().numel())
        row['mean_gate_logit_moved']=float(x[y].mean()) if y.any() else None
        row['mean_gate_logit_blocked']=float(x[~y].mean()) if (~y).any() else None
        rng=torch.Generator().manual_seed(20260929);diffs=[]
        mm=(truth[:,lo:hi]>=0)&alive[:,lo:hi]
        for _ in range(2000):
            choice=torch.randint(len(groups),(len(groups),),generator=rng)
            ix=torch.cat([groups[int(j)] for j in choice])
            valid=mm[ix];yy=truth[ix,lo:hi][valid]>0
            if not len(yy):continue
            pp=pred[ix,lo:hi][valid];qq=pair_shift[ix,lo:hi][valid]!=0
            diffs.append(float((pp==yy).float().mean()-(qq==yy).float().mean()))
        row['gate_minus_pair_seed_ci95']=torch.tensor(diffs).quantile(torch.tensor([.025,.975])).tolist()
        out['by_depth'][name]=row
    HERE.joinpath('learned_gate_rollout.json').write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps({'stage':'complete','by_depth':out['by_depth']}),flush=True)
if __name__=='__main__':main()
