"""CPU decomposition of the saved six-candidate corrt decoder emissions.

Usage: decoder_decomposition.py <eval_uNNNN/pred_ARM.pt> ...
Requires saved FP32 weights and generated candidates, verifies reconstruction
before interpreting candidates. Atomic root batches are resumable. Candidate
ablations alter only the emitted token; they do not run a counterfactual world.
"""
import argparse
import hashlib
import json
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "artifacts/eda/oracle_v1"
MAPC = [i for i in range(63) if i != 31]
NEIGHBOURS = ((-1, 0), (1, 0), (0, -1), (0, 1))
NAMES = ("self", "up", "down", "left", "right", "generator")
TYPES = ("zombie", "skeleton", "cow", "arrow")
LAYOUT_CH = (0, 2, 1, 3)
GROUPS = {"1": [0], "2-4": [1, 2, 3], "5-8": [4, 5, 6, 7], "9-16": list(range(8, 16))}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(1024*1024), b""): h.update(b)
    return h.hexdigest()


def atomic(value, path):
    tmp = path.with_suffix(path.suffix + ".tmp")
    if path.suffix == '.json': tmp.write_text(json.dumps(value, indent=2))
    else: torch.save(value,tmp)
    tmp.replace(path)


class Reader:
    def __init__(self):
        s=torch.load(OUT/'snap_bank_readers.pt',map_location='cpu',weights_only=False)
        self.mob=s['heads']['mob'];self.thr=torch.tensor(s['thresholds'])
    def scores(self,x):
        q=x.reshape(-1,192).float();s=self.mob;out=[]
        for i in range(0,len(q),4096):
            z=((q[i:i+4096]-s['mu'])/s['sd']).double()
            out.append((torch.cat([z,torch.ones(len(z),1,dtype=torch.float64)],1)@s['w']).float())
        return torch.cat(out).view(*x.shape[:-1],4)
    def decode(self,x):return self.scores(x)>self.thr


def source_positions():
    grid=torch.arange(81).view(9,9)
    pad=F.pad(grid,(1,1,1,1),value=-1)
    return torch.stack([grid]+[pad[1+dr:10+dr,1+dc:10+dc] for dr,dc in NEIGHBOURS],-1).view(81,5)


def candidates(previous,generated):
    grid=previous.view(*previous.shape[:2],9,9,192)
    pad=F.pad(grid,(0,0,1,1,1,1))
    c=[grid]+[pad[:,:,1+dr:10+dr,1+dc:10+dc] for dr,dc in NEIGHBOURS]
    return torch.stack([v.reshape_as(previous) for v in c]+[generated],-2)


def measure_batch(pred,weights,generated,previous,truth,prev_truth,true_tokens,reader):
    # Each tensor is [B,H,81,*] apart from layouts [B,H,63,4].
    assert all(v.dtype==torch.float32 for v in (pred,weights,generated,previous))
    assert weights.shape[-1]==6 and torch.allclose(weights.sum(-1),torch.ones_like(weights[...,0]),atol=2e-6)
    c=candidates(previous,generated)
    mix=(weights[...,None]*c).sum(-2)
    rebuilt=F.layer_norm(mix,(192,))
    diff=(rebuilt-pred).abs()
    maximum=float(diff.max())
    assert torch.allclose(rebuilt,pred,atol=5e-5,rtol=1e-5), {'decoder_reconstruction_max_abs':maximum}
    rebuilt_dec=reader.decode(rebuilt)
    decoded=reader.decode(pred)
    assert torch.equal(rebuilt_dec[:,:,:63][:,:,MAPC],decoded[:,:,:63][:,:,MAPC]), 'CPU reconstruction changed a map mob decision'
    positions=source_positions()
    masks={'player':positions==31,'hud':positions>=63,'padding':positions<0}
    maps=torch.zeros_like(decoded)
    maps[:,:,:63]=truth[...,LAYOUT_CH]>0
    maps[:,:,31]=False
    target_dec=reader.decode(true_tokens)
    map_mask=torch.zeros(81,dtype=torch.bool);map_mask[MAPC]=True
    physical_fp=decoded&~maps&map_mask[None,None,:,None]
    clean_fp=physical_fp&~target_dec
    tp=decoded&maps&map_mask[None,None,:,None]
    cand_norm=F.layer_norm(c,(192,))
    cand_dec=reader.decode(cand_norm)                  # [B,H,81,6,4]
    copy_positive=cand_dec[...,:5,:].any(-2)
    gen_positive=cand_dec[...,5,:]
    pre_dec=reader.decode(mix)
    copy_mix=(weights[...,:5,None]*c[...,:5,:]).sum(-2)
    copy_weight=weights[...,:5].sum(-1,keepdim=True)
    alternatives={
        'copy_only_retained_scale':F.layer_norm(copy_mix,(192,)),
        'copy_only_renormalized':F.layer_norm(copy_mix/copy_weight.clamp_min(1e-12),(192,)),
        'generator_only':cand_norm[...,5,:],
    }
    for label,remove in (('no_player',masks['player']),('no_hud',masks['hud']),
                         ('no_player_hud',masks['player']|masks['hud']),('no_padding',masks['padding'])):
        keep=torch.cat([~remove,torch.ones(81,1,dtype=torch.bool)],-1)
        w=weights*keep
        w=w/w.sum(-1,keepdim=True).clamp_min(1e-12)
        alternatives[label]=F.layer_norm((w[...,None]*c).sum(-2),(192,))
    metrics={'true':maps.sum(2), 'true_reader_tp':(target_dec&maps&map_mask[None,None,:,None]).sum(2),
             'true_reader_fp':(target_dec&~maps&map_mask[None,None,:,None]).sum(2),
             'full_tp':tp.sum(2), 'full_fp':physical_fp.sum(2), 'clean_fp':clean_fp.sum(2),
             'generator_only_positive':(clean_fp&gen_positive&~copy_positive).sum(2),
             'copy_only_positive':(clean_fp&copy_positive&~gen_positive).sum(2),
             'both_candidate_types_positive':(clean_fp&copy_positive&gen_positive).sum(2),
             'emergent_no_candidate_positive':(clean_fp&~copy_positive&~gen_positive).sum(2),
             'final_ln_crossing':(clean_fp&~pre_dec).sum(2)}
    alt_dec={name:reader.decode(v) for name,v in alternatives.items()}
    for name,de in alt_dec.items():
        metrics[name+'_tp']=(de&maps&map_mask[None,None,:,None]).sum(2)
        metrics[name+'_fp']=(de&~maps&map_mask[None,None,:,None]).sum(2)
        metrics[name+'_clean_fp_removed']=(clean_fp&~de).sum(2)
        metrics[name+'_clean_fp_remaining']=(clean_fp&de).sum(2)
    for j,name in enumerate(NAMES):
        metrics['candidate_'+name+'_clean_fp_positive']=(clean_fp&cand_dec[...,j,:]).sum(2)
        metrics['candidate_'+name+'_weight_on_clean_fp']=(weights[...,j,None]*clean_fp).sum(2)
    # Where positive copied candidates originated. Physical labels are a
    # reference under imagined feedback, rather than labels of the imagined token.
    prior_flags=torch.zeros(*prev_truth.shape[:2],81,4,dtype=torch.bool)
    prior_flags[:,:,:63]=prev_truth[...,LAYOUT_CH]>0
    prior_flags[:,:,31]=False
    source_flags=prior_flags[:,:,positions.clamp(min=0)]
    source_flags &= (positions>=0)[None,None,:,:,None]
    copy_dec=cand_dec[...,:5,:]
    for label,source in (('physical_same_mob',source_flags),('physical_absent',~source_flags)):
        positive=(copy_dec&source).any(-2)
        metrics['copy_'+label+'_available_on_clean_fp']=(clean_fp&positive).sum(2)
    for label,mask in masks.items():
        positive=(copy_dec&mask[None,None,:,:,None]).any(-2)
        metrics['copy_'+label+'_positive_on_clean_fp']=(clean_fp&positive).sum(2)
        mass=(weights[...,:5]*mask).sum(-1)
        metrics['copy_'+label+'_weight_on_clean_fp']=(mass[...,None]*clean_fp).sum(2)
    # A readable FALSE predecessor copied into today's false positive is the
    # operational propagation signature, conditional on the recorded truth.
    regular=(positions>=0)&(positions<63)&(positions!=31)
    bad_copy=(copy_dec&~source_flags&regular[None,None,:,:,None]).any(-2)
    metrics['copy_regular_false_predecessor_on_clean_fp']=(clean_fp&bad_copy).sum(2)
    metrics['generator_weight_on_clean_fp']=metrics['candidate_generator_weight_on_clean_fp']
    assert torch.equal(metrics['clean_fp'],sum(metrics[k] for k in ('generator_only_positive','copy_only_positive','both_candidate_types_positive','emergent_no_candidate_positive')))
    return {'metrics':metrics,'reconstruction_max_abs':maximum,
            'reconstruction_mean_abs':float(diff.mean()),'reconstruction_mob_equal':True}


def summarize(rows,aligned):
    res={}
    for mask_name,mask in (('all',torch.ones_like(aligned)),('aligned',aligned)):
        groups={}
        for group,steps in GROUPS.items():
            sel=mask[:,steps]
            out={'root_steps':int(sel.sum()),'types':{}}
            for j,typ in enumerate(TYPES):
                values={k:float(v[:,steps,j][sel].sum()) for k,v in rows.items()}
                clean=values['clean_fp']
                true=values['true']
                values['full_recall']=values['full_tp']/true if true else None
                values['full_fp_per_true']=values['full_fp']/true if true else None
                values['clean_fp_generator_weight_mean']=values['generator_weight_on_clean_fp']/clean if clean else None
                for k in ('generator_only_positive','copy_only_positive','both_candidate_types_positive','emergent_no_candidate_positive','final_ln_crossing'):
                    values[k+'_share']=values[k]/clean if clean else None
                out['types'][typ]=values
            groups[group]=out
        res[mask_name]=groups
    return res


def main():
    p=argparse.ArgumentParser();p.add_argument('predictions',nargs='+');p.add_argument('--batch',type=int,default=8)
    args=p.parse_args();torch.set_num_threads(4);reader=Reader()
    panel=torch.load(OUT/'panel.pt',map_location='cpu',weights_only=False)
    for argument in args.predictions:
        path=Path(argument).resolve(); saved=torch.load(path,map_location='cpu',weights_only=False)
        required=('tf','gen','tf_weights','gen_weights','tf_generated','gen_generated','aligned')
        assert all(k in saved for k in required), f'missing saved decoder intermediates: {path}'
        arm=path.stem.removeprefix('pred_'); destination=path.parent/f'decoder_decomposition_{arm}.json'
        contract={'schema':'corrt_decoder_decomposition_v1','inputs':{str(f.relative_to(ROOT)):sha(f) for f in
                  (path,OUT/'panel.pt',OUT/'snap_bank_readers.pt',Path(__file__),ROOT/'artifacts/experiments/20260927_levers/tworld.py')},
                  'prediction_contract':saved.get('contract_id'),'batch':args.batch,
                  'candidate_order':NAMES,'target_cells':MAPC,'thresholds':'original frozen TRAIN reader',
                  'ablations':'emission-only; no altered feedback trajectories',
                  'conditional_fp':'physically absent AND actual target reader absent',
                  'copy_ablation_scale':'retained and renormalized both reported'}
        cid=hashlib.sha256(json.dumps(contract,sort_keys=True).encode()).hexdigest()
        resume=path.parent/'decoder_decomposition_resume'/arm;resume.mkdir(parents=True,exist_ok=True)
        R,H=saved['gen'].shape[:2]; result={'contract':contract,'contract_id':cid,'arm':arm,'modes':{}}
        for mode in ('tf','gen'):
            chunks=[]
            for start in range(0,R,args.batch):
                end=min(start+args.batch,R); cache=resume/f'{mode}_{start:04d}_{end:04d}.pt'
                if cache.exists():
                    piece=torch.load(cache,map_location='cpu',weights_only=False)
                    assert piece['contract_id']==cid, 'resume source/input drift'
                else:
                    sl=slice(start,end)
                    previous=panel['tokens'][sl,15:31].float() if mode=='tf' else torch.cat([panel['tokens'][sl,15:16].float(),saved['gen'][sl,:-1]],1)
                    measured=measure_batch(saved[mode][sl],saved[mode+'_weights'][sl],saved[mode+'_generated'][sl],previous,
                                          panel['layouts'][sl,16:32],panel['layouts'][sl,15:31],panel['tokens'][sl,16:32].float(),reader)
                    piece={'contract_id':cid,'start':start,'end':end,**measured};atomic(piece,cache)
                chunks.append(piece)
                print(json.dumps({'stage':'decoder_batch','arm':arm,'mode':mode,'roots':end,'max_abs':piece['reconstruction_max_abs']}),flush=True)
            rows={k:torch.cat([b['metrics'][k] for b in chunks]) for k in chunks[0]['metrics']}
            aligned=saved['aligned'] if mode=='gen' else torch.ones(R,H,dtype=torch.bool)
            result['modes'][mode]={'reconstruction_max_abs':max(b['reconstruction_max_abs'] for b in chunks),
                                   'reconstruction_mob_equal':all(b['reconstruction_mob_equal'] for b in chunks),
                                   'groups':summarize(rows,aligned),'rows':{k:v.tolist() for k,v in rows.items()}}
        atomic(result,destination)
        print(json.dumps({'stage':'decoder_done','path':str(destination),'late':{mode:result['modes'][mode]['groups']['all']['9-16']['types']['cow'] for mode in ('tf','gen')}}),flush=True)


if __name__=='__main__': main()
