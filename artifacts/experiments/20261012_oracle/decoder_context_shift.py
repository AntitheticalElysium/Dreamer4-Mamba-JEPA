"""CPU factual/imagined decoder-context comparison at identical target cells.

Saved TF and imagined paths use the SAME oracle layout at each predicted step.
Eight emission-only swaps separate routing, copied contents, and raw generator.
Primary source interpretation requires camera alignment at BOTH current and
previous steps and an actual TRUE source token whose same reader is absent.
"""
import argparse
import hashlib
import itertools
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from decoder_decomposition import ROOT, OUT, MAPC, NAMES, TYPES, LAYOUT_CH, GROUPS, Reader, candidates, source_positions, sha, atomic


def decode(reader,tokens):
    d=reader.decode(tokens);d[:,:,31]=False;d[:,:,63:]=False
    return d


def summary(rows,mask,draws=2000):
    R,H=mask.shape;result={}
    rng=torch.Generator().manual_seed(20261012)
    boot=torch.randint(R,(draws,R),generator=rng)
    for group,steps in GROUPS.items():
        sel=mask[:,steps];item={'root_steps':int(sel.sum()),'types':{}}
        for j,typ in enumerate(TYPES):
            values={k:float(v[:,steps,j][sel].sum()) for k,v in rows.items()}
            cp=values['clean_gen_fp']
            for mode in ('tf','gen'):
                for cand in NAMES:
                    key=f'{mode}_weight_{cand}_on_clean_gen_fp'
                    values[key+'_mean']=values[key]/cp if cp else None
            for prefix in ('wgen_copygen_gengen','wtf_copygen_gengen','wgen_copytf_gengen','wgen_copygen_gentf','wtf_copytf_gentf'):
                tr=values['true'];values[prefix+'_recall']=values[prefix+'_tp']/tr if tr else None
            contrast={}
            baseline='wgen_copygen_gengen'
            denominator=(rows['true'][:,steps,j]*sel).sum(1).double()
            for prefix in ('wtf_copygen_gengen','wgen_copytf_gengen','wgen_copygen_gentf','wtf_copytf_gentf'):
                c={}
                for statistic in ('fp','tp'):
                    diff=((rows[prefix+'_'+statistic][:,steps,j]-rows[baseline+'_'+statistic][:,steps,j])*sel).sum(1).double()
                    bs=diff[boot].sum(1)/denominator[boot].sum(1).clamp_min(1)
                    c[statistic+'_per_true_delta']={'estimate':float(diff.sum()/denominator.sum().clamp_min(1)),
                                                  'interval':[float(bs.quantile(.025)),float(bs.quantile(.975))]}
                contrast[prefix]=c
            item['types'][typ]={'counts':values,'paired_contrasts':contrast}
        result[group]=item
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('predictions',nargs='+');args=p.parse_args();torch.set_num_threads(4)
    panel=torch.load(OUT/'panel.pt',map_location='cpu',weights_only=False);reader=Reader();pos=source_positions()
    regular=(pos>=0)&(pos<63)&(pos!=31)
    for arg in args.predictions:
        path=Path(arg).resolve();st=torch.load(path,map_location='cpu',weights_only=False);R,H=st['gen'].shape[:2]
        result={'schema':'corrt_decoder_context_shift_v1','inputs':{str(f.relative_to(ROOT)):sha(f) for f in (path,OUT/'panel.pt',OUT/'snap_bank_readers.pt',Path(__file__),Path(__file__).with_name('decoder_decomposition.py'))},
                'source_reference':'generated positive AND physical previous mob absent AND TRUE previous source reader absent; only both-aligned camera subset is physically interpretable',
                'swaps':'emission-only; no altered hidden states or recursive trajectories','rows':{},'groups':{}}
        chunks=[]
        for start in range(0,R,8):
            end=min(start+8,R);sl=slice(start,end)
            prev_tf=panel['tokens'][sl,15:31].float()
            prev_gen=torch.cat([panel['tokens'][sl,15:16].float(),st['gen'][sl,:-1]],1)
            ctf=candidates(prev_tf,st['tf_generated'][sl]);cgen=candidates(prev_gen,st['gen_generated'][sl])
            actual=torch.zeros(end-start,H,81,4,dtype=torch.bool);actual[:,:,:63]=panel['layouts'][sl,16:32,...,LAYOUT_CH]>0;actual[:,:,31]=False
            prev_actual=torch.zeros_like(actual);prev_actual[:,:,:63]=panel['layouts'][sl,15:31,...,LAYOUT_CH]>0;prev_actual[:,:,31]=False
            true_dec=decode(reader,panel['tokens'][sl,16:32].float())
            gd=decode(reader,st['gen'][sl]);td=decode(reader,st['tf'][sl]);clean=gd&~actual&~true_dec
            gen_copy_de=reader.decode(F.layer_norm(cgen[...,:5,:],(192,)))
            tf_copy_de=reader.decode(F.layer_norm(ctf[...,:5,:],(192,)))
            previous_physical=prev_actual[:,:,pos.clamp_min(0)]&(pos>=0)[None,None,:,:,None]
            strong=gen_copy_de&~previous_physical&~tf_copy_de&regular[None,None,:,:,None]
            weak=gen_copy_de&~previous_physical&regular[None,None,:,:,None]
            rows={'true':actual.sum(2),'clean_gen_fp':clean.sum(2),'tf_also_fp_on_clean_gen_fp':(clean&td).sum(2),
                  'false_copy_predecessor_on_clean_gen_fp':(clean&weak.any(-2)).sum(2),
                  'strong_false_copy_predecessor_on_clean_gen_fp':(clean&strong.any(-2)).sum(2),
                  'strong_false_self_predecessor_on_clean_gen_fp':(clean&strong[...,0,:]).sum(2)}
            for mode in ('tf','gen'):
                for j,cand in enumerate(NAMES):
                    rows[f'{mode}_weight_{cand}_on_clean_gen_fp']=(st[mode+'_weights'][sl,...,j,None]*clean).sum(2)
            for wsrc,copy,gen in itertools.product(('gen','tf'),repeat=3):
                cs=torch.cat([(cgen if copy=='gen' else ctf)[...,:5,:],(cgen if gen=='gen' else ctf)[...,5:6,:]],-2)
                out=F.layer_norm((st[wsrc+'_weights'][sl,...,None]*cs).sum(-2),(192,))
                key=f'w{wsrc}_copy{copy}_gen{gen}'
                d=decode(reader,out)
                if wsrc==copy==gen:
                    ref=st[wsrc][sl]
                    assert torch.allclose(out,ref,atol=5e-5,rtol=1e-5), (key,float((out-ref).abs().max()))
                    assert torch.equal(d,td if wsrc=='tf' else gd), key
                rows[key+'_tp']=(d&actual).sum(2)
                rows[key+'_fp']=(d&~actual).sum(2)
                rows[key+'_clean_gen_fp_remaining']=(d&clean).sum(2)
            chunks.append(rows)
            print(json.dumps({'stage':'context_batch','roots':end,'total':R}),flush=True)
        rows={k:torch.cat([v[k] for v in chunks]) for k in chunks[0]}
        both=st['aligned']&torch.cat([torch.ones(R,1,dtype=torch.bool),st['aligned'][:,:-1]],1)
        result['rows']={k:v.tolist() for k,v in rows.items()}
        result['groups']={'all':summary(rows,torch.ones(R,H,dtype=torch.bool)), 'both_aligned':summary(rows,both)}
        result['both_aligned']=both.tolist()
        dest=path.parent/path.name.replace('pred_','decoder_context_shift_').replace('.pt','.json')
        atomic(result,dest)
        print(json.dumps({'stage':'context_done','path':str(dest),'late':{k:result['groups'][k]['9-16']['types']['cow'] for k in ('all','both_aligned')}}),flush=True)


if __name__=='__main__':main()
