"""CPU separating readings: completion, channel use, and sampled distributions.

Generated heads are fitted ONLY on posterior-completed recorded branches. Prior
samples are never paired with the recorded branch's labels during head fitting.
All TEST readout episodes are excluded from fitting and alpha/threshold choice.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).parent))
import innovation_train as T
import loss_intervention as L
import generated_readout_refit as R
import snap_eval as S

CELLS=R.CELLS
REGIONS={'whole':list(range(81)),'map':list(range(63)),
         'near':[22,40,30,32],'HUD':list(range(63,81))}


def energy(x,y):
    """Standard energy score averaged over independent real draws, per root.

Euclidean distance / sqrt(dim); off-diagonal ensemble pairs give an unbiased
diversity term. Deterministic forecasts have a zero diversity term.
"""
    x=x.double().flatten(2);y=y.double().flatten(2)
    xy=torch.cdist(x,y,compute_mode='donot_use_mm_for_euclid_dist').mean((1,2))/np.sqrt(x.shape[-1])
    n=x.shape[1]
    diversity=torch.cdist(x,x,compute_mode='donot_use_mm_for_euclid_dist').sum((1,2))/(n*(n-1)) if n>1 else torch.zeros(len(x),dtype=torch.float64)
    return xy-.5*diversity/np.sqrt(x.shape[-1])


def interval(values,weights):
    values=np.asarray(values,dtype=float)
    boot=weights@values/weights.sum(1)
    return {'mean':float(values.mean()),'ci95':np.quantile(boot,[.025,.975]).tolist()}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--updates',type=int,nargs='+',default=[0,500,1500,3000])
    p.add_argument('--horizon',type=int,choices=(1,16),default=1)
    p.add_argument('--run-directory',type=Path,default=T.OUT)
    a=p.parse_args();torch.set_num_threads(4)
    panel=torch.load(L.OUT/'panel.pt',map_location='cpu',weights_only=False)
    bank=torch.load(L.OUT/'snap_bank_readers.pt',map_location='cpu',weights_only=False)
    reader=S.Readers(bank)
    visible=torch.load(L.OUT/'snap_visible.pt',map_location='cpu',weights_only=False)
    assert visible['roots']==panel['roots']
    real=panel['tokens'][:,16:16+a.horizon].float()
    hud_truth=visible['visible'][:,16:16+a.horizon,1512:1516].float()*9
    tile_truth=visible['visible'][:,16:16+a.horizon,:1071].view(200,a.horizon,63,17).argmax(-1)
    labels=panel['layouts'][:,16:16+a.horizon,CELLS][:,:,:,[0,2,1,3]].float()
    permutation=np.random.default_rng(20261012).permutation(200)
    fit,dev,test=permutation[:100],permutation[100:150],permutation[150:]
    weights=np.random.default_rng(20261013).multinomial(50,np.full(50,1/50),size=2000)
    fan=None
    if a.horizon==1:
        fan=torch.load(a.run_directory/'real_futures/fan.pt',map_location='cpu',weights_only=False)
        assert fan['roots']==panel['roots'] and fan['replay_maxdiff']==0
    for update in a.updates:
        directory=a.run_directory/f'eval_u{update:04d}_h{a.horizon:02d}'
        paths=sorted(directory.glob('pred_*.pt'))
        assert paths, directory
        source_paths=[__file__,R.__file__,S.__file__]
        contract={'format':'innovation_separating_readings_v1','sources':{str(Path(p).relative_to(L.ROOT)):L.sha(p)for p in source_paths},
                  'prediction_hashes':{p.name:L.sha(p)for p in paths},'panel_sha256':L.sha(L.OUT/'panel.pt'),
                  'reader_sha256':L.sha(L.OUT/'snap_bank_readers.pt'),'horizon':a.horizon,
                  'FIT':[panel['roots'][i]['episode_id']for i in fit],
                  'DEV':[panel['roots'][i]['episode_id']for i in dev],
                  'TEST':[panel['roots'][i]['episode_id']for i in test],
                  'proper_score':'energy score, normalized Euclidean, unbiased ensemble diversity term; 32 real draws',
                  'head_training':'real branch and each arm posterior branch only; never independently sampled prior labels',
                  'scope':'reused DEV world roots; 50 readout TEST episodes; not an actor evaluation'}
        if fan is not None:contract['fan_sha256']=L.sha(a.run_directory/'real_futures/fan.pt')
        cid=hashlib.sha256(json.dumps(contract,sort_keys=True).encode()).hexdigest()
        manifest=directory/'analysis_contract.json'
        if manifest.exists():assert json.loads(manifest.read_text())==contract
        else:L.atomic(contract,manifest,True)
        dst=directory/'analysis.json'
        if dst.exists():
            assert json.loads(dst.read_text())['contract_id']==cid
            print(json.dumps({'stage':'analysis_resume','update':update,'horizon':a.horizon}),flush=True);continue
        heads={'real':R.fit(real[:,:,CELLS],labels,fit,dev)}
        for arm in ('capacity','rate'):
            path=directory/f'pred_{arm}_posterior.pt'
            if path.exists():
                completed=torch.load(path,map_location='cpu',weights_only=False)['predictions'][:,0]
                heads[arm]=R.fit(completed[:,:,CELLS],labels,fit,dev)
        L.atomic({'contract_id':cid,'heads':heads},directory/'mob_heads.pt')
        predictions={'TRUE':real[:,None],'copy_root':panel['tokens'][:,15,None,None].float().expand(-1,1,a.horizon,-1,-1)}
        extra={}
        for path in paths:
            data=torch.load(path,map_location='cpu',weights_only=False)
            key=path.stem.removeprefix('pred_')
            predictions[key]=data['predictions']
            extra[key]=data
        groups={'1':[0]} if a.horizon==1 else {'1':[0],'2-4':[1,2,3],'5-8':[4,5,6,7],'9-16':list(range(8,16))}
        result={'contract_id':cid,'scope':contract['scope'],'root_count':200,'readout_TEST_episodes':50,
                'heads':{k:{s:h[s]for s in ('alpha','dev_mse','dev_mse_by_alpha','thresholds')}for k,h in heads.items()},
                'models':{},'paired_contrasts':{},'energy':{},'raw_per_root':{},'code_use':{},
                'sampled_recorded_branch_overlap_is_not_sample_correctness':True,
                'mean_latent_regression_is_diagnostic_not_distributional_score':True}
        if fan is not None:
            result['real_fan_energy_reference']={region:interval(energy(fan['tokens'][:,16:,cells].float(),
                                                                       fan['tokens'][:,:16,cells].float())[test].numpy(),weights)
                                               for region,cells in REGIONS.items()}
        for name,x in predictions.items():
            result['models'][name]={};result['raw_per_root'][name]={}
            mean=x.double().mean(1)
            err=(mean-real.double()).abs()
            for group,depths in groups.items():
                components={'MSE':err[:,depths].square().mean((1,2,3)),
                            'L1':err[:,depths].mean((1,2,3)),
                            'frame_max_squared':err[:,depths].square().flatten(2).max(-1).values.mean(1)}
                components['D']=components['MSE']+.1*components['L1']+.01*components['frame_max_squared']
                result['models'][name][group]={k:interval(v[test].numpy(),weights)for k,v in components.items()}
                result['raw_per_root'][name][group]={k:v.numpy().tolist()for k,v in components.items()}
                # A pooled averaged latent is scored above only as a diagnostic;
                # outcome readouts always operate on each sampled branch first.
                z=x[test][:,:,depths][:,:,:,CELLS].reshape(-1,192)
                chosen_heads={'original':None,'new_real':heads['real']}
                arm=name.split('_')[0]
                if arm in heads:chosen_heads['new_posterior']=heads[arm]
                reading={}
                for label,head in chosen_heads.items():
                    scores=(reader.mob(z) if head is None else R.head_scores(head,z)).float().view(50,x.shape[1],len(depths),len(CELLS),4)
                    threshold=reader.thr if head is None else head['thresholds']['dev_f1']
                    decoded=scores>torch.tensor(threshold)
                    expected=labels[test][:,depths,None].transpose(1,2).expand_as(decoded).bool()
                    reading[label]={mob:{'true':int(expected[...,j].sum()),'tp':int((decoded[...,j]&expected[...,j]).sum()),
                                         'fp':int((decoded[...,j]&~expected[...,j]).sum()),
                                         'mean_predicted_per_frame':float(decoded[...,j].sum(-1).float().mean())}
                                    for j,mob in enumerate(R.TYPES)}
                result['models'][name][group]['mob_readers']=reading
                branch_frames=x[test][:,:,depths]
                hud=reader.hud(branch_frames[:,:,:,63:].reshape(-1,18*192))[:,:4].view(50,x.shape[1],len(depths),4)
                result['models'][name][group]['original_HUD_mean_forecast_MSE']={counter:interval(
                    (hud.double().mean(1)[...,j]-hud_truth[test][:,depths,j].double()).square().mean(1).numpy(),weights)
                    for j,counter in enumerate(('health','food','drink','energy'))}
                tiles=reader.tile(branch_frames[:,:,:,CELLS].reshape(-1,192)).argmax(-1).view(50,x.shape[1],len(depths),len(CELLS))
                correct=tiles==tile_truth[test][:,depths][:,:,CELLS][:,None]
                result['models'][name][group]['original_tile_recorded_branch_agreement']={
                    region:interval(correct[:,:,:,[CELLS.index(c)for c in cells if c in CELLS]].float().mean((1,2,3)).numpy(),weights)
                    for region,cells in REGIONS.items() if region in ('map','near')}
            if a.horizon==1 and name!='TRUE':
                result['energy'][name]={}
                for region,cells in REGIONS.items():
                    score=energy(x[:,:,0,cells],fan['tokens'][:,:,cells].float())
                    result['energy'][name][region]=interval(score[test].numpy(),weights)
                    result['raw_per_root'][name]['energy_'+region]=score.tolist()
                counters=reader.hud(x[:,:,0,63:].reshape(-1,18*192))[:,:4].view(200,x.shape[1],4)
                result['energy'][name]['counter_joint_original']=interval(energy(counters,fan['hud'])[test].numpy(),weights)
                result['energy'][name]['counter_original']={counter:interval(energy(counters[...,j:j+1],fan['hud'][...,j:j+1])[test].numpy(),weights)
                                                          for j,counter in enumerate(('health','food','drink','energy'))}
                for headlabel,head in chosen_heads.items():
                    local=x[test][:,:,0,CELLS].reshape(-1,192)
                    sc=(reader.mob(local) if head is None else R.head_scores(head,local))
                    threshold=reader.thr if head is None else head['thresholds']['dev_f1']
                    probability=(sc.view(50,x.shape[1],len(CELLS),4)>torch.tensor(threshold)).float().mean(1)
                    observed=fan['layouts'][test][:,:,CELLS][:,:,:,[0,2,1,3]].float()
                    brier=(probability[:,None]-observed).square().mean((1,2))
                    result['energy'][name]['occupancy_'+headlabel]={mob:interval(brier[:,j].numpy(),weights)
                                                                  for j,mob in enumerate(R.TYPES)}
            if name in extra and extra[name]['codes'] is not None:
                codes=extra[name]['codes']
                entropy=[];used=[]
                for slot in range(6):
                    p=torch.bincount(codes[...,slot].flatten(),minlength=8).double();p=p/p.sum()
                    entropy.append(float(-(p[p>0]*p[p>0].log2()).sum()));used.append(int((p>0).sum()))
                rate=extra[name]['assignment_nll_bits']
                result['code_use'][name]={'marginal_entropy_bits_per_factor':entropy,'used_codes_per_factor':used,
                                          'marginal_entropy_sum_is_not_joint_entropy':True,
                                          'assignment_nll_bits_mean':float(rate.mean()) if rate is not None else None}
        control='control_control'
        if control in predictions:
            for name in predictions:
                if name in ('TRUE','copy_root',control):continue
                comparison={}
                for group in groups:
                    comparison[group]={}
                    for metric in ('MSE','L1','frame_max_squared','D'):
                        delta=np.array(result['raw_per_root'][name][group][metric])-np.array(result['raw_per_root'][control][group][metric])
                        comparison[group][metric]=interval(delta[test],weights)
                if a.horizon==1:
                    comparison['energy']={}
                    for region in REGIONS:
                        delta=np.array(result['raw_per_root'][name]['energy_'+region])-np.array(result['raw_per_root'][control]['energy_'+region])
                        comparison['energy'][region]=interval(delta[test],weights)
                result['paired_contrasts'][name+'_minus_control']=comparison
        L.atomic(result,dst,True)
        print(json.dumps({'stage':'analysis_done','update':update,'horizon':a.horizon,'output':str(dst)}),flush=True)


if __name__=='__main__':main()
