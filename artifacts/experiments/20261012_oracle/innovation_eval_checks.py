"""Actual-model protocol checks, plus CPU scoring and resume controls."""
import json
import sys
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).parent))
import innovation_model as I
import innovation_train as T
import innovation_eval as E
import innovation_analysis as A
import loss_intervention as L
import oracle_eval as OE
from d4mj.config import config_from_dict


def main():
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    config=config_from_dict(torch.load(L.TW.S.CHECKPOINT,map_location='cpu',weights_only=False)['config'])
    panel=torch.load(L.OUT/'panel.pt',map_location='cpu',weights_only=False)
    frames=panel['tokens'][:2,1:16].cuda().float();actions=panel['actions'][:2,1:16].cuda()
    world=L.TW.TWorld('corrt',None,'fmamba','all',0,False,16).cuda().eval()
    saved=torch.load(T.OUT/'capacity_u0000.pt',map_location='cpu',weights_only=False)
    world.load_state_dict(saved['world'])
    channel=I.Innovation().cuda().eval();channel.load_state_dict(saved['channel'])
    baseline,_,_=E.predict(world,None,frames,actions,config,'control')
    factual=OE.step(world,None,frames,actions,None,torch.device('cuda'),config)
    assert torch.equal(baseline.cpu(),factual),'wrong frame/action position'
    belief=I.native_belief(world,frames,actions,config)
    with torch.no_grad():past=E.historical(channel,belief,frames)
    differences={}
    for method in E.METHODS:
        kwargs={'successor':panel['tokens'][:2,16].cuda().float()} if method=='posterior'else {}
        if method=='shuffle_current':kwargs['shuffled_codes']=past[:,0].roll(1,0)
        value,_,_=E.predict(world,channel,frames,actions,config,method,past,
                            generator=torch.Generator(device='cuda').manual_seed(91),**kwargs)
        differences[method]=float((value-baseline).abs().max())
        assert differences[method]==0.
    rejected=False
    try:E.predict(world,channel,frames,actions,config,'prior_mode',past,successor=frames[:,-1])
    except AssertionError:rejected=True
    assert rejected
    # With observed historical codes supplied, no prior call may invoke the
    # future-conditioned posterior. This checks the executed serving path.
    original=channel.posterior.forward
    def forbidden(*args,**kwargs):raise AssertionError('prior called the future-conditioned posterior')
    channel.posterior.forward=forbidden
    E.predict(world,channel,frames,actions,config,'prior_sample',past,
              generator=torch.Generator(device='cuda').manual_seed(91))
    channel.posterior.forward=original
    fixture=T.OUT/'analysis_fixture/eval_u0000_h01/analysis.json'
    data=json.loads(fixture.read_text())
    values=[v for contrast in data['paired_contrasts'].values()for group in contrast.values()for v in group.values()]
    fixture_max=max(max(abs(v['mean']),*(abs(c)for c in v['ci95']))for v in values)
    assert fixture_max<1e-14, fixture_max
    fan=torch.load(T.OUT/'real_futures/fan.pt',map_location='cpu',weights_only=False)
    assert fan['roots']==panel['roots'] and fan['replay_maxdiff']==fan['encoder_maxabs']==0
    proof={'initial_method_outputs_maxabs':differences,'true_evaluator_position_exact':True,
           'prior_rejects_future':rejected,'prior_future_posterior_calls':0,
           'identical_fixture_paired_estimates_and_intervals_maxabs':fixture_max,
           'real_fan_roots':len(fan['roots']),'independent_keys_per_root':fan['tokens'].shape[1],
           'real_fan_recorded_replay_pixel_maxdiff':fan['replay_maxdiff'],'encoder_parity_maxabs':fan['encoder_maxabs'],
           'sources':{str(Path(x).relative_to(L.ROOT)):L.sha(x)for x in (__file__,I.__file__,T.__file__,E.__file__,A.__file__)}}
    L.atomic(proof,T.OUT/'evaluation_validation.json',True)
    print(json.dumps(proof),flush=True)


if __name__=='__main__':main()
