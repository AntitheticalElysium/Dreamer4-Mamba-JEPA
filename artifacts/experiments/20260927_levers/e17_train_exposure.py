"""E17 exact long-teacher exposure census, declared Oct7 before execution.

Reconstruct6000x40 sampler draws from seed11 and the literal long TRAIN branch;
require reconstructed end generator state to equal all four saved training states.
Verify labels/episode ledger against completed pool manifest. Count per-position
deaths, ordinary living>=2 drops and unchanged targets, with unique episode/
transition counts, not repeated exposures called independent. No GPU/fitting.
This establishes exposure and opportunity for a position shortcut, not the
historical cause of failure or a trained repair. Source/input-bound atomic report.
"""
import ast
import json
from pathlib import Path
import numpy as np
import torch
import h16_resume as R
import tworld as TW

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]


def main():
    torch.set_num_threads(2)
    pool=TW.POOLS['rawlong'];manifest=json.loads((pool/'manifest.json').read_text())
    ledger=HERE.parent/'20260929_mamba_integration/long_ledger.jsonl'
    assert R.file_hash(ledger)==manifest['contract']['ledger_sha256']
    assert R.file_hash(pool/'labels.pt')==manifest['labels_sha256']
    pp=torch.load(pool/'labels.pt',mmap=True,weights_only=False,map_location='cpu')
    ids=[json.loads(s) for s in ledger.read_text().splitlines()]
    assert len(ids)==len(pp['terminal'])==19789
    assert torch.equal(pp['terminal'],torch.tensor([r['terminal'] for r in ids]))
    ep_names=sorted(set(r['episode_id'] for r in ids));ep_index={s:i for i,s in enumerate(ep_names)}
    ep=torch.tensor([ep_index[r['episode_id']] for r in ids]);starts=torch.tensor([r['start'] for r in ids])
    original=HERE.parent/'20261004_resume_audit/TRAIN_SOURCE_BEFORE_RESUME.py'
    def branch(path):
        tree=ast.parse(Path(path).read_text())
        f=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='train')
        loop=next(n for n in f.body if isinstance(n,ast.For))
        return ast.dump(loop.body[0],include_attributes=False)
    assert branch(original)==branch(TW.__file__),'Long sampler branch drifted'
    states={};input_files=[pool/'manifest.json',pool/'labels.pt',ledger,original]
    for bb in ('full','fmamba'):
        for seed in (7,8):
            name=f'corrt_rawlong_teacher_s{seed}'+('_fmamba' if bb=='fmamba' else '')+'_L16b40_from36000'
            f=TW.OUT/'state'/f'{name}.state.pt';s=torch.load(f,mmap=True,weights_only=False,map_location='cpu')
            assert s['update']==6000;states[name]=s['order'];input_files.append(f)
    spec={'scope':__doc__,'sources':{f:R.file_hash(f) for f in (__file__,R.__file__,TW.__file__)},
          'inputs':{str(f):R.file_hash(f) for f in input_files},'terminal_share':TW.TERMINAL_SHARE,
          'sampler_branch_equal_before_resume':True,'runtime':str(torch.__version__)}
    store=R.Store(HERE/'evals/resume/e17_train_exposure',spec)
    with store.lock():
        done=store.load('result')
        if done is not None:print(json.dumps(done),flush=True);return
        main_rows=torch.where(~pp['terminal'])[0];term_rows=torch.where(pp['terminal'])[0]
        g=torch.Generator().manual_seed(11);rr=[];off=[]
        for _ in range(6000):
            term=torch.rand(40,generator=g)<TW.TERMINAL_SHARE
            r=torch.where(term,term_rows[torch.randint(len(term_rows),(40,),generator=g)],
                          main_rows[torch.randint(len(main_rows),(40,),generator=g)])
            t0=torch.where(term,64-16,torch.randint(0,65-16,(40,),generator=g))
            rr.append(r);off.append(t0)
        assert all(torch.equal(g.get_state(),v) for v in states.values())
        rows=torch.stack(rr).flatten();t0=torch.stack(off).flatten()
        cols=t0[:,None]+torch.arange(1,16)[None,:]
        reward=pp['reward_led'][rows[:,None],cols].double()
        achievements=torch.ceil(reward-.1-1e-6).clamp_min(0)
        dh=torch.round((reward-achievements)/.1).long()
        assert bool(((reward-achievements-.1*dh).abs()<1e-5).all())
        alive=pp['alive'][rows[:,None],cols];prev=pp['alive'][rows[:,None],cols-1]
        death=prev&~alive;ordinary=prev&alive&(dh<=-2);unchanged=prev&alive&(dh==0)
        actual_time=starts[rows,None]+cols
        base=int(actual_time.max())+1;transition_key=ep[rows,None]*base+actual_time
        classes={}
        for name,mask in (('death',death),('living_drop_at_least2',ordinary),('living_unchanged',unchanged)):
            per=mask.sum(0)
            classes[name]={'exposures':int(mask.sum()),'per_target_position_1_to15':per.tolist(),
                          'rates_by_position':(per/len(rows)).tolist(),
                          'distinct_episode_transitions':int(torch.unique(transition_key[mask]).numel())}
        result={'scope':__doc__,'contract':spec,'windows':len(rows),'targets':cols.numel(),
            'same_reconstructed_sampler_matches_all4_world_rng_states':True,
            'classes':classes,'death_last_position_fraction':float(death[:,-1].sum()/death.sum()),
            'terminal_windows_drawn':int(pp['terminal'][rows].sum())}
        store.save('ledger',{'rows':rows.view(6000,40),'offsets':t0.view(6000,40)},1)
        store.save('result',result,1);R.atomic_json(HERE/'evals/e17_train_exposure.json',result)
        print(json.dumps({k:v for k,v in result.items() if k not in ('scope','contract')}),flush=True)


if __name__=='__main__':main()
