"""Matched teacher/generated-prefix continuation with known mob outcomes.

The main teacher term is unchanged. A matched additional endpoint is scored at
the evaluation position (15 input frames, last output), with the same target,
actions and oracle layouts in each arm. Only the additional endpoint's input
prefix differs. Prefix construction is detached; final completion is trainable.
All data/model/optimizer/random-stream state is bound and atomically resumable.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import sys
import time

os.environ.setdefault('TRITON_F32_DEFAULT', 'ieee')
if os.environ['TRITON_F32_DEFAULT'] != 'ieee':
    raise RuntimeError('this experiment requires TRITON_F32_DEFAULT=ieee')

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).parent))
import loss_intervention as LI
OW, TW = LI.OW, LI.TW
from d4mj.config import config_from_dict
from d4mj.train import _phase_lr, autocast_context, optimizer_step, phase_optimizer

OUT = ROOT / 'artifacts/eda/oracle_v1'
ARMS = ('teacher', 'prefix')
INIT = OUT / 'loss_intervention/D_u3000.pt'
CONTEXT = 15
EXPOSURES = (1, 2, 4, 8, 15)


def conditioned(world, cond, frames, actions, layouts, config):
    """Layouts at each input position are for its immediate successor."""
    assert frames.shape[:2] == actions.shape == layouts.shape[:2]
    assert frames.shape[1] == CONTEXT, 'endpoint context must match evaluator'
    with autocast_context(config):
        world.delta = cond(layouts, CONTEXT)
        try:
            predicted = world(frames.float(), actions)[0]
        finally:
            world.delta = None
    return predicted[:, -1].float()


def window(stream, starts, length):
    """Gather per-row chronological windows without mixing episode rows."""
    index = starts[:, None] + torch.arange(length, device=starts.device)[None]
    assert int(index.min()) >= 0 and int(index.max()) < stream.shape[1], 'window leaves recorded support'
    return stream[torch.arange(len(stream), device=starts.device)[:, None], index]


def endpoint(world, cond, stream, actions, layouts, targets, exposure, generated, config):
    """Predict each targett with exactly15frames ending at t-1.

    Generated arm observes15frames [t-p-15:t-p), then recursively constructs
    p detached outputs [t-p:t). Every step uses the same15frame sliding window,
    matching recorded outgoing actions and full next-screen oracle outcomes.
    Teacher uses real [t-15:t). Targett and exposurep are independent draws.
    """
    assert 0 <= exposure <= CONTEXT
    assert stream.shape[1] == 64 and actions.shape[1] == 63 and layouts.shape[1] == 64
    assert targets.shape == (len(stream),)
    final_start = targets - CONTEXT
    if generated:
        start = targets - exposure - CONTEXT
        initial = window(stream, start, CONTEXT)
        frames = [initial[:, j] for j in range(CONTEXT)]
        with torch.no_grad():
            for k in range(exposure):
                nxt = conditioned(world, cond, torch.stack(frames[-CONTEXT:], 1),
                                  window(actions, start+k, CONTEXT),
                                  window(layouts, start+k+1, CONTEXT), config)
                assert not nxt.requires_grad, 'prefix construction leaked gradient'
                frames.append(nxt.detach())
        inputs = torch.stack(frames[-CONTEXT:], 1).detach()
    else:
        inputs = window(stream, final_start, CONTEXT)
    predicted = conditioned(world, cond, inputs,
                            window(actions, final_start, CONTEXT),
                            window(layouts, final_start+1, CONTEXT), config)
    target = stream[torch.arange(len(stream), device=targets.device), targets]
    return predicted, target, inputs


def main_teacher(world, cond, frames, actions, next_layouts, config):
    with autocast_context(config):
        world.delta = cond(next_layouts, 16)
        try:
            predicted = world(frames.float(), F.pad(actions, (0, 1)))[0][:, :15]
        finally:
            world.delta = None
    return LI.components(predicted, frames[:, 1:])


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--updates', type=int, default=3000)
    p.add_argument('--windows', type=int, default=16)
    p.add_argument('--save-every', type=int, default=100)
    p.add_argument('--stop-after', type=int)
    p.add_argument('--prefix', default='generated_prefix')
    p.add_argument('--endpoint-weight', type=float, default=.5)
    a = p.parse_args()
    assert a.updates > 0 and a.windows > 0 and a.save_every > 0
    assert 0 <= a.endpoint_weight <= 1
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(7); torch.cuda.manual_seed_all(7)
    started = time.time()
    config = config_from_dict(torch.load(TW.S.CHECKPOINT, weights_only=False, map_location='cpu')['config'])
    sources = [Path(__file__), Path(LI.__file__), Path(OW.__file__), Path(TW.__file__),
               ROOT/'d4mj/train.py', ROOT/'d4mj/mamba_recurrence.py', Path(TW.S.__file__),
               Path(TW.S.CHECKPOINT), INIT, TW.POOLS['rawlong']/'labels.pt',
               TW.POOLS['rawlong']/'manifest.json', OUT/'layouts.npy', OUT/'layouts_valid.npy']
    contract = {
        'format': 'oracle_prefix_intervention_v1',
        'sources_inputs': {str(f.relative_to(ROOT)): LI.sha(f) for f in sources},
        'updates': a.updates, 'windows': a.windows, 'arms': ARMS,
        'batch_seed': 11, 'exposure_seed': 20261012, 'init_seed': 7,
        'precision': 'BF16 model autocast, IEEE Triton, TF32 disabled; FP32 detached prefixes',
        'optimizer': 'same fresh phase AdamW and warmup in both arms',
        'loss': 'D = mean squared error +0.1 mean absolute error +0.01 mean frame max squared error',
        'objective': {'main_teacher': 1-a.endpoint_weight, 'matched_endpoint': a.endpoint_weight},
        'teacher': 'old16frame support-v2 sampler unchanged, all15targets',
        'endpoint': {'main_target': 'uniform frame30..63', 'terminal_target':63,
                     'exposure':list(EXPOSURES), 'exposure_probability':'uniform independent of target/terminal label',
                     'context':15,'gradient_output_position':14},
        'prefix': 'teacher:real[target-15:target]; prefix:15realframes then p detached recursive outputs',
        'target': 'identical endpoint frame, actions, oracle outcomes and source row in both arms',
        'oracle_scope': 'full true next-screen mob layout; privileged branch-matched completion, not innovation prediction',
        'trainable': 'world and OracleCond in both arms; detached construction never propagates gradients',
        'question': 'effect of generated-context exposure, conditional on known outcomes, at shared completed D checkpoint',
    }
    contract_id = hashlib.sha256(json.dumps(contract, sort_keys=True).encode()).hexdigest()
    directory = OUT / a.prefix
    contract_path, resume_path = directory/'contract.json', directory/'state.pt'
    if contract_path.exists():
        assert json.loads(contract_path.read_text()) == json.loads(json.dumps(contract)), 'contract drift'
    else:
        LI.atomic(contract, contract_path, True)
    pool = torch.load(TW.POOLS['rawlong']/'labels.pt', weights_only=False, map_location='cpu', mmap=True)
    manifest = json.loads((TW.POOLS['rawlong']/'manifest.json').read_text())
    assert LI.sha(TW.POOLS['rawlong']/'labels.pt') == manifest['labels_sha256']
    token_path = TW.POOLS['rawlong']/'tokens.f16'
    token_stat = token_path.stat()
    fingerprint = {'sha256':manifest['tokens_sha256'], 'size':token_stat.st_size, 'mtime_ns':token_stat.st_mtime_ns}
    token_proof = directory/'token_verification.json'
    if token_proof.exists():
        assert json.loads(token_proof.read_text()) == fingerprint, 'training-token file changed'
    else:
        previous = OUT/'loss_intervention/token_verification.json'
        if previous.exists() and json.loads(previous.read_text()) == fingerprint:
            LI.log(stage='tokens_previous_full_hash_verified', fingerprint=fingerprint)
        else:
            LI.log(stage='tokens_hash_start', size=token_stat.st_size)
            assert LI.sha(token_path) == fingerprint['sha256'], 'pool token hash mismatch'
        LI.atomic(fingerprint, token_proof, True)
    n = len(pool['terminal'])
    tokens = np.memmap(token_path, dtype=np.float16, mode='r', shape=(n,64,81,192))
    layouts = np.load(OUT/'layouts.npy', mmap_mode='r')
    valid = torch.from_numpy(np.load(OUT/'layouts_valid.npy'))
    main_rows = torch.where(~pool['terminal'] & valid)[0]
    term_rows = torch.where(pool['terminal'] & valid)[0]
    LI.log(stage='rows', main=len(main_rows), terminal=len(term_rows), contract_id=contract_id)
    parent = torch.load(INIT, weights_only=False, map_location='cpu')
    assert parent['update'] == 3000 and parent['args']['arm'] == 'oracle'
    worlds, conds, opts, params = {}, {}, {}, {}
    for arm in ARMS:
        world = TW.TWorld('corrt', None, 'fmamba', 'all', 0, False, 16)
        world.load_state_dict(parent['world'])
        oc = OW.OracleCond(); oc.load_state_dict(parent['oracle'])
        worlds[arm], conds[arm] = world.to('cuda').train(), oc.to('cuda').train()
        opts[arm] = phase_optimizer([worlds[arm], conds[arm]], config)
        params[arm] = [q for group in opts[arm].param_groups for q in group['params']]
    assert LI.restored_equal(LI.cpu_state(worlds['teacher']), LI.cpu_state(worlds['prefix'])), 'world init differs'
    assert LI.restored_equal(LI.cpu_state(conds['teacher']), LI.cpu_state(conds['prefix'])), 'oracle init differs'
    order = torch.Generator().manual_seed(11)
    exposure = torch.Generator().manual_seed(20261012)
    history = []
    summaries = {arm:{'norm_sum':0.,'clip_count':0,'steps':0,'compute_seconds_sum':0.,
                     'teacher_targets':0,'endpoint_targets':0} for arm in ARMS}
    sampling = {'terminal_rows':0, 'exposure_counts':{str(p):0 for p in EXPOSURES},
                'terminal_exposure_counts':{str(p):0 for p in EXPOSURES},'main_targets':[0]*64}
    update0, ledger_chain = 0, 'initial'
    if resume_path.exists():
        st = torch.load(resume_path, weights_only=False, map_location='cpu')
        assert st['contract_id'] == contract_id, 'resume source/input drift'
        for arm in ARMS:
            worlds[arm].load_state_dict(st['worlds'][arm]); conds[arm].load_state_dict(st['oracles'][arm])
            opts[arm].load_state_dict(st['optimizers'][arm])
            assert LI.restored_equal(LI.cpu_state(worlds[arm]), st['worlds'][arm]), 'world restoration mismatch'
            assert LI.restored_equal(LI.cpu_state(conds[arm]), st['oracles'][arm]), 'oracle restoration mismatch'
            assert LI.restored_equal(opts[arm].state_dict(), st['optimizers'][arm]), 'optimizer restoration mismatch'
        order.set_state(st['order']); exposure.set_state(st['exposure'])
        torch.set_rng_state(st['torch_rng']); torch.cuda.set_rng_state_all(st['cuda_rng'])
        update0, history, summaries = st['update'], st['history'], st['summaries']
        sampling, ledger_chain = st['sampling'], st['ledger_chain']
        LI.log(stage='resume', update=update0, model_optimizer_restored_exactly=True)

    def checkpoint(update):
        LI.atomic({'contract_id':contract_id,'contract':contract,'update':update,
                   'worlds':{arm:LI.cpu_state(worlds[arm]) for arm in ARMS},
                   'oracles':{arm:LI.cpu_state(conds[arm]) for arm in ARMS},
                   'optimizers':{arm:copy.deepcopy(opts[arm].state_dict()) for arm in ARMS},
                   'order':order.get_state(),'exposure':exposure.get_state(),'torch_rng':torch.get_rng_state(),
                   'cuda_rng':torch.cuda.get_rng_state_all(),'ledger_chain':ledger_chain,
                   'history':history,'summaries':summaries,'sampling':sampling}, resume_path)

    def snapshots(update):
        for arm in ARMS:
            LI.atomic({'name':f'{a.prefix}_{arm}_u{update}', 'args':{'arm':'oracle','recon':'D','prefix_arm':arm},
                       'world':LI.cpu_state(worlds[arm]),'oracle':LI.cpu_state(conds[arm]),
                       'contract':contract,'contract_id':contract_id,'update':update,'ledger_chain':ledger_chain,
                       'history':history,'summaries':summaries,'sampling':sampling}, directory/f'{arm}_u{update:04d}.pt')
    if update0 == 0:
        snapshots(0)
    milestones = {500,1500,a.updates}
    for update in range(update0,a.updates):
        term = torch.rand(a.windows,generator=order) < TW.TERMINAL_SHARE
        rows = torch.where(term,term_rows[torch.randint(len(term_rows),(a.windows,),generator=order)],
                           main_rows[torch.randint(len(main_rows),(a.windows,),generator=order)])
        starts = torch.where(term,64-16,torch.randint(0,49,(a.windows,),generator=order))
        targets = torch.where(term,63,torch.randint(30,64,(a.windows,),generator=exposure))
        pfx = EXPOSURES[int(torch.randint(len(EXPOSURES),(),generator=exposure))]
        raw = rows.numpy().tobytes()+starts.numpy().tobytes()+targets.numpy().tobytes()+pfx.to_bytes(4,'little')
        ledger_chain = hashlib.sha256((b'' if ledger_chain=='initial' else bytes.fromhex(ledger_chain))+raw).hexdigest()
        sampling['terminal_rows'] += int(term.sum()); sampling['exposure_counts'][str(pfx)] += a.windows
        sampling['terminal_exposure_counts'][str(pfx)] += int(term.sum())
        for t in targets[~term].tolist(): sampling['main_targets'][t] += 1
        s = torch.stack([torch.from_numpy(np.array(tokens[int(i),int(j):int(j)+16])) for i,j in zip(rows,starts)]).to('cuda').float()
        ac = torch.stack([pool['actions'][int(i),int(j):int(j)+15] for i,j in zip(rows,starts)]).to('cuda')
        nl = torch.stack([torch.from_numpy(np.array(layouts[int(i),int(j)+1:int(j)+16])) for i,j in zip(rows,starts)]).to('cuda')
        aux = torch.from_numpy(np.array(tokens[rows.numpy()])).to('cuda').float()
        aux_actions = pool['actions'][rows].to('cuda')
        aux_layouts = torch.from_numpy(np.array(layouts[rows.numpy()])).to('cuda')
        targets = targets.to('cuda')
        step_rows = {}
        cpu_rng,gpu_rng = torch.get_rng_state(),torch.cuda.get_rng_state_all()
        for arm in ARMS:
            arm_started = time.perf_counter()
            torch.set_rng_state(cpu_rng); torch.cuda.set_rng_state_all(gpu_rng)
            world,oc = worlds[arm],conds[arm]
            main_c = main_teacher(world,oc,s,ac,nl,config)
            pred,target,inputs = endpoint(world,oc,aux,aux_actions,aux_layouts,targets,pfx,arm=='prefix',config)
            end_c = LI.components(pred[:,None],target[:,None])
            main_loss,end_loss = LI.objective(main_c,'D'),LI.objective(end_c,'D')
            loss = (1-a.endpoint_weight)*main_loss+a.endpoint_weight*end_loss
            norm = float(optimizer_step(opts[arm],loss,params[arm],learning_rate=_phase_lr(config,update),
                                       grad_clip=config.agent.grad_clip,strict=True,zero_grad=True))
            step_rows[arm] = {'loss':float(loss.detach()),'teacher':float(main_loss.detach()),'endpoint':float(end_loss.detach()),
                              'teacher_components':{k:float(v.detach()) for k,v in main_c.items()},
                              'endpoint_components':{k:float(v.detach()) for k,v in end_c.items()},
                              'prefix_mse':float((inputs-window(aux,targets-CONTEXT,CONTEXT)).square().mean()),
                              'gradient_norm':norm,'compute_seconds':time.perf_counter()-arm_started}
            summaries[arm]['norm_sum'] += norm; summaries[arm]['clip_count'] += int(norm>config.agent.grad_clip); summaries[arm]['steps'] += 1
            summaries[arm]['compute_seconds_sum'] += step_rows[arm]['compute_seconds']
            summaries[arm]['teacher_targets'] += a.windows*15
            summaries[arm]['endpoint_targets'] += a.windows
            del main_c,end_c,main_loss,end_loss,loss,pred,target,inputs
        if (update+1)%100==0 or update==0:
            row = {'update':update+1,'exposure':pfx,'arms':step_rows,'seconds':round(time.time()-started,1),
                   'gpu_peak_gb':round(torch.cuda.max_memory_allocated()/2**30,3),'ledger_chain':ledger_chain}
            history.append(row); LI.log(stage='train',**row)
        if (update+1)%a.save_every==0 or update+1 in milestones or update+1==a.stop_after:
            checkpoint(update+1)
            if update+1 in milestones:
                snapshots(update+1)
        if update+1==a.stop_after:
            LI.log(stage='interruption_check',update=update+1,resumable=True); return
    final = {'contract_id':contract_id,'update':a.updates,'summaries':summaries,'sampling':sampling,
             'history':history,'ledger_chain':ledger_chain}
    LI.atomic(final,directory/'training.json',True)
    LI.log(stage='done',**{k:v for k,v in final.items() if k!='history'})


if __name__ == '__main__':
    main()
