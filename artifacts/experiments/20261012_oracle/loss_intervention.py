"""Matched loss intervention from one completed oracle-D world.

D, L1 and D without its maximum-element term share initial world/OracleCond,
data draws, fresh optimizers and schedule. This tests adaptation of this trained
world, not from-scratch superiority. All optimizer/RNG/order states are resumable.
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

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'artifacts/experiments/20260927_levers'))
sys.path.insert(0, str(Path(__file__).parent))
import oracle_world as OW
import tworld as TW
from d4mj.config import config_from_dict
from d4mj.train import _phase_lr, autocast_context, optimizer_step, phase_optimizer

OUT = ROOT / 'artifacts/eda/oracle_v1'
ARMS = ('D', 'L1', 'no_max')
INIT = OUT / 'worlds/oracle_deltairis_from_corrt_rawlong_teacher_s7_fmamba_di_L16b40_from36000_u3000_b16.pt'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(8 << 20), b''):
            h.update(b)
    return h.hexdigest()


def atomic(obj, path, json_format=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    with tmp.open('w' if json_format else 'wb') as f:
        json.dump(obj, f, indent=2) if json_format else torch.save(obj, f)
        f.flush(); os.fsync(f.fileno())
    os.replace(tmp, path)


def log(**row):
    print(json.dumps(row), flush=True)


def components(pred, target):
    err = (pred.float() - target.float()).abs()
    return {'mse': err.square().mean(), 'mae': err.mean(),
            'max': err.square().flatten(2).max(-1).values.mean()}


def objective(c, arm):
    if arm == 'L1': return c['mae']
    return c['mse'] + .1 * c['mae'] + (.01 * c['max'] if arm == 'D' else 0)


def cpu_state(module):
    return {k: v.detach().cpu().clone() for k, v in module.state_dict().items()}


def restored_equal(left, right):
    if isinstance(left, torch.Tensor):
        return isinstance(right, torch.Tensor) and torch.equal(left.detach().cpu(), right.detach().cpu())
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(restored_equal(left[k], right[k]) for k in left)
    if isinstance(left, (list, tuple)):
        return len(left) == len(right) and all(restored_equal(l, r) for l, r in zip(left, right))
    return left == right


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--updates', type=int, default=3000)
    p.add_argument('--windows', type=int, default=16)
    p.add_argument('--save-every', type=int, default=250)
    p.add_argument('--stop-after', type=int)
    p.add_argument('--prefix', default='loss_intervention')
    a = p.parse_args()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(7); torch.cuda.manual_seed_all(7)
    started = time.time()
    config = config_from_dict(torch.load(TW.S.CHECKPOINT, weights_only=False, map_location='cpu')['config'])
    files = [Path(__file__), Path(OW.__file__), Path(TW.__file__), ROOT/'d4mj/train.py',
             ROOT/'d4mj/mamba_recurrence.py', Path(TW.S.__file__), INIT,
             TW.POOLS['rawlong']/'labels.pt', TW.POOLS['rawlong']/'manifest.json',
             OUT/'layouts.npy', OUT/'layouts_valid.npy']
    contract = {'format': 'oracle_loss_intervention_v1', 'sources_inputs': {str(f.relative_to(ROOT)): sha(f) for f in files},
                'updates': a.updates, 'windows': a.windows, 'arms': ARMS, 'batch_seed': 11, 'init_seed': 7,
                'precision': 'BF16 model autocast, IEEE Triton, TF32 disabled',
                'optimizer': 'same fresh phase AdamW and warmup in every arm',
                'questions': 'causal continuation effect of loss at this shared trained oracle checkpoint'}
    contract_id = hashlib.sha256(json.dumps(contract, sort_keys=True).encode()).hexdigest()
    directory = OUT / a.prefix
    contract_path, resume_path = directory/'contract.json', directory/'state.pt'
    if contract_path.exists():
        assert json.loads(contract_path.read_text()) == json.loads(json.dumps(contract)), 'contract drift'
    else: atomic(contract, contract_path, True)
    pool = torch.load(TW.POOLS['rawlong']/'labels.pt', weights_only=False, map_location='cpu', mmap=True)
    manifest = json.loads((TW.POOLS['rawlong']/'manifest.json').read_text())
    assert sha(TW.POOLS['rawlong']/'labels.pt') == manifest['labels_sha256']
    token_path = TW.POOLS['rawlong']/'tokens.f16'
    # The pool already records a full hash; verify it once and seal file metadata
    # into this run. Every resume checks the verified file's size/mtime as well.
    token_proof = directory/'token_verification.json'
    token_stat = token_path.stat()
    fingerprint = {'sha256': manifest['tokens_sha256'], 'size': token_stat.st_size, 'mtime_ns': token_stat.st_mtime_ns}
    if token_proof.exists():
        assert json.loads(token_proof.read_text()) == fingerprint, 'training-token file changed'
    else:
        log(stage='tokens_hash_start', size=token_stat.st_size)
        assert sha(token_path) == fingerprint['sha256'], 'pool token hash mismatch'
        atomic(fingerprint, token_proof, True)
    n = len(pool['terminal'])
    tok = np.memmap(token_path, dtype=np.float16, mode='r', shape=(n,64,81,192))
    layout = np.load(OUT/'layouts.npy', mmap_mode='r')
    valid = torch.from_numpy(np.load(OUT/'layouts_valid.npy'))
    main_rows = torch.where(~pool['terminal'] & valid)[0]
    term_rows = torch.where(pool['terminal'] & valid)[0]
    log(stage='rows', main=len(main_rows), terminal=len(term_rows), contract_id=contract_id)
    parent = torch.load(INIT, weights_only=False, map_location='cpu')
    worlds, conds, opts, params = {}, {}, {}, {}
    for arm in ARMS:
        world = TW.TWorld('corrt', None, 'fmamba', 'all', 0, False, 16)
        world.load_state_dict(parent['world'])
        oc = OW.OracleCond(); oc.load_state_dict(parent['oracle'])
        worlds[arm], conds[arm] = world.to('cuda').train(), oc.to('cuda').train()
        opts[arm] = phase_optimizer([worlds[arm], conds[arm]], config)
        params[arm] = [v for group in opts[arm].param_groups for v in group['params']]
    assert all(torch.equal(v, worlds['L1'].state_dict()[k]) and torch.equal(v, worlds['no_max'].state_dict()[k])
               for k,v in worlds['D'].state_dict().items()), 'world initial weights differ'
    assert all(torch.equal(v, conds['L1'].state_dict()[k]) and torch.equal(v, conds['no_max'].state_dict()[k])
               for k,v in conds['D'].state_dict().items()), 'oracle initial weights differ'
    order = torch.Generator().manual_seed(11)
    history = []; summaries = {arm:{'norm_sum':0.,'clip_count':0,'steps':0} for arm in ARMS}
    update0 = 0
    if resume_path.exists():
        state = torch.load(resume_path, weights_only=False, map_location='cpu')
        assert state['contract_id'] == contract_id, 'resume source/input drift'
        for arm in ARMS:
            worlds[arm].load_state_dict(state['worlds'][arm]); conds[arm].load_state_dict(state['oracles'][arm])
            opts[arm].load_state_dict(state['optimizers'][arm])
            assert restored_equal(cpu_state(worlds[arm]), state['worlds'][arm]), 'model restoration mismatch'
            assert restored_equal(cpu_state(conds[arm]), state['oracles'][arm]), 'oracle restoration mismatch'
            assert restored_equal(opts[arm].state_dict(), state['optimizers'][arm]), 'optimizer restoration mismatch'
        order.set_state(state['order']); torch.set_rng_state(state['torch_rng'])
        torch.cuda.set_rng_state_all(state['cuda_rng'])
        update0, history, summaries = state['update'], state['history'], state['summaries']
        # Preserve an incremental digest as a chain so it can resume exactly.
        ledger_chain = state['ledger_chain']
        log(stage='resume', update=update0, model_optimizer_restored_exactly=True)
    else: ledger_chain = 'initial'

    def checkpoint(update):
        state = {'contract_id':contract_id, 'contract':contract, 'update':update,
                 'worlds':{arm:cpu_state(worlds[arm]) for arm in ARMS},
                 'oracles':{arm:cpu_state(conds[arm]) for arm in ARMS},
                 'optimizers':{arm:copy.deepcopy(opts[arm].state_dict()) for arm in ARMS},
                 'order':order.get_state(), 'torch_rng':torch.get_rng_state(),
                 'cuda_rng':torch.cuda.get_rng_state_all(), 'ledger_chain':ledger_chain,
                 'history':history, 'summaries':summaries}
        atomic(state, resume_path)
        return state

    def snapshots(update, state=None):
        for arm in ARMS:
            atomic({'name':f'{a.prefix}_{arm}_u{update}', 'args':{'arm':'oracle','recon':arm},
                    'world':cpu_state(worlds[arm]), 'oracle':cpu_state(conds[arm]),
                    'contract':contract,'contract_id':contract_id,'update':update,'ledger_chain':ledger_chain,
                    'history':history,'summaries':summaries}, directory/f'{arm}_u{update:04d}.pt')
    if update0 == 0: snapshots(0)
    milestones = {500, 1500, a.updates}
    for update in range(update0, a.updates):
        term = torch.rand(a.windows, generator=order) < TW.TERMINAL_SHARE
        rows = torch.where(term, term_rows[torch.randint(len(term_rows),(a.windows,),generator=order)],
                           main_rows[torch.randint(len(main_rows),(a.windows,),generator=order)])
        starts = torch.where(term,64-16,torch.randint(0,49,(a.windows,),generator=order))
        ledger_chain = hashlib.sha256(bytes.fromhex(ledger_chain) + rows.numpy().tobytes() + starts.numpy().tobytes()).hexdigest() \
            if ledger_chain != 'initial' else hashlib.sha256(rows.numpy().tobytes()+starts.numpy().tobytes()).hexdigest()
        s = torch.stack([torch.from_numpy(np.array(tok[int(i),int(j):int(j)+16])) for i,j in zip(rows,starts)]).to('cuda').float()
        ac = torch.stack([pool['actions'][int(i),int(j):int(j)+15] for i,j in zip(rows,starts)]).to('cuda')
        nl = torch.stack([torch.from_numpy(np.array(layout[int(i),int(j)+1:int(j)+16])) for i,j in zip(rows,starts)]).to('cuda')
        if update == 0:
            # Same predicted tensors demonstrate objective implementation parity.
            with torch.no_grad(), autocast_context(config):
                worlds['D'].delta = conds['D'](nl,16)
                pred = worlds['D'](s,F.pad(ac,(0,1)))[0][:,:15]
                cc = components(pred,s[:,1:])
                expected = TW.rollout_losses(worlds['D'],s,ac,'teacher',recon='deltairis')
                assert torch.equal(objective(cc,'D'),expected), 'D formula does not match original'
                original_l1 = TW.rollout_losses(worlds['D'],s,ac,'teacher',recon='l1')
                assert torch.equal(objective(cc,'L1'),original_l1), 'L1 formula does not match original'
                worlds['D'].delta = None
            log(stage='loss_parity',D=True,L1=True,initial_weights_equal=True)
        step_rows = {}
        cpu_rng, gpu_rng = torch.get_rng_state(), torch.cuda.get_rng_state_all()
        for arm in ARMS:
            torch.set_rng_state(cpu_rng); torch.cuda.set_rng_state_all(gpu_rng)
            world,oc = worlds[arm],conds[arm]
            with autocast_context(config):
                world.delta = oc(nl,16)
                pred = world(s,F.pad(ac,(0,1)))[0][:,:15]
                c = components(pred,s[:,1:]); loss = objective(c,arm)
                world.delta = None
            norm = float(optimizer_step(opts[arm],loss,params[arm],learning_rate=_phase_lr(config,update),
                         grad_clip=config.agent.grad_clip,strict=True,zero_grad=True))
            row = {'loss':float(loss.detach()),'components':{k:float(v.detach()) for k,v in c.items()},'gradient_norm':norm}
            summaries[arm]['norm_sum'] += norm
            summaries[arm]['clip_count'] += int(norm > config.agent.grad_clip)
            summaries[arm]['steps'] += 1
            step_rows[arm] = row
        if (update+1)%100 == 0 or update==0:
            row = {'update':update+1,'arms':step_rows,'seconds':round(time.time()-started,1),
                   'gpu_peak_gb':round(torch.cuda.max_memory_allocated()/2**30,3),'ledger_chain':ledger_chain}
            history.append(row); log(stage='train',**row)
        if (update+1)%a.save_every==0 or update+1 in milestones or update+1==a.stop_after:
            checkpoint(update+1)
            if update+1 in milestones: snapshots(update+1)
        if update+1 == a.stop_after:
            log(stage='interruption_check',update=update+1,resumable=True); return
    final = {'contract_id':contract_id,'update':a.updates,'summaries':summaries,'history':history,'ledger_chain':ledger_chain}
    atomic(final,directory/'training.json',True)
    log(stage='done',**{k:v for k,v in final.items() if k!='history'})


if __name__ == '__main__': main()
