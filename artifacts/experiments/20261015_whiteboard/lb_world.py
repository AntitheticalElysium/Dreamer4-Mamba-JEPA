"""LB: an ordinary evolving latent-belief world model -- the straightforward recurrent baseline (2026-10-15 late; user request,
ARCHITECTURE_RESEARCH.md family 2). LB_PREREG.md governs the comparison with PI.

R2-Dreamer's RSSM imported UNCHANGED from the vendored source (third_party/sources/NM512__r2dreamer: rssm.RSSM, its block-GRU
deterministic path, one-hot categorical stochastic state with unimix, KL balancing with free bits) at its 12M configuration
(deter 2048, hidden 256, stoch 32 x discrete 16, blocks 8, obs_layers 1, img_layers 2, dyn_layers 1, unimix 0.01), with its LaProp
optimizer and adaptive gradient clipping (lr 4e-5, eps 1e-20, betas 0.9 / 0.999, AGC 0.3, pmin 1e-3, warmup 1,000), KL free 1.0,
scales dyn 1.0 / rep 0.1. Observations enter ONLY through the posterior; imagination advances (stoch, deter) by prior samples and
never feeds predictions back.
Two adaptations to our setting (disclosed):
  embed    an MLP over the frozen Raw JEPA tokens of the frame (81 x 192 -> 1024 -> 1024 -> 1024; RMSNorm + SiLU), in place of
           R2-Dreamer's pixel CNN (our evidence is the token grid, as for PI)
  readout  the representation objective is the prediction of the CURRENT frame's JEPA tokens from feat = (stoch, deter) by a
           token head (feat -> 1024 -> 81 x 64, + learned cell embedding, per-cell MLP 64 -> 256 -> 192), scored with the same D
           token loss PI uses (loss_intervention.objective 'D') -- reconstruction-free (no pixels) and the same target family as
           PI; R2-Dreamer's Barlow alignment would leave no token readout for the frozen readers. Scale 1.0.
Windows / batch exactly as PI (seeded training-ledger windows of 16 frames, batch 16); each window starts the RSSM (is_first).
Actions: one-hot 17; the RSSM step for frame i takes the action at frame i-1 (zero at the first frame).
Usage: lb_world.py --seed-set 1|2 [--updates 30000] [--stop-after N] -> transport/lb_seed<s>/lb_u<updates>.pt (+ resumable state)
"""
import argparse
import copy as cp
import hashlib
import json
import os
import sys
import time
import types
from pathlib import Path
from types import SimpleNamespace as NS

os.environ.setdefault('TRITON_F32_DEFAULT', 'ieee')
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
R2D = ROOT / 'third_party/sources/NM512__r2dreamer'
sys.path.insert(0, str(HERE.parent / '20261012_oracle'))
sys.path.insert(0, str(HERE))
_stub = types.ModuleType('torch.utils.tensorboard'); _stub.SummaryWriter = object              # R2-Dreamer tools.py logs only
sys.modules.setdefault('torch.utils.tensorboard', _stub)
sys.path.insert(0, str(R2D))

import torch
from torch import nn
from torch.nn import functional as F

import rssm as R2RSSM                                                              # R2-Dreamer, unchanged
from optim import LaProp, clip_grad_agc_                                           # R2-Dreamer, unchanged
from tools import weight_init_                                                     # R2-Dreamer, unchanged

OUT_NAME = 'transport'
CFG = dict(stoch=32, deter=2048, hidden=256, discrete=16, act='SiLU', unimix_ratio=0.01, initial='learned', obs_layers=1,
           img_layers=2, dyn_layers=1, blocks=8)
OPT = dict(lr=4e-5, eps=1e-20, beta1=0.9, beta2=0.999, agc=0.3, pmin=1e-3, warmup=1000)
KL_FREE, DYN, REP, TOK = 1.0, 1.0, 0.1, 1.0
N_ACT = 17


def block(i, o):
    return nn.Sequential(nn.Linear(i, o), nn.RMSNorm(o, eps=1e-4), nn.SiLU())


class LBWorld(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.rssm = R2RSSM.RSSM(NS(**CFG, device=device), embed_size=1024, act_dim=N_ACT)
        self.embed = nn.Sequential(block(81 * 192, 1024), block(1024, 1024), block(1024, 1024))
        self.head_in = nn.Sequential(block(self.rssm.feat_size, 1024), nn.Linear(1024, 81 * 64))
        self.cell = nn.Parameter(torch.zeros(81, 64))
        self.head_out = nn.Sequential(block(64, 256), nn.Linear(256, 192))
        for m in (self.embed, self.head_in, self.head_out):
            m.apply(weight_init_)

    def readout(self, stoch, deter):
        """-> predicted LN tokens [..., 81, 192] of the frame the state describes"""
        f = self.rssm.get_feat(stoch, deter)
        x = self.head_in(f).reshape(*f.shape[:-1], 81, 64) + self.cell
        return F.layer_norm(self.head_out(x), (192,))

    def observe(self, frames, actions):
        """frames [B,T,81,192], actions [B,T] long (action at frame t) -> posterior (stoch, deter, logits) for every frame"""
        B, T = actions.shape
        prev = torch.zeros(B, T, N_ACT, device=frames.device)
        prev[:, 1:] = F.one_hot(actions[:, :-1], N_ACT).float()
        first = torch.zeros(B, T, dtype=torch.bool, device=frames.device); first[:, 0] = True
        embed = self.embed(frames.flatten(-2))
        return self.rssm.observe(embed, prev, self.rssm.initial(B), first)


def losses(w, frames, actions):
    import loss_intervention as L
    stoch, deter, post_logit = w.observe(frames, actions)
    _, prior_logit = w.rssm.prior(deter)
    dyn, rep = w.rssm.kl_loss(post_logit, prior_logit, KL_FREE)
    pred = w.readout(stoch, deter)
    tok = L.objective(L.components(pred, frames.float()), 'D')
    loss = DYN * dyn.mean() + REP * rep.mean() + TOK * tok
    return loss, {'dyn': float(dyn.mean()), 'rep': float(rep.mean()), 'token': float(tok),
                  'prior_entropy': float(w.rssm.get_dist(prior_logit).entropy().mean()),
                  'post_entropy': float(w.rssm.get_dist(post_logit).entropy().mean())}


def main():
    import direct_train as BASE
    import loss_intervention as L
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--seed-set', type=int, required=True, choices=(1, 2))
    ap.add_argument('--updates', type=int, default=30000)
    ap.add_argument('--stop-after', type=int)
    a = ap.parse_args()
    prereg = HERE / 'LB_PREREG.md'
    assert prereg.exists()
    manifest, windows, actions, tokens = BASE.load_data()
    init_seed, window_seed = {1: (7, 20261014), 2: (17, 20261016)}[a.seed_set]           # PI's / TRANSPORT_IC's window streams
    torch.manual_seed(init_seed)
    w = LBWorld('cuda').cuda().train()
    params = list(w.parameters())
    opt = LaProp(params, lr=OPT['lr'], betas=(OPT['beta1'], OPT['beta2']), eps=OPT['eps'])
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / OPT['warmup']))
    d = L.OUT / OUT_NAME / f'lb_seed{a.seed_set}'; d.mkdir(parents=True, exist_ok=True)
    source = [Path(__file__), prereg, R2D / 'rssm.py', R2D / 'optim/laprop.py', R2D / 'optim/agc.py', R2D / 'distributions.py']
    contract = {'format': 'lb_v1', 'sources_inputs': {str(p.relative_to(L.ROOT)): L.sha(p) for p in source},
                'data_contract_id': manifest['contract_id'], 'seed_set': a.seed_set, 'updates': a.updates, 'cfg': CFG, 'opt': OPT,
                'parameters': sum(p.numel() for p in params), 'training_semantic_or_simulator_labels': False}
    cid = hashlib.sha256(json.dumps(contract, sort_keys=True).encode()).hexdigest()
    if (d / 'contract.json').exists():
        assert json.loads((d / 'contract.json').read_text()) == contract, 'source/input drift'
    else:
        L.atomic(contract, d / 'contract.json', True)
    order = torch.Generator().manual_seed(window_seed)
    start, history = 0, []
    if (d / 'state.pt').exists():
        st = torch.load(d / 'state.pt', map_location='cpu', weights_only=False)
        assert st['contract_id'] == cid
        w.load_state_dict(st['world']); opt.load_state_dict(st['optimizer']); sched.load_state_dict(st['sched'])
        order.set_state(st['order']); torch.set_rng_state(st['torch_rng']); torch.cuda.set_rng_state_all(st['cuda_rng'])
        start, history = st['update'], st['history']
        print(json.dumps({'stage': 'resume', 'update': start}), flush=True)

    def save(update, snap=False):
        state = L.cpu_state(w)
        L.atomic({'contract_id': cid, 'update': update, 'world': state, 'optimizer': cp.deepcopy(opt.state_dict()),
                  'sched': sched.state_dict(), 'order': order.get_state(), 'torch_rng': torch.get_rng_state(),
                  'cuda_rng': torch.cuda.get_rng_state_all(), 'history': history}, d / 'state.pt')
        if snap:
            L.atomic({'contract_id': cid, 'update': update, 'arm': 'lb', 'world': state}, d / f'lb_u{update:05d}.pt')
    began = time.perf_counter()
    for update in range(start, a.updates):
        idx = torch.randint(len(windows), (16,), generator=order)
        frames, ac, _ = BASE.batch(tokens, actions, windows, idx)
        loss, stats = losses(w, frames, ac)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        clip_grad_agc_(params, OPT['agc'], OPT['pmin'], foreach=True)
        opt.step(); sched.step()
        if (update + 1) % 250 == 0 or update == start:
            history.append({'update': update + 1, **{k: round(v, 5) for k, v in stats.items()}, 'elapsed': round(time.perf_counter() - began, 1),
                            'peak_allocated_gib': round(torch.cuda.max_memory_allocated() / 2**30, 3)})
            print(json.dumps(history[-1]), flush=True)
        snap = (update + 1) % 5000 == 0 or update + 1 == a.updates
        if (update + 1) % 500 == 0 or snap:
            save(update + 1, snap)
        if a.stop_after and update + 1 >= a.stop_after:
            return
    print(json.dumps({'stage': 'done', 'update': a.updates}), flush=True)


if __name__ == '__main__':
    main()
