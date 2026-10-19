"""Head-only content categorical for the trained PI world (PI_INNOV_PREREG.md governs; 2026-10-15 night).

The PI u3000 world of the seed set is frozen. Per update (batch 16 seeded training-ledger windows, L16), PI's own TRAINING-TIME
inference exactly as pi_train.losses (copy under the label-free scroll estimate; provisional afterstate with the gate's own
decisions; IC v3 posterior with the chance head's prior; edit posterior q = sigmoid(gain / s + gate logit)) selects the
innovation cells (q > 0.5, or entering). A linear head (256 -> 4096, zero init) on the frozen belief at those cells is
trained by cross-entropy toward the nearest code (E2's Raw k-means codebook, K 4096, levers_codebooks_v1/raw_K4096.pt) of the
pre-chance target z_pre. No simulator labels.
Usage: pi_innov_train.py --seed-set 1|2 [--updates 1500] [--stop-after N] -> transport/pi_seed<s>/innov_u1500.pt
"""
import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault('TRITON_F32_DEFAULT', 'ieee')
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / '20261012_oracle'))
sys.path.insert(0, str(HERE))

import torch
from torch.nn import functional as F

import audit_s5_direct as A5
import direct_model as DM
import direct_train as BASE
import loss_intervention as L
import transport_ic_train as IC
import pi_model as PI
from d4mj.train import autocast_context

OUT = L.OUT / 'transport'
PREREG = HERE / 'PI_INNOV_PREREG.md'
CODES = L.ROOT / 'artifacts/eda/levers_codebooks_v1/raw_K4096.pt'


@torch.no_grad()
def innovation_cells(w, frames, ac, noise, config, scroll):
    """PI's training-time innovation cells and targets: belief [B,15,81,256], innov [B,15,81] bool, z_pre [B,15,81,192]"""
    est = scroll.estimate(frames[:, :-1].float(), frames[:, 1:].float())
    with autocast_context(config):
        belief, ha = w.backbone_full(frames, ac)
        copy, gen, glogit, enter, _ = w.afterstate_parts(frames, ac, belief, ha, scroll=F.pad(est, (0, 1)))
        logits = w.completion.propose(w.completion.trunk(belief, noise)).float()
    belief, copy, gen, glogit, enter, logits = (x[:, :15] for x in (belief, copy, gen, glogit, enter, logits))
    logits = logits.masked_fill(~w.completion.valid, -torch.inf)
    z = frames[:, 1:].float()
    a0 = torch.where(((glogit > 0) | enter)[..., None], gen, copy)
    partner, _, _, _, _, _ = IC.infer(F.layer_norm(a0, (192,)), z, IC.pair_prior_logit(logits))
    z_pre = z.gather(2, partner[..., None].expand(-1, -1, -1, 192))
    res_copy = (F.layer_norm(copy, (192,)) - z_pre).norm(dim=-1)
    gain = res_copy - (F.layer_norm(gen, (192,)) - z_pre).norm(dim=-1)
    s = res_copy[..., :PI.MAP].mean(-1, keepdim=True).clamp_min(1e-6)
    innov = (torch.sigmoid(gain / s + glogit) > .5) | enter
    return belief.float(), innov, z_pre


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--seed-set', type=int, required=True, choices=(1, 2))
    ap.add_argument('--updates', type=int, default=1500)
    ap.add_argument('--stop-after', type=int)
    a = ap.parse_args()
    assert PREREG.exists()
    config = BASE.settings()
    manifest, windows, actions, tokens = BASE.load_data()
    d = OUT / f'pi_seed{a.seed_set}'
    w = PI.PIWorld().cuda().eval()
    w.load_state_dict(torch.load(d / 'pi_u3000.pt', map_location='cpu', weights_only=False)['world'])
    for p in w.parameters():
        p.requires_grad_(False)
    codes = torch.load(CODES, map_location='cpu', weights_only=False)['codes'].float().cuda()
    head = torch.nn.Linear(256, codes.shape[0]).cuda()
    torch.nn.init.zeros_(head.weight); torch.nn.init.zeros_(head.bias)
    opt = torch.optim.Adam(head.parameters(), lr=1e-3)
    scroll = A5.load_scroll()
    order = torch.Generator().manual_seed(20261020 + a.seed_set); noise_rng = torch.Generator().manual_seed(20261030 + a.seed_set)
    source = [Path(__file__), Path(PI.__file__), Path(IC.__file__), PREREG, CODES, d / 'pi_u3000.pt']
    contract = {'format': 'pi_innov_v1', 'sources_inputs': {str(p.relative_to(L.ROOT)): L.sha(p) for p in source},
                'data_contract_id': manifest['contract_id'], 'seed_set': a.seed_set, 'updates': a.updates,
                'training_semantic_or_simulator_labels': False}
    cid = hashlib.sha256(json.dumps(contract, sort_keys=True).encode()).hexdigest()
    history, began = [], time.perf_counter()
    for update in range(a.updates):
        idx = torch.randint(len(windows), (16,), generator=order)
        frames, ac, _ = BASE.batch(tokens, actions, windows, idx)
        noise = DM.draw_noise(frames.shape[:2], noise_rng).cuda()
        belief, innov, z_pre = innovation_cells(w, frames, ac, noise[0], config, scroll)
        target = torch.cdist(z_pre[innov], codes).argmin(-1)
        logits = head(belief[innov])
        loss = F.cross_entropy(logits, target)
        opt.zero_grad(); loss.backward(); opt.step()
        if (update + 1) % 100 == 0 or update == 0:
            with torch.no_grad():
                p = logits.softmax(-1)
                row = {'update': update + 1, 'ce': round(float(loss), 4), 'top1': round(float((logits.argmax(-1) == target).float().mean()), 4),
                       'entropy': round(float(-(p * p.clamp_min(1e-12).log()).sum(-1).mean()), 3),
                       'innov_per_frame': round(float(innov.float().sum(-1).mean()), 3), 'elapsed': round(time.perf_counter() - began, 1)}
            history.append(row); print(json.dumps(row), flush=True)
        if a.stop_after and update + 1 >= a.stop_after:
            return
    L.atomic({'contract_id': cid, 'contract': contract, 'update': a.updates, 'head': L.cpu_state(head), 'history': history},
             d / f'innov_u{a.updates:04d}.pt')
    print(json.dumps({'stage': 'done', 'update': a.updates}), flush=True)


if __name__ == '__main__':
    main()
