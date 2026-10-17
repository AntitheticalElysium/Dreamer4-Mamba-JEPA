"""MATCHED_ENCODER_PREREG.md checks R0a-i / R0a-ii and the MoP-s1 pool.

r0a       R0a-i: are the encoder tensors of raw/bridge/step-002000.pt (which built spatial_pool_v1) bitwise equal to those of
          raw/joint/step-010000.pt (canonical Raw-s1)? R0a-ii: for 512 seeded stored windows, re-encode the corpus frames at the
          stored (episode, start) with the bridge encoder through spatial.encode and compare with the stored tokens; compare
          the stored actions with the corpus actions.
build     spatial_pool_mop_s1_v1/pool.pt: every stored window, re-encoded by MoP-s1's encoder through the SAME spatial.encode
          (LN of last-layer patch tokens, LN of CLS); actions / reward_led / reward_valid / alive / dh / ids / terminal copied.
Usage: matched_pool.py r0a | build   -> artifacts/eda/matched_encoder_v1/r0a.json, artifacts/eda/spatial_pool_mop_s1_v1/
"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'artifacts/experiments/20260921_readout_ladder'))
import spatial as S                                                            # noqa: E402

RAW_POOL = ROOT / 'artifacts/eda/spatial_pool_v1'
MOP_POOL = ROOT / 'artifacts/eda/spatial_pool_mop_s1_v1'
MOP_CK = ROOT / 'artifacts/eda/spatial_encoder_v1/stage2_mop_sig_s1/step-010000.pt'
JOINT_CK = ROOT / 'artifacts/lewm_m4_canonical/raw/joint/step-010000.pt'
OUTD = ROOT / 'artifacts/eda/matched_encoder_v1'
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()


def corpus():
    from d4mj.data import load_joint_corpus
    _, config = S.bridge()
    record = json.loads(S.DATASET.read_text())
    episodes, contract = load_joint_corpus(record['paths'], config)
    assert contract == record['contract'], 'corpus differs from its dataset contract'
    return {e.episode_id: e for e in episodes}


def mop_encoder():
    from d4mj.config import load_recipe
    from d4mj.lewm import LeWMEncoder
    enc = LeWMEncoder(load_recipe(ROOT / 'd4mj/recipes/lewm_mamba_raw.json')).cuda().eval()
    enc.load_state_dict(torch.load(MOP_CK, map_location='cuda', weights_only=False)['encoder'])
    for p in enc.parameters():
        p.requires_grad_(False)
    return enc


@torch.no_grad()
def r0a():
    from d4mj.checkpoint import read_lewm_bundle
    bridge_enc, _ = S.bridge()
    joint = read_lewm_bundle(JOINT_CK)['modules']['encoder']
    b = {k: v.detach().cpu() for k, v in bridge_enc.state_dict().items()}
    keys_equal = sorted(b) == sorted(joint)
    tensors_equal = keys_equal and all(torch.equal(b[k], joint[k].cpu()) for k in b)
    pool = torch.load(RAW_POOL / 'pool.pt', weights_only=False, mmap=True)
    eps = corpus()
    pick = torch.randperm(len(pool['ids']), generator=torch.Generator().manual_seed(20261108))[:512].tolist()
    frames = np.stack([np.asarray(eps[pool['ids'][i][0]].observations[pool['ids'][i][1]:pool['ids'][i][1] + S.W]) for i in pick])
    _, tok = S.encode(bridge_enc.cuda(), frames, 'cuda')
    stored = pool['tokens'][pick].float()
    acts = torch.stack([torch.as_tensor(np.asarray(eps[pool['ids'][i][0]].actions_taken[pool['ids'][i][1]:pool['ids'][i][1] + S.W - 1])).long() for i in pick])
    res = {'R0a_i_encoder_keys_equal': keys_equal, 'R0a_i_encoder_tensors_bitwise_equal': tensors_equal,
           'R0a_ii_windows_checked': len(pick), 'R0a_ii_max_abs_token_diff': float((tok.float() - stored).abs().max()),
           'R0a_ii_tolerance': 2e-3, 'R0a_ii_actions_equal': bool(torch.equal(acts, pool['actions'][pick])),
           'bridge_checkpoint_sha256': sha(S.CHECKPOINT), 'joint_checkpoint_sha256': sha(JOINT_CK), 'pool_sha256': sha(RAW_POOL / 'pool.pt')}
    res['R0a_pass'] = bool(res['R0a_i_encoder_tensors_bitwise_equal'] and res['R0a_ii_max_abs_token_diff'] <= 2e-3 and res['R0a_ii_actions_equal'])
    OUTD.mkdir(parents=True, exist_ok=True)
    (OUTD / 'r0a.json').write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


@torch.no_grad()
def build():
    assert json.loads((OUTD / 'r0a.json').read_text())['R0a_pass'], 'R0a has not passed'
    pool = torch.load(RAW_POOL / 'pool.pt', weights_only=False, mmap=True)
    eps = corpus()
    enc = mop_encoder()
    n = len(pool['ids'])
    out = {'z': torch.empty(n, S.W, S.WIDTH), 'tokens': torch.empty(n, S.W, S.TOKENS, S.WIDTH, dtype=torch.float16)}
    for b in range(0, n, 256):
        ids = pool['ids'][b:b + 256]
        frames = np.stack([np.asarray(eps[e].observations[s:s + S.W]) for e, s in ids])
        z, tok = S.encode(enc, frames, 'cuda')
        out['z'][b:b + len(ids)], out['tokens'][b:b + len(ids)] = z, tok
    new = {k: (pool[k].clone() if isinstance(pool[k], torch.Tensor) else pool[k]) for k in ('actions', 'reward_led', 'reward_valid', 'alive', 'dh', 'ids', 'terminal')}
    meta = {**pool['meta'], 'encoder': 'MoP+SIGReg seed 1, backbone-LN (spatial.encode)', 'encoder_checkpoint': str(MOP_CK.relative_to(ROOT)),
            'encoder_checkpoint_sha256': sha(MOP_CK), 'source_pool_sha256': sha(RAW_POOL / 'pool.pt'),
            'note': 'windows, actions and labels copied from spatial_pool_v1; only z / tokens re-encoded (MATCHED_ENCODER_PREREG.md)'}
    MOP_POOL.mkdir(parents=True, exist_ok=True)
    torch.save({**new, **out, 'meta': meta}, MOP_POOL / 'pool.pt')
    (MOP_POOL / 'pool.json').write_text(json.dumps({**meta, 'pool_sha256': sha(MOP_POOL / 'pool.pt')}, indent=2) + '\n')
    print(json.dumps({'built': str(MOP_POOL), 'windows': n}))


if __name__ == '__main__':
    {'r0a': r0a, 'build': build}[sys.argv[1]]()
