"""Actual full-world reload, S1 encoding, cascade and saved-head read paths on two dev episodes."""
import importlib
import json
from pathlib import Path
import numpy as np
import torch
from saccade import model as M, train as TR
from saccade.eval import s1 as S
C=importlib.import_module('artifacts.experiments.20261103_p1f2.chooser')
HERE=Path(__file__).resolve().parent


@torch.no_grad()
def main():
    torch.set_num_threads(4);ckpath=HERE/'checks/smoke_model.pt';ck=torch.load(ckpath,map_location='cpu',weights_only=False)
    cfg=ck['cfg'];m=M.M0(cfg).cuda().eval();m.load_state_dict(ck['model']);dev=torch.device('cuda')
    labels=S.first_episodes(dict(np.load(S.LABELS/'s0_dev_labels.npz')),2);enc=S.encode(m,labels,cfg)
    e,t=0,9;q=enc[e];take=lambda k,s:q[k][s:s+1].to(dev)
    cache=m.denoiser.new_cache();gen=torch.Generator(device=dev).manual_seed(612350)
    for s in range(t-8,t):
        cache=M.context_pass(m.denoiser,tuple(take(k,s).float()[None] for k in ('p','g','v')),
            take('prev_action',s)[None],m.belief.memory(take('bp',s).float())[:,None],take('fix',s)[None],cache,cfg,gen)
    out=M.sample(m.denoiser,[tuple(q[k].shape[1:]) for k in ('p','g','v')],
        take('prev_action',t)[None],m.belief.memory(take('bp',t).float())[:,None],take('fix',t)[None],cache,cfg,gen)
    assert all(torch.isfinite(x).all() for x in out)
    # Exact coarse-only/full-perception equivalence also under production CUDA autocast.
    frames=S.D.RP.episode_fields(S.episodes(labels)[0].shard,S.episodes(labels)[0].slot)['observations'][:16].cuda()
    with TR.autocast(dev,cfg):
        p,v=C.coarse(m.perception,frames);full=m.perception(frames,torch.full((len(frames),2),4,device=dev))
    assert torch.equal(p,full.p) and torch.equal(v,full.v)
    # Load the actual online head from the sustained smoke; no disposable head refit.
    h=C.Q.Head().cuda().eval();state=torch.load(HERE/'checks/sustained64/chooser_last.pt',map_location=dev,weights_only=False)
    h.load_state_dict(state['head']);Q=C.Q;Q.HERE=HERE/'checks/heldout_head';Q.HERE.mkdir(exist_ok=True);Q.CK=ckpath
    data=Q.Data(Q.collect(m,cfg,'dev',2,dev),dev)
    result=Q.predict(h,data,data.row_ids([0,1],[9,9]))
    assert np.isfinite(result['prob']).all() and np.allclose(result['prob'].sum(-1),1)
    report=dict(encoded_episodes=len(enc),cascade_shapes=[list(x.shape) for x in out],cascade_finite=True,
        cuda_coarse_bitwise_equal=True,online_head_updates=state['updates'],heldout_policy_rows=2,
        checkpoint_sha256=TR.file_sha256(ckpath))
    (HERE/'eval_smoke.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)


if __name__=='__main__':main()
