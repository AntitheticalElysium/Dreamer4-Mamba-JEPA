"""Exact checks of the proposed pooled-global P2 loss; no learned-model claim."""
import json
from pathlib import Path
import numpy as np
import torch
import sys


def population_score(forecast, truth):
    """Enumerate independent draws from uniform finite supports, including repeats."""
    a=np.linalg.norm(forecast[:,None]-truth[None,:],axis=-1).mean()
    b=np.linalg.norm(forecast[:,None]-forecast[None,:],axis=-1).mean()
    return float(a-0.5*b)


def main():
    H=16
    P=np.zeros((2,H,9,9,1)); Q=np.zeros_like(P)
    # P alternates between diagonal corner pairs; Q between top and bottom pairs.
    # Each complete trajectory keeps its selected pattern throughout all H steps.
    for world, corners in enumerate((((0,0),(8,8)),((0,8),(8,0)))):
        for r,c in corners: P[world,:,r,c,0]=1
    for world, corners in enumerate((((0,0),(0,8)),((8,0),(8,8)))):
        for r,c in corners: Q[world,:,r,c,0]=1
    patch_true=patch_wrong=0.0
    marginals_equal=True
    for r in range(7):
        for c in range(7):
            x=P[:,:,r:r+3,c:c+3].reshape(2,-1)
            z=Q[:,:,r:r+3,c:c+3].reshape(2,-1)
            marginals_equal &= sorted(map(tuple,x))==sorted(map(tuple,z))
            patch_true+=population_score(x,x)
            patch_wrong+=population_score(z,x)
    p=P.mean((2,3)).reshape(2,-1); q=Q.mean((2,3)).reshape(2,-1)
    full_p=P.reshape(2,-1); full_q=Q.reshape(2,-1)
    # Forward stochastic draws can be correct while carrying no pathwise gradient.
    logit=torch.tensor(0.0,requires_grad=True)
    discrete=torch.distributions.Bernoulli(logits=logit).sample()
    location=torch.tensor(0.3,requires_grad=True)
    bin_index=torch.round(location/.01)
    quantized=.01*bin_index
    quantized.backward()
    result={
        'conditional_on':'fixed central glimpse; lighting, reward and termination constant',
        'all_49_spatiotemporal_patch_marginals_identical':bool(marginals_equal),
        'pooled_trajectory_distributions_identical':bool(np.array_equal(p,q)),
        'patch_score_correct':patch_true,'patch_score_wrong':patch_wrong,
        'global_pooled_score_correct':population_score(p,p),
        'global_pooled_score_wrong':population_score(q,p),
        'full_grid_score_correct':population_score(full_p,full_p),
        'full_grid_score_wrong':population_score(full_q,full_p),
        'Bernoulli_sample_requires_grad':discrete.requires_grad,
        'hard_quantization_pathwise_gradient':float(location.grad),
        'P1_updates_before_valid_frame_masks':{
            '64_steps_16_streams':12_800_000/(64*16),
            '256_steps_16_streams':12_800_000/(256*16)},
        'corner_training_patch_memberships':1,'center_training_patch_memberships':9,
    }
    assert marginals_equal and patch_true==patch_wrong and np.array_equal(p,q)
    assert result['full_grid_score_wrong']>result['full_grid_score_correct']
    path=Path(__file__).resolve().with_name('proposal_checks.json')
    path.write_text(json.dumps(result,indent=2)); print(json.dumps(result,indent=2))


def bookkeeping():
    from saccade.eval import gates as G
    L=dict(np.load(Path(__file__).resolve().parents[2]/'eda/s0/s0_dev_labels.npz'))
    offsets=np.r_[0,np.cumsum(L['ep_T'])]
    counts=np.zeros((2,2)); nums=np.zeros((2,4)); dens=np.zeros((2,4)); windows=0
    for e,ep in enumerate(G.episode_grids(L)):
        for t in range(0,len(ep['reward'])-16+1,16):
            if not L['t_exogenous'][offsets[e]+t:offsets[e]+t+16].all(): continue
            done=ep['done'][t:t+16]
            n=int(np.argmax(done))+2 if done.any() else 17
            grid=ep['grid'][t:t+n]; pos=ep['pos'][t:t+n]
            corrupt=grid.copy(); corrupt[1:][corrupt[1:]==4]=0
            for arm,g in enumerate((grid,corrupt)):
                counts[arm]+=[(g[1:]>0).sum(),len(g)-1]
                a,b=G.persistence(g,pos); nums[arm]+=a; dens[arm]+=b
            windows+=1
    count=counts[:,0]/counts[:,1]; persist=nums[:,3]/dens[:,3]
    delta=float(count[1]/count[0]-1); pdelta=float(persist[1]-persist[0])
    result={'windows':windows,'corruption':'all arrows removed from generated frames; real start kept',
            'reference_mean_count':float(count[0]),'corrupt_mean_count':float(count[1]),
            'relative_count_difference':delta,'reference_persistence_4':float(persist[0]),
            'corrupt_persistence_4':float(persist[1]),'persistence_difference':pdelta,
            'blocked_by_proposed_aggregate_bookkeeping':abs(delta)>.30 or pdelta<-.10}
    Path(__file__).resolve().with_name('proposal_bookkeeping.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__': bookkeeping() if len(sys.argv)>1 else main()
