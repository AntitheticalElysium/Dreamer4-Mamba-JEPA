"""Mathematical normalisation check at actual captured parameters, not modified model inputs."""
from .common import *
from .density import reference_nll


if __name__=='__main__':
    z=torch.load(OUT/'captured_real_density.pt',weights_only=False,mmap=True);mu=z['mu'].flatten();ls=z['log_s'].flatten()
    # Integer selection avoids float32 linspace rounding at tens of millions of indices.
    ids=torch.from_numpy(np.arange(2048,dtype=np.int64)*(len(mu)-1)//2047)
    m=mu[ids,None];s=ls[ids,None];y=torch.arange(256,dtype=torch.uint8)[None]
    model_log=-M.pixel_nll(m,s,y);ref_log=-reference_nll(m,s,y)
    total=model_log.double().exp().sum(-1);rt=ref_log.exp().sum(-1)
    save('likelihood_normalisation.json',{'n_actual_parameter_pairs':len(ids),
         'max_absolute_normalisation_error_canonical':float((total-1).abs().max()),
         'max_absolute_normalisation_error_reference':float((rt-1).abs().max()),
         'max_abs_pairwise_nll_difference':float((model_log.double()-ref_log).abs().max()),
         'fraction_bins_abs_nll_error_gt_01':float(((model_log.double()-ref_log).abs()>.1).double().mean())})
