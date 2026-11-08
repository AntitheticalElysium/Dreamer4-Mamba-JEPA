"""Exact instrumented replay of completed E2 fits, not a new head recipe."""
import importlib
import time
import numpy as np
import torch

E=importlib.import_module('artifacts.experiments.20261103_chooser_development.campaign')
C=E.C


def main():
    torch.set_num_threads(2); torch.backends.cuda.matmul.allow_tf32=True
    device=torch.device('cuda'); start=time.time(); identity={}
    train=E.Q.Data(E.load_cache(10,'train',device),device)
    dev=E.Q.Data(E.load_cache(10,'dev',device),device)
    for seed in E.SEEDS:
        h,opt,rng=E.init_fit(seed,device)
        E.snapshot(h,dev,f'replay_fresh_{seed}_u00000')
        def hook(step):
            if step in (100,1000,2000,4000,6000,8000,10000,12000):
                before=rng.get_state().clone()
                E.snapshot(h,dev,f'replay_fresh_{seed}_u{step:05d}')
                assert torch.equal(before,rng.get_state())
        curve=E.advance(h,opt,rng,train,0,12000,12000,hook)
        differences={}
        for cohort in ('saved','confirmation'):
            old=np.load(C.HERE/f'e2_fresh_{seed}_ck10_endpoint_{cohort}.npz')
            new=np.load(C.HERE/f'replay_fresh_{seed}_u12000_{cohort}.npz')
            differences[cohort]={k:float(np.max(np.abs(old[k].astype(float)-new[k].astype(float))))
                                 for k in ('C','S','prediction','noop','mse','counter_C','counter_mse')}
            assert all(v==0 for v in differences[cohort].values()),differences
        identity[str(seed)]=dict(endpoint_max_abs_differences=differences,curve=curve)
    C.write(C.HERE/'fresh_replay_identity.json',dict(seeds=identity,seconds=time.time()-start,
        peak_GiB=torch.cuda.max_memory_allocated()/2**30,
        script_sha256=C.sha(__file__),plan_sha256=C.sha(C.HERE/'DISSECTION_PLAN.md')))


if __name__=='__main__':main()
