"""One-step repeated-key MSE decomposition on the independent 1002-root future store.

For K=4 true successors per action, compare the frozen world's prediction to
the sample conditional mean, persistent root, and irreducible key variance.
Descriptive: roots come from ordinary BC trajectories, not fatal opportunities.
"""
import json
from pathlib import Path

import torch

HERE = Path(__file__).parent
ROOT = HERE.parents[2]
DATA = ROOT / "artifacts/eda/diagnosis_rollouts_v1"

def main():
    torch.set_num_threads(6)
    out = {}
    for arm in ("U", "W", "H2", "Z"):
        d = torch.load(DATA / f"{arm}.pt", weights_only=False, mmap=True)
        root, gen, true = d["root"].float(), d["one"].float(), d["one_true"].float()
        n, k, a, dim = true.shape
        mean = true.mean(1)
        V = float(true[:, 0].reshape(-1, dim).var(0).sum())
        norm = lambda x: x.square().sum(-1)
        # per (root,action), unbiased random-key variance and unbiased squared bias
        noise = norm(true - mean[:, None]).sum(1) / (k-1)
        mean_uncertainty = noise/k
        bias = norm(gen - mean) - mean_uncertainty
        signal = norm(mean - root[:, None]) - mean_uncertainty
        oracle_mse = norm(true[:,0]-mean)
        gen_mse = norm(gen-true[:,0])
        persistent_mse = norm(root[:,None]-true[:,0])
        groups={"all":torch.ones(n,a,dtype=torch.bool),
                "move_actions":torch.zeros(n,a,dtype=torch.bool),
                "stay_actions":torch.ones(n,a,dtype=torch.bool)}
        groups["move_actions"][:,1:5]=True; groups["stay_actions"][:,1:5]=False
        out[arm]={"roots":n,"keys":k,"actions":a,"V":V,"groups":{}}
        for name,m in groups.items():
            b,ng,sg,gm,pm,om=[float(t[m].mean()/V) for t in (bias,noise,signal,gen_mse,persistent_mse,oracle_mse)]
            out[arm]["groups"][name]={"bias_to_conditional_mean":b,"key_noise":ng,
                                       "predictable_signal":sg,"fraction_signal_captured":1-b/sg,
                                       "gen_to_one_true":gm,"persist_to_one_true":pm,
                                       "sample_mean_to_one_true":om,
                                       "excess_gen_error_over_key_noise":gm-ng}
        print(arm,json.dumps(out[arm]["groups"]),flush=True)
    (HERE/"mean_bound.json").write_text(json.dumps(out,indent=2)+"\n")

if __name__=="__main__":main()
