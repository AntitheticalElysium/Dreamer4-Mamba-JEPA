"""Tie-grouped average precision (correction of spatial_eval_v2.average_precision; Codex audit
../20261017_frozen_state_audit/ASSESSMENT.md, metric_falsifiers.py).

spatial_eval_v2.average_precision sorts scores and takes the precision at every positive in that order. With tied scores and mixed
labels the result depends on the arbitrary order inside the tie (4 equal scores, labels [1,1,0,0]: 1.0 vs 0.4167 reordered).
ap_grouped treats each distinct score as ONE threshold (precision at the end of the tie group x recall gained by the group), the
standard step-function AP (scikit-learn average_precision_score); it is Codex's grouped_ap verbatim.
ap_grouped_boot gives the same AP for episode-replicated bootstrap samples exactly, by weighting each item with its episode's
multiplicity (replicated copies share their score, hence their tie group) - no re-sorting per draw.
"""
import torch


def ap_grouped(score, label):
    s, l = score.double(), label.bool()
    if not bool(l.any()):
        return None
    o = s.argsort(descending=True)
    s, l = s[o], l[o].double()
    _, counts = torch.unique_consecutive(s, return_counts=True)
    ends = counts.cumsum(0) - 1
    tp = l.cumsum(0)[ends]
    inc = torch.diff(torch.cat([torch.zeros(1, dtype=torch.float64), tp]))
    return float(((tp / (ends + 1)) * inc).sum() / l.sum())


def ap_grouped_boot(score, label, ep, mult):
    """score, label, ep [n] (ep = episode index per item); mult [B, E] episode multiplicities -> (full-sample AP, tensor[B])"""
    s, l = score.double(), label.bool()
    o = s.argsort(descending=True)
    s, l, e = s[o], l[o].double(), ep[o]
    _, counts = torch.unique_consecutive(s, return_counts=True)
    ends = counts.cumsum(0) - 1
    group = torch.repeat_interleave(torch.arange(len(counts)), counts)

    def ap(w):                                                                  # w [n] item weights
        tp = torch.zeros(len(counts), dtype=torch.float64).index_add_(0, group, w * l).cumsum(0)
        n = torch.zeros(len(counts), dtype=torch.float64).index_add_(0, group, w).cumsum(0)
        inc = torch.diff(torch.cat([torch.zeros(1, dtype=torch.float64), tp]))
        P = tp[-1]
        return float(((tp / n.clamp_min(1e-12)) * inc).sum() / P) if P > 0 else float('nan')
    full = ap(torch.ones(len(s), dtype=torch.float64))
    boot = torch.tensor([ap(mult[b].double()[e]) for b in range(len(mult))]) if len(mult) else torch.empty(0)
    return full, boot


if __name__ == '__main__':                                                      # self-test against Codex's counterexamples
    z = torch.zeros(4)
    assert ap_grouped(z, torch.tensor([1, 1, 0, 0])) == 0.5 == ap_grouped(z, torch.tensor([0, 0, 1, 1]))
    g = torch.Generator().manual_seed(0)
    for trial in range(50):                                                     # weighted form == explicit replication
        n, E = 300, 12
        s = torch.randint(0, 8, (n,), generator=g).float(); l = torch.rand(n, generator=g) < .3; ep = torch.randint(E, (n,), generator=g)
        m = torch.randint(0, 3, (2, E), generator=g)
        full, boot = ap_grouped_boot(s, l, ep, m)
        assert abs(full - ap_grouped(s, l)) < 1e-12
        for b in range(2):
            idx = torch.cat([torch.nonzero(ep == k)[:, 0].repeat(int(m[b, k])) for k in range(E)])
            ref = ap_grouped(s[idx], l[idx]) if len(idx) and bool(l[idx].any()) else float('nan')
            assert (ref != ref and boot[b] != boot[b]) or abs(boot[b] - ref) < 1e-12, (boot[b], ref)
    print('metrics_tied self-test passed')
