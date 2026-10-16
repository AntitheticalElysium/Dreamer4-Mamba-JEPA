"""BatchNorm check of the Spatial patch projector at step 10,000 (R58 follow-up; CPU; evaluation only).
Zombie / cow linear decodability of the DEPLOYED state with the saved running statistics (eval mode, as everywhere) vs with the
evaluated frames' own batch statistics (train-mode BN, no update), same episode-disjoint fit / test roots (split_v2.json, first
120 fit roots and all test roots for speed), same ridge and metrics. Also the gap between running and batch statistics.
If batch statistics restore decodability, the running statistics are mismatched to this distribution (a reader artefact);
if not, the loss is in the projected state itself.
Usage: bn_check_10k.py <spatial checkpoint> -> prints

CORRECTION (R59, after Codex's BN_AUDIT.md; code unchanged): 'train_batch_stats' normalizes each root / future SEQUENCE (<= 17
correlated frames x 81 patches, including the frame's own future) as one batch, separately for fit and test. That is a different
function, not a calibration, and it cannot separate stale running statistics from cohort effects. The copied projector's BN
buffers DO update in .train() (the 'no update' wording above is wrong); the original checkpoint is unaffected."""
import json, os, sys, copy
from pathlib import Path
os.environ['EVAL_DEVICE'] = 'cpu'
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import torch
import spatial_eval as EV
import spatial_eval_v2 as EV2
import objective_audit as OA


@torch.no_grad()
def main():
    ck = sys.argv[1]
    rows = [r for f in sorted(OA.DIAG.glob('seed-*.pt')) for r in torch.load(f, weights_only=False)]
    split = json.loads(EV2.SPLIT.read_text())
    ps = EV.PatchState('spatial:' + ck)
    proj_train = copy.deepcopy(ps.proj).train()
    out = {}
    for mode in ('eval_running_stats', 'train_batch_stats'):
        feats = {}
        for part, idx in (('fit', split['fit'][:120]), ('test', split['test'])):
            S, L = [], []
            for ri in idx:
                r = rows[ri]
                for f in range(5):
                    alive = int((~r['future_dead'][f].bool()).long().cumprod(0).sum())
                    frames = torch.cat([r['context'][-1:], r['future_frames'][f][:alive]])
                    _, _, tok, _, _ = ps.enc._hidden(frames[:, None])
                    tok = tok.float().reshape(-1, 192)
                    p = (ps.proj if mode == 'eval_running_stats' else proj_train)(tok).view(-1, 81, 192)
                    S.append(p[:, :63].reshape(-1, 192)); L.append(OA.layout(torch.cat([r['root_visible'][None], r['future_visible'][f][:alive]]))[1].reshape(-1))
            feats[part] = (torch.cat(S), torch.cat(L))
        Xf, Lf = feats['fit']; Xt, Lt = feats['test']
        res = {}
        for bit, nm in ((1, 'zombie'), (2, 'cow')):
            W = EV.ridge(Xf, ((Lf & bit) > 0).float()[:, None])
            st = EV.apply(W, Xt)[:, 0]; lt = (Lt & bit) > 0
            res[nm] = {'ap': round(EV2.average_precision(st, lt), 4), 'auc': round(EV2.auc_tie(st, lt), 4)}
        out[mode] = res
    # running vs batch statistics of the BN input on the test frames
    bn = [m for m in ps.proj.modules() if isinstance(m, torch.nn.BatchNorm1d)][0]
    lin = ps.proj[0]
    toks = []
    for ri in split['test'][:40]:
        r = rows[ri]
        _, _, tok, _, _ = ps.enc._hidden(r['future_frames'][0][:, None])
        toks.append(lin(tok.float().reshape(-1, 192)))
    h = torch.cat(toks)
    out['bn_gap'] = {'mean_abs_diff_of_means_over_running_std': round(float(((h.mean(0) - bn.running_mean).abs() / bn.running_var.sqrt()).mean()), 4),
                     'mean_ratio_batch_std_over_running_std': round(float((h.std(0) / bn.running_var.sqrt()).mean()), 4),
                     'num_batches_tracked': int(bn.num_batches_tracked)}
    print(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
