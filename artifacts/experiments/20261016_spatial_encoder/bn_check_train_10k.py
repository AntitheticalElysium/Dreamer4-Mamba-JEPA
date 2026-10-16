"""BN gap of the Spatial patch projector at step 10,000 on TRAINING batches (Raw's corpus, JointSampler at seed + 1, i.e. the
training stream's first batches; no update). Compared with bn_check_10k.py's gap on the diagnosis frames: is the running /
batch mismatch a stale-statistics artefact (large on training data too) or a distribution shift of the evaluation frames?
Usage: bn_check_train_10k.py <spatial checkpoint> [seed] -> prints"""
import json, os, sys, dataclasses
from pathlib import Path
os.environ['EVAL_DEVICE'] = 'cpu'
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import torch
import spatial_encoder as SE
import spatial_eval as EV
from d4mj.config import load_recipe
from d4mj.data import load_joint_corpus, JointSampler


@torch.no_grad()
def main():
    ck = sys.argv[1]; seed = int(sys.argv[2]) if len(sys.argv) > 2 else 20260731
    config = dataclasses.replace(load_recipe(SE.RECIPE), seed=seed)
    episodes, _ = load_joint_corpus(SE.CORPUS, config)
    sampler = JointSampler(episodes, config, torch.Generator().manual_seed(config.seed + 1))
    ps = EV.PatchState('spatial:' + ck)
    bn = [m for m in ps.proj.modules() if isinstance(m, torch.nn.BatchNorm1d)][0]
    gaps = []
    for _ in range(3):
        b = sampler.sample()
        h = []
        for i in range(0, b.frames.shape[0], 32):
            _, _, tok, _, _ = ps.enc._hidden(b.frames[i:i + 32])
            h.append(ps.proj[0](tok.float().reshape(-1, 192)))
        h = torch.cat(h)
        gaps.append({'mean_abs_diff_of_means_over_running_std': round(float(((h.mean(0) - bn.running_mean).abs() / bn.running_var.sqrt()).mean()), 4),
                     'mean_ratio_batch_std_over_running_std': round(float((h.std(0) / bn.running_var.sqrt()).mean()), 4)})
    print(json.dumps({'train_batches': gaps}, indent=1))


if __name__ == '__main__':
    main()
