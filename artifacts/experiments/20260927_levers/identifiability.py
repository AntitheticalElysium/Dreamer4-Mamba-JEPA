"""E4a. The exact ceiling on Delta-JEPA's action decoding in Craftax-Classic: how often is the executed action
identifiable from the frame pair (o_t, o_{t+1}) at all?

Diagnostic futures (1,002 roots), all 17 actions from each root under the SAME environment key (key 0), so the only
difference between successors is the action. Actions whose successor frames are pixel-identical cannot be told apart
by any function of (o_t, o_{t+1}); LDAD sees only z_{t+1} - z_t, a function of that pair. Bayes accuracy of the best
possible decoder, per root = sum over identical-frame classes of max prior within the class; averaged over roots:
  uniform   actions uniform (the fan)
  corpus    the corpus action frequencies (interface pool, every action taken)
Also: which action groups collide most often, and the share of transitions whose action is uniquely identified.
"""
import json
import sys
from collections import Counter
from pathlib import Path

import torch

ROOT = Path("/home/antithetical/EPITA/PERSO/DynamicHorizons-Mamba-JEPA")
HERE = Path(__file__).parent
sys.path.insert(0, str(ROOT / "artifacts/experiments/20260926_diagnosis"))
from rollouts import load_roots  # noqa: E402

NAMES = ["NOOP", "LEFT", "RIGHT", "UP", "DOWN", "DO", "SLEEP", "P_STONE", "P_TABLE", "P_FURNACE", "P_PLANT",
         "M_WPICK", "M_SPICK", "M_IPICK", "M_WSWORD", "M_SSWORD", "M_ISWORD"]


def main():
    pool = torch.load(ROOT / "artifacts/eda/interface_pool_v1/pool.pt", weights_only=False, mmap=True)
    a = pool["actions"].flatten()
    prior = torch.bincount(a[(a >= 0) & (a < 17)], minlength=17).double()
    prior /= prior.sum()
    uni, cor, unique, collide = [], [], [], Counter()
    for r in load_roots():
        f = r["onestep_frames"][0].flatten(1)                       # [17, pixels], key 0
        classes = {}
        for i in range(17):
            classes.setdefault(f[i].numpy().tobytes(), []).append(i)
        groups = list(classes.values())
        uni.append(len(groups) / 17)
        cor.append(sum(float(prior[g].max()) for g in groups))
        unique.append(sum(float(prior[g].sum()) for g in groups if len(g) == 1))
        for g in groups:
            if len(g) > 1:
                collide[" ".join(NAMES[i] for i in g)] += 1
    out = {"roots": len(uni), "bayes_accuracy_uniform_actions": sum(uni) / len(uni),
           "bayes_accuracy_corpus_prior": sum(cor) / len(cor),
           "share_uniquely_identified_corpus_prior": sum(unique) / len(unique),
           "corpus_prior": {NAMES[i]: round(float(prior[i]), 4) for i in range(17)},
           "most_common_collision_groups": collide.most_common(8)}
    (HERE / "identifiability.json").write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
