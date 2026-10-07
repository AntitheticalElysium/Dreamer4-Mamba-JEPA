# E20 scheduling amendment — 2026-10-07

User requested seed7 results review before any second-seed run. Seed8 training
and its endpoints are held in `artifacts/eda/levers_e20_v1/schedule_review.json`, approved
seeds `[7]`. A/B seed7 are complete; C seed7 continues its original 6,000-update
budget uninterrupted. Seed7's original endpoints and parent comparisons remain
queued. No data, training/evaluation code, objective, threshold or budget changed.

The two orchestration sources changed only scheduling and completed-admission
reuse. Original sources and pin ledgers are preserved in the existing EDA root.
Admission reuse verifies its numerical source hashes and the pool/labels/reader/
proof hashes recorded in A's training contract. This avoids rerunning the immutable,
timing-bearing CUDA proof after a scheduling restart. Scientific states retain
their original contracts. The handoff waits for C's completed, hash-verified final
checkpoint before replacing the stopped coordinator; the C trainer is not stopped.

Scope: E20 is a fixed-budget adaptation pilot. Each parent already underwent 36k
six-frame updates plus 6k L16 updates; E20 loads world weights and restarts AdamW.
The L16 phase supervised 600 targets/update (3.6M total); E20 supervises 40
selected endpoints/update (240k total). This is a count of correlated target
presentations, not independent information or proof of adequate optimization.

B−A and C−B estimate effects conditional on this parent and adaptation budget.
A negative is not a falsification of the corrected recipe from initialization;
six thousand updates are not a validated convergence budget. Parent→A additionally
changes supervision granularity and class weighting: E20 preserves E19's teacher
class mass, not the M16 parent's target distribution. No pure alias claim follows.

The original two-seed adoption requirement remains unmet while seed8 is held.
One seed is a pilot; paired root-cluster intervals do not estimate training-seed
variability. Replication, longer adaptation or a matched scratch contrast are
review decisions, not automatic consequences of a positive or negative pilot.

Relevant primary methodology:
- Hafner et al., [Training Agents Inside of Scalable World Models](https://arxiv.org/html/2509.24527v1#S3.SS3): Dreamer 4 fine-tunes pretrained dynamics with additional task losses while retaining the video-prediction loss. This supports staged adaptation, but does not validate E20's health objective or 6k budget.
- Ash & Adams, [On Warm-Starting Neural Network Training](https://papers.neurips.cc/paper_files/paper/2020/file/288cd2567953f06e460a33951f55daaf-Paper.pdf): supervised warm starts can generalize worse despite similar train fit; architecture/task differ from E20.
- Nikishin et al., [The Primacy Bias in Deep Reinforcement Learning](https://proceedings.mlr.press/v162/nikishin22a.html): the same replay buffer can support a fresh agent when continued training fails; bootstrapped RL differs from this factual world.
- Lyle et al., [Understanding Plasticity in Neural Networks](https://proceedings.mlr.press/v202/lyle23b.html): directly compare fitting new targets from checkpoints and initialization. These papers do not demonstrate plasticity loss in E20.
