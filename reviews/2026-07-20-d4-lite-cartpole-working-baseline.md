# D4-lite CartPole working baseline

Date: 2026-07-20

## Verdict

The project now has a positive, reproducible executed-control baseline before
Mamba or JEPA/CDP is enabled.

On 30 fresh `CartPole-v1` seeds, the frozen learned pixel policy achieved:

| Policy | Mean return | Median | Min | Max |
|---|---:|---:|---:|---:|
| Uniform random | 17.60 | 15.00 | 9 | 63 |
| Frozen learned policy | **288.70** | **274.00** | **63** | **500** |
| Privileged reference controller | 500.00 | 500.00 | 500 | 500 |

The paired learned-minus-random delta is **+271.10**, with a seeded
episode-paired bootstrap 95% interval of **[234.07, 309.77]**. The learned
policy wins all 30 matched seeds. It passes every preregistered working-baseline
condition: mean at least 100, at least twice random, confidence interval above
zero, and at least 80% paired wins.

The complete 30-seed evaluation was repeated. After removing wall-clock timing,
the raw episode rows, summaries, paired differences, confidence interval, and
gate decision are identical. The policy-training repeat also produced
bit-identical policy tensors and identical losses and held-out metrics.

## What is actually working

The working stack is:

1. the unchanged, pinned MMBench2 Transformer tokenizer and block-causal
   shortcut dynamics at the reduced `D4LiteConfig` scale;
2. official Gymnasium `CartPole-v1` dynamics and renderer;
3. a documented pixel-only 64×64 adapter;
4. a categorical port of MMBench2's attention-pooled `PolicyHeadMTP`;
5. head-only behavior cloning from deterministic balance demonstrations;
6. frozen execution from pixels and past actions.

The world and tokenizer are frozen during policy training. The policy receives
no simulator state. Privileged state is used only by the transparent
demonstration collector to choose demonstration actions, and by the separately
labelled oracle reference.

Held-out demonstration action accuracy is **87.04%** (action 0: 92.96%;
action 1: 81.00%). Three of the 30 fresh evaluation episodes reach the
environment limit of 500.

## The implementation issue that was corrected

The initial local composition did not match the upstream trainer's task-head
route. It trained reward and continuation from a second clean-latent dynamics
pass. MMBench2 trains the reward head from the agent tokens returned by the
same noised shortcut-flow forward pass. The local reward and continuation heads
now use that upstream route.

This is a material correction, not tuning. A regression test now fails if
`world_loss` makes the old second clean task-head pass. Older Crafter
checkpoints retain their historical implementation hashes and are not silently
relabelled.

Under the corrected route and task-preserving pixel adapter, shuffled actions
raise held-out flow loss by 18.16%, and the correct-action generated latent has
lower error than the wrong-action latent on average. The model is no longer
merely reconstructing a nearly static background.

## Pixel adapter

The model input is derived only from the official RGB render:

- channel 0: full-scene foreground;
- channel 1: a 256-pixel cart-localized zoom;
- channel 2: full-scene frame difference;
- foreground is distance from the white RGB background followed by a
  three-pixel max filter;
- the cart location for the zoom is detected from its rendered RGB color, not
  simulator state;
- action repeat is 2.

This adapter is deliberately registered as the first transfer suspect. Raw
64×64 CartPole frames are overwhelmingly static white background; in the first
run the reconstruction tokenizer retained cart position but largely discarded
pole angle and action effects. The adapter is not presented as a general visual
representation result.

## Imagination result

The world-model planner is also genuinely positive, but it is not the result
used to declare the baseline working.

With all 256 binary horizon-8 plans enumerated, common random numbers across
candidate denoising trajectories, and first-action receding-horizon execution,
the frozen imagined planner scores **44.87** versus **19.00** random on a
separate 30-seed set. Its paired delta is **+25.87 [15.80, 36.73]**, with
21 wins, two ties, and seven losses.

That planner misses its separately declared absolute mean-return threshold of
50. The honest conclusion is:

- learned imagination is action-sensitive and improves real behavior;
- imagination planning is not yet reliable enough to be the sole control
  baseline;
- the source-shaped frozen policy is the working positive control.

## Claim boundary

This is not a full Dreamer 4 reproduction. In particular:

- the policy is behavior-cloned from demonstrations, not trained by an
  actor/value objective in imagination;
- CartPole is a small control benchmark, not Crafter;
- the observation adapter is task-preserving and pixel-only, but not raw RGB;
- neither Mamba nor JEPA/CDP is active.

The result proves that the reduced source-pinned tokenizer, Transformer world,
agent-token readout, checkpointing, and executed-control path can support
strong learned behavior. It does not yet prove Dreamer-4-style online
reinforcement learning or solved imagination control.

## Provenance

Primary sources:

- MMBench2 commit
  `3dda6ea5bc60382ad9e1dcd1c6c3af67d69326a9`,
  `src/model.py` SHA-256
  `40f0c763e3e2a62c1dee2786cc6faffb7b08c8145068d8cf7d853ae89c893510`;
- Gymnasium `v1.2.2` commit
  `a923da5d4415a1aa5195d99341069da5e16deed7`,
  `cartpole.py` SHA-256
  `b758e3286711a2c44b0817265412c9fab1dce8b1b385e2126bc710ceedd47378`.

Data:

- world train replay:
  `75177c4f28f54378eb6a0d26ef47292ffcae2fa700c417c0ace54d7bb8e005ff`;
- world dev replay:
  `1ecaa877829913f48bbfb24d5a2c9e53a64261cfac80e11d3aa2aa3589d44747`;
- deterministic expert train replay:
  `3f3e5969c7cd12f8e9ff2f48a1a6e10e48eadd06f89083f06ee1efa493d5ef9d`;
- deterministic expert dev replay:
  `606bda40aceab2064bbf3af1caa0972b7feaeaa744db894e522910e6ff1f872d`.

Checkpoints and reports:

- tokenizer:
  `d32a8e536e1ea848956eec34ff104528feceafda1518b2a0d3cfd508c65ebd40`;
- 20,000-update world:
  `a63bb1fe31b69f8b24e68534401fd18be50b36bb06fb3fedbac7f9231c32551d`;
- 3,000-update policy:
  `c57f9bbf688e5d54cb6f50df7b2ea87110de58860a73ae901931970379ba80bd`;
- first 30-seed control report:
  `92ad77a9d9560c55e89e230c8c54e6f54fa2464c93fb7280a795d5cf9e2fc0ec`;
- repeated 30-seed control report:
  `384ad3d80aa57af82008a67d2495d8546e1203dd22152d31e09634267fb19ee4`;
- evaluation implementation:
  `c52dfc2e590838c3e1e7cb3023bd398ae9e202fafd5063a5d28fb31d3617395a`.

## Reproduction commands

The full commands are intentionally ordinary module entry points:

```bash
SDL_VIDEODRIVER=dummy .venv/bin/python -m d4_mamba_jepa.cartpole_baseline \
  train \
  --train-replay outputs/d4_mamba_jepa/cartpole_baseline_v3/train_replay.pt \
  --dev-replay outputs/d4_mamba_jepa/cartpole_baseline_v3/dev_replay.pt \
  --output-dir outputs/d4_mamba_jepa/cartpole_baseline_v3/run_seed20260722 \
  --tokenizer-steps 2000 --world-steps 20000 --batch-size 8 \
  --learning-rate 0.0003 --terminal-fraction 0.25 \
  --seed 20260722 --device cuda
```

```bash
SDL_VIDEODRIVER=dummy .venv/bin/python -m d4_mamba_jepa.cartpole_baseline \
  train-policy \
  --world-checkpoint outputs/d4_mamba_jepa/cartpole_baseline_v3/run_seed20260722/world.pt \
  --world-checkpoint-sha256 a63bb1fe31b69f8b24e68534401fd18be50b36bb06fb3fedbac7f9231c32551d \
  --train-replay outputs/d4_mamba_jepa/cartpole_baseline_v3/expert_train_replay.pt \
  --dev-replay outputs/d4_mamba_jepa/cartpole_baseline_v3/expert_dev_replay.pt \
  --output outputs/d4_mamba_jepa/cartpole_baseline_v3/policy_expert.pt \
  --steps 3000 --batch-size 16 --learning-rate 0.0003 \
  --seed 20260724 --device cuda
```

```bash
SDL_VIDEODRIVER=dummy .venv/bin/python -m d4_mamba_jepa.cartpole_baseline \
  evaluate-policy \
  --world-checkpoint outputs/d4_mamba_jepa/cartpole_baseline_v3/run_seed20260722/world.pt \
  --world-checkpoint-sha256 a63bb1fe31b69f8b24e68534401fd18be50b36bb06fb3fedbac7f9231c32551d \
  --policy-checkpoint outputs/d4_mamba_jepa/cartpole_baseline_v3/policy_expert.pt \
  --policy-checkpoint-sha256 c57f9bbf688e5d54cb6f50df7b2ea87110de58860a73ae901931970379ba80bd \
  --output outputs/d4_mamba_jepa/cartpole_baseline_v3/policy_expert_fresh30.json \
  --seeds 950000:950030 --context 8 \
  --policy-seed-base 20260724 --device cuda
```

All local architecture departures are enumerated in
`d4_mamba_jepa/DEVIATION_LEDGER.md`.
