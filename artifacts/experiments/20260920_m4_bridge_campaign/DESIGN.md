# M4 bridge — signature plan (checklist 0.3)

Written before any code, as the `d4mj` contract requires: every function named, with module,
inputs and outputs, checked against what already exists. **No new files and no new functions in
`d4mj/`.** All campaign code lives in this experiment directory, where every other experiment lives.

## The governing discovery

The agent surface touches exactly **15** attributes of the legacy `Config`:

| module | attributes |
|---|---|
| `actor_critic.py` | `gamma, lam, pmpo_alpha, prior_beta` |
| `agent.py` | `bins, d_model, mtp_leads, n_actions, symlog_limit` |
| `execution.py` | `bootstrap, device, horizon_eval, n_actions, seed` |
| `imagination.py` | `horizon` |

`LeWMConfig` already has `seed`. `d_model` is `dynamics.width` (**256 — the same as `Config.d_model`,
so `Heads` already fits LeWM's readout**), `n_actions` is `dynamics.n_actions`, `device` is
`runtime.device`. The remainder are genuine agent constants.

So `LeWMConfig` gains **properties**, not fields, making it duck-type compatible with `Config`
across the whole agent surface. Properties are not dataclass fields, so `asdict`, `recipe_dict` and
`recipe_digest` are untouched.

**Consequence: `agent.py`, `imagination.py`, `actor_critic.py` and `execution.py` need NO changes.**
`Heads`, `imagine`, `run_episode`, `evaluate`, `returns`, `actor_loss` and `critic_loss` are reused
exactly as the Direct line uses them.

## Changes inside `d4mj/` — four files, no new functions

### 1. `lewm_config.py`
```python
@dataclass(frozen=True)
class AgentSettings:          # NEW dataclass; carries M4 authorization by its presence
    horizon: int = 8;  horizon_eval: int = 10000;  bootstrap: int = 2000
    bins: int = 255;   symlog_limit: float = 20.0; mtp_leads: int = 8
    gamma: float = 0.997; lam: float = 0.95; pmpo_alpha: float = 0.5; prior_beta: float = 0.3
    batch: int = 4;  sequence: int = 16;  actor_batch: int = 16;  event_fraction: float = 0.5
    terminal_batch: int = 1; readout_steps: int; head_steps: int; actor_steps: int
    recursive_depth: int = 2          # H2, raised to H16 by the schedule
    learning_rate: float; eval_episodes: int
    fork_mass: float = 0.2            # Direct's --terminal-mass
    fork_roots: int = 4               # Direct's --terminal-roots
    fork_second_weight: float = 0.5   # Direct's --second-weight
```
`LeWMConfig.agent: AgentSettings | None = None` — **None by default**, so every sealed M0–M3 recipe
is byte-identical and `require_control` keeps blocking them.
Properties added to `LeWMConfig`: `device, n_actions, d_model, horizon, horizon_eval, bootstrap,
bins, symlog_limit, mtp_leads, gamma, lam, pmpo_alpha, prior_beta`.
`validate_recipe` validates `agent` only when present.

### 2. `config.py` — one line in `recipe_dict`
Pop `agent` when it is `None`, exactly as `stride` and the centering pair are popped. Without this
every sealed LeWM checkpoint fails `recipe identity mismatch`, which is checked *before* the
frozen-eval fallback and so cannot be recovered by re-measuring parity.

### 3. `world_api.py` — one condition in `require_control`
```python
if isinstance(self.config, LeWMConfig) and self.config.agent is None:
    raise RuntimeError("phase_gate: LeWM M0-M3 has no trained heads/readout or validated actor horizon")
```
Authorization is carried by the recipe: an M4 recipe declares `agent`, an M0–M3 recipe does not.
Sealed M0–M3 checkpoints stay blocked by construction.

### 4. `data.py` — widen one signature, add nothing
`load_joint_corpus(path | Sequence[path], config)` accepts several stores, concatenates their
episodes, and records one contract entry per source. This is what makes the merged corpus
immutable and lineage-explicit instead of a 45 GB physical copy.

**Not changed:** `lewm.py` (the caller enables `agent_readout`, no source edit needed) and
`experiments.py` (the M0–M3 CLI keeps refusing `bridge`/`actor`/`play`; campaign drivers call the
library directly). `joint_loss` is **not** touched — per the instruction, branch supervision stays
separately declared in the driver and never becomes the LeWM objective.

## Campaign drivers — this directory, as built

> **Superseded in part.** This section originally named functions that were never written
> (`fit_readout`, `fit_heads`, `recursive`, `run`, `train_actor`) and predated the flattened fork
> condition. It now describes the code that exists.

| file | key functions | in → out |
|---|---|---|
| `build_corpus.py` | `merge` | expert archive → episode store; declares the merged corpus |
| `flatten_forks.py` | `episodes_of` | `broad_forks_v2` → 255,272 ordinary episodes (**the selected condition**) |
| `fork_pool.py` | `windows` | `broad_forks_v2` → grouped root windows (grouped condition only) |
| `train_joint_pair.py` | `Forks`, `branch_term`, `_branch`, `supplement` | paired Raw/TC from update 0; **2k → canonical pair seal → G1 → budget** |
| `bridge.py` | `agent_config`, `rollout`, `dynamics_loss`, `head_group`, `dev_gate` | Phase 2 to spec: teacher+recursive MSE, 0.5/0.5 strata, group RMS, H2 → DEV gate → H16 |
| `actor.py` | `load_bridge` | PMPO actor with `_balance`/`_update`; horizon bounded by the recorded validated depth |
| `evaluate.py` | `policies_from` | actor vs its own frozen BC prior on real Craftax, identity-bound episode cache |
| `reseal.py` | `dump` | re-measures frozen-evaluation parity after in-closure edits |
| `test_campaign.py` | — | the contracts above, which the shared suite does not cover |

## Phase 2, as specified

`spec/lewm/ARCHITECTURE.md` §5 and `DECISIONS.md` define this bridge in detail, and the first
implementation did not follow them — it trained no dynamics loss at all. The objective is:

- `L_dyn = mean(teacher SE) + mean(recursive SE)`, each over its own valid positions
- heads `0.5 * prefix + 0.5 * (0.5 * observed suffix + 0.5 * generated suffix)`
- continuation `0.8 * main + 0.2 * paired_terminal`, before balancing
- independent running-RMS over dynamics / policy / reward / continuation, decay 0.99
- batch 16 main + 4 terminal, 32 frames with 128 every fourth update, true-start 0.25
- 2,000 updates at H=2, a DEV gate, then 8,000 at H=16; AdamW 1e-4, warmup 1,000 then constant

The DEV gate applies the repo's own horizon rule (S63): a generated rollout must beat the
persistence (marginal) predictor on held-out DEV rows. `validated_recursive_depth` is written from
that measurement and from nothing else, because a configured horizon is not validation.

## Verified facts this rests on

- Fork store is **raw uint8 HWC 63×63** (`frames (32,63,63,3)`, `successors (17,63,63,3)`,
  `second (17,63,63,3)`, `second_valid (17,)`) — directly encodable by the jointly trained encoder,
  no cached latents involved.
- Window alignment is `frames[-w:]` with `led_to_action[-w+1:]` — the alignment already verified in
  the 2026-09-19 arms experiment.
- Direct's fork contract: `--terminal-mass 0.2`, `--terminal-roots 4`, `--terminal-actions 17`,
  `--second-weight 0.5`, second step scored only where `second_valid`.
- `JointSampler` draws `split == "train" and uniform_eligible`; `sample_batch(mixture=True)` gives
  the relevant/uniform half-and-half and `sample_terminal_batch` the terminal stratum.
- Archive episodes are `bc_eligible=True`; support-v2 is `bc_eligible=False`. BC therefore trains on
  the archive subset while the world sees both — stated, not discovered later.
