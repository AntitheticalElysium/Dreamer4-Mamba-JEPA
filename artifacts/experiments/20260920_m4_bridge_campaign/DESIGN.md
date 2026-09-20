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

## Campaign drivers — this directory

| file | function | in → out |
|---|---|---|
| `build_corpus.py` | `merge()` | archive + support-v2 → immutable merged store manifest, split/eligibility preserved |
| `fork_pool.py` | `windows()` | `broad_forks_v2/seed-*.pt` → root windows, 17 successors, second successors + `second_valid` |
| `train_joint.py` | `branch_term()`, `run()` | fresh Raw/TC from step 0; `joint_loss` + separately-declared fork term at Direct's mass |
| `bridge.py` | `fit_readout()`, `fit_heads()`, `recursive()` | joint world → trained `agent_readout` + `Heads`, H2→H16 |
| `actor.py` | `train_actor()` | heads + world → actor/critic by PMPO over imagined rollouts |
| `evaluate.py` | `main()` | actor and its own BC → real Craftax, paired seeds, bootstrap interval |

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
