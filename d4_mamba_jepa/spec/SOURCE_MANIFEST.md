# Source manifest

Audit date: 2026-07-30 (rewritten after the Craftax migration; the previous
revision was audited 2026-07-20 and still registered `danijar/crafter` as the
live environment).

Every source below was checked at the recorded revision on the audit date. The
local checkouts are clean and detached at those revisions. Digests are of the
exact files this repository reads.

`reuse boundary` says what we actually take. `imported` means the bytes execute
in our process. `read reference` means a human read it and no code was copied.

## Imported at runtime — digest-verified before execution

These are the only third-party bytes that run. `source.py` verifies each digest
and hard-fails on drift.

| Component | Repository | Commit | License | File | SHA-256 |
|---|---|---|---|---|---|
| Tokenizer, block-causal dynamics, reward MTP head, `EmaRms`, symlog/two-hot | `nicklashansen/mmbench2` | `3dda6ea5bc60382ad9e1dcd1c6c3af67d69326a9` | MIT | `src/model.py` | `40f0c763e3e2a62c1dee2786cc6faffb7b08c8145068d8cf7d853ae89c893510` |
| Temporal selective state-space layer (M arm only) | `state-spaces/mamba` | `f577286d052741c35d39cd43bdc3fad27120f22c` | Apache-2.0 | installed `mamba_ssm/modules/mamba2.py` | `605e4439ff0baec8d8acaf4a191d9f0570eea9900065a065909124c472b08707` |
| Environment | `craftax` **1.6.1** (PyPI; no source checkout) | pinned by distribution version + file digests | MIT | `craftax_classic/game_logic.py` | `e5812a161b485a5edba6da0e34b7e3352550fe29ed7d0c8f66c8071ecac20755` |
| " | " | " | " | `craftax_classic/constants.py` | `5b00ec29b51f7d011bb01c98aa74e5fd6b8a7cee6ab717f61dc59d6407f6baa4` |
| " | " | " | " | `craftax_classic/renderer.py` | `e415a83a2ce6d859d960be3e2d591b2c347b77d6e91275981b22dd769390ba13` |
| SIGReg (`sigreg` anti-collapse only; not the default) | `rbalestr-lab/lejepa` | `c293d291ca87cd4fddee9d3fffe4e914c7272052` | see repo | `lejepa/multivariate/slicing.py` | `86c0fe3a714dc945ba3e23ab4093f6ed41966039f9681bd53c733e3ca5dff56b` |
| " | " | " | " | `lejepa/univariate/epps_pulley.py` | `e6554ee42de27b74d62befb5f353d4d7a4f92c6c1eade25edaa6595a6b593149` |
| " | " | " | " | `lejepa/univariate/base.py` | `08d4f115990656ea3459dacbba5991622725f9040ebc64ae8d34f4e76299eef6` |
| CartPole (retired line; loader kept for old checkpoints) | `Farama-Foundation/Gymnasium` `v1.2.2` | `a923da5d4415a1aa5195d99341069da5e16deed7` | MIT | installed `gymnasium/envs/classic_control/cartpole.py` | `b758e3286711a2c44b0817265412c9fab1dce8b1b385e2126bc710ceedd47378` |

Loading rules that make "digest-verified" true of what actually executes:

- MMBench2 `src/model.py` is executed under an isolated module name via
  `importlib.util.spec_from_file_location`; the upstream `src` directory is
  never added to `sys.path`.
- The three LeJEPA files are likewise executed under isolated names with
  synthetic parent packages. Importing them as `lejepa.*` would first run
  `lejepa/__init__.py`, `lejepa/univariate/__init__.py` and
  `lejepa/multivariate/__init__.py`, which pull in 17 further un-pinned modules
  — the executed set would exceed the verified set.
- Craftax is located through `importlib` metadata and hashed without being
  imported, so the digest check stays JAX-free.

## Read references — no code executes, nothing digest-gated at runtime

Digests recorded so a later reader can confirm they saw the same bytes.

| Purpose | Repository | Commit | File | SHA-256 |
|---|---|---|---|---|
| Self-prediction loss shape, `jumps`/t0 split, projector branches, EMA tau | `mila-iqia/spr` | `0b9dd4e7b9bbdfaecdf9a3713bf5931fb54ab0ca` | `src/models.py` | `8601ef69b24e89b9ecf7af9ce8377aa51fed212d4635270a5bc3276111bbf6fa` |
| " | " | " | `src/algos.py` | `f7dd3c924ab6c943e3aa7fb089bc0b9e1e918ce10154807dd7c8390c05aa2b07` |
| SPR shipped CLI defaults (`--local-spr`, `--classifier`, `--momentum-tau`) | " | " | `scripts/run.py` | `70f607b5bc5847ee29a5f114b671c5d95ef80e8cdf48ac3f272977e0e9ab30bc` |
| Target-encoder momentum ramp form and published endpoints | `facebookresearch/ijepa` | `52c1ae95d05f743e000e8f10a1f3a79b10cff048` | `src/train.py` | `6b5467b6663ace871ede718168b152d6759a2ec26b94bb688ae6e46c80e835aa` |
| Per-token target scoring (LayerNorm + Lp) | `facebookresearch/vjepa2` (vendored snapshot, locally modified) | shallow clone | `app/vjepa/train.py` | `abd6431069d541c96b9b31ec05fce954664c56d2d4ca6d67bb4c9e2ddd775a5f` |
| Predictor shape, `deter: 8192`, `enc_lr`/`dyn_lr`, global `dyn_deter` cosine | `fmi-basel/Dreamer-CDP` | `a851fa3e3d70b624b094ee1810ad4bb602346092` | `dreamerv3/rssm.py` | `76e4c87005fc997299470723adb392e08a59c8c6f08ceaae3db913f045001946` |
| " | " | " | `dreamerv3/configs.yaml` | `0ff44e9b80f58b2c44b56219361f1426d87d6362e9195b6dfc9413ca7345f6d9` |
| Actor/value loop shape, BC parameter tree, sliding imagination context | `edwhu/dreamer4-jax` | `8144b940d801971f12ec5633553b95001e555949` | `scripts/train_policy.py` | `d16d9e6ba220664afbb73e7f4f80056371dd6fffb3c592d2d09a7ef2b840d7d1` |
| " | " | " | `scripts/train_bc_rew_heads.py` | `5e30b694cf2935391879a6c7698a162ebc26429b643d2b7353c25f36aa4e8ef4` |
| " | " | " | `dreamer/imagination.py` | `562bab8c4bd5d465c8661022cefdeca37cce419b52e16c5d63db8ddca0b4d4ac` |
| Paper-scale configs (`N_b`, `N_z`, context `C`) | " | " | `docs/appendix.txt` | `9ebf48d26d895abd0fb46073e3e5380992c69c2ac23a4ad34eebba394a2dd5a6` |
| Official geometric-mean achievement score; `discount = 1 - dead` | `danijar/crafter` | `e04542a2159f1aad3d4c5ad52e8185717380ee3a` | `analysis/common.py` | `26d6e0751e09efae547b155e4836640192a749115774024b497db0a23108a92a` |
| Categorical value learning, `bins: 255` | `danijar/dreamerv3` | `e3f02248693a79dc8b0ebd62c93683888ddaccfe` | `dreamerv3/configs.yaml` | `9dff9c7062e3e33951cb54c6dd4b598aaf7e56e18e2cff39c812eaa797bcfcfc` |
| " | " | " | `dreamerv3/agent.py` | `adce8e4274bc098c218bf9a20fd3327545f0ad7d850b5fe328597382e91b5269` |
| Full Dreamer-4 algorithm | Dreamer 4, arXiv:2509.24527v1 | — | `third_party/papers/2509.24527v1.pdf` | `8655cce4bf12ce6210f6694f83c1a723c7acd7579214ca3ebc57c4394d0b1aeb` |

`danijar/crafter` is no longer the environment. It remains a read reference for
the scoring formula and the `1 - dead` continuation convention only.

## Cited but ABSENT — unverifiable

| Component | Cited as | Status |
|---|---|---|
| PPO expert that generated the entire training corpus | `MichaelTMatthews/Craftax_Baselines@7ce36fa05b84a2c9e758012f1e6da402e1e3a891`, `ppo_rnn.py` + `wrappers.py` | **No checkout under `third_party/`, no entry in `SOURCES.lock`.** `expert/ppo_expert.py` is a re-implementation with documented substitutions (distrax, chex, orbax, wandb, logz all replaced or dropped; target env changed to `Craftax-Classic-Symbolic-v1`). It cannot be source-diffed, and the replay manifest records no PPO training lineage. Do not describe the expert as faithful to that source. |
| SPR EMA helper | `rlpyt.models.utils.update_state_dict` (imported at `spr/src/models.py:5`) | Neither vendored nor installed. `update_jepa_target` implements the published I-JEPA/V-JEPA-2 schedule; its equivalence to SPR is asserted, not verified. |
| LeJEPA training loop | `scripts/je.py` (referenced by `scripts/launch_inet10.py`) | Not present in the pinned checkout. The convex `lambda*sigreg + (1-lambda)*invariance` form is documented in the tracked `MINIMAL.md:172-174`, so the loss form IS verifiable; the surrounding launcher is not. |

## Upstream contracts not locally reinterpreted

Changing any of these requires a registered `diff` line in `ARCHITECTURE.md`:

- MMBench2 token layout and spatial attention masks;
- tokenizer patchification, masking, encoder, bottleneck, decoder;
- symlog/symexp reward representation and two-hot cross-entropy;
- official Mamba-2 recurrence, cache tensors, and its `_no_weight_decay`
  markers on `dt_bias`/`A_log`/`D`;
- Dreamer-4 PMPO sign-balanced policy loss and reverse behavioural-prior KL;
- TD-lambda transition indexing and categorical value targets;
- Craftax action IDs, reward, termination, and achievement statistics.

Contracts currently carrying a registered deviation (see `ARCHITECTURE.md`):
shortcut signal/step embeddings and the flow output head (degenerate in the
JEPA arm, §6), and Mamba's `use_mem_eff_path` default (§7).
