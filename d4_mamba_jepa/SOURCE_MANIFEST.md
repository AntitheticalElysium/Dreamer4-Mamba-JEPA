# Source manifest

Audit date: 2026-07-20
Workspace baseline: `d1ccfa1`

Remote `HEAD` was checked on the audit date for every source below. The local
source checkouts were clean and detached at the recorded revisions.

## Reused primary implementations

| Component | Repository | Commit | License | Local evidence | Reuse boundary |
|---|---|---|---|---|---|
| D4-style tokenizer, block-causal dynamics, reward distribution, shortcut forcing | `nicklashansen/mmbench2` | `3dda6ea5bc60382ad9e1dcd1c6c3af67d69326a9` | MIT | `third_party/sources/nicklashansen__mmbench2` | Import `src/model.py` unchanged; port only the minimum trainer equations that cannot be imported safely from its flat executable |
| JEPA-style continuous deterministic prediction | `fmi-basel/Dreamer-CDP` | `a851fa3e3d70b624b094ee1810ad4bb602346092` | MIT | `third_party/sources/fmi-basel__Dreamer-CDP` | Reuse predictor, stop-gradient target, cosine loss, and learning-rate separation as source references; do not transplant the RSSM |
| Temporal selective state-space layer | `state-spaces/mamba` | `f577286d052741c35d39cd43bdc3fad27120f22c` | Apache-2.0 | `third_party/sources/state-spaces__mamba` | Call official `Mamba2`, `allocate_inference_cache`, and `step`; no local kernel fork |
| Environment | `danijar/crafter` | `e04542a2159f1aad3d4c5ad52e8185717380ee3a` | MIT | `third_party/sources/danijar__crafter` | Use official environment; wrap only observation layout and deterministic snapshot iteration |
| Full Dreamer-4 algorithm reference | `edwhu/dreamer4-jax` | `8144b940d801971f12ec5633553b95001e555949` | No license file found in inspected checkout | `third_party/sources/edwhu__dreamer4-jax` | Read-only algorithm reference; no code copied |

## Byte-level source identities

| File | SHA-256 |
|---|---|
| MMBench2 `src/model.py` | `40f0c763e3e2a62c1dee2786cc6faffb7b08c8145068d8cf7d853ae89c893510` |
| MMBench2 `src/train_tokenizer.py` | `dc97309c8c4ae8c50dab8093bebe0aba0a776904eb0590275faf03e10d013ee3` |
| MMBench2 `src/train_dynamics.py` | `43df2876c0ac01073968c37001ff6e0b3c9500e9c1507ab950171fb7ca055e5d` |
| MMBench2 `src/interactive.py` | `7affa900cfb1bd80ac1ce9501af74cf5c558a85eada445cce724649002c74cf3` |
| MMBench2 `LICENSE` | `780d8a79689622aa4eb3a96c7b1659c1da9c569e5669801c0b1aef991ebb28a0` |
| Dreamer-CDP `dreamerv3/rssm.py` | `76e4c87005fc997299470723adb392e08a59c8c6f08ceaae3db913f045001946` |
| Dreamer-CDP `dreamerv3/agent.py` | `f2110dedc641e6526206b3fbecb9ebd0b7f29f6cc8852fc72bb165c96a8e6ac8` |
| Dreamer-CDP `dreamerv3/configs.yaml` | `0ff44e9b80f58b2c44b56219361f1426d87d6362e9195b6dfc9413ca7345f6d9` |
| Dreamer-CDP `embodied/jax/heads.py` | `437641cde21e7f9e3f69b88ad8f6b7e7c22e54eec8c5b19eef6127afde1a9b3f` |
| Dreamer-CDP `embodied/jax/outs.py` | `7e80691f175c71be614f089023cce3a809e0d026c6d5ce89bf566d5f11eb3ed0` |
| Dreamer-CDP `LICENSE` | `47506bef866f9fe31951f4b86e02a3b56990a664f5a11a98ce7371e86eb5e43f` |
| Mamba `mamba_ssm/modules/mamba2.py` | `605e4439ff0baec8d8acaf4a191d9f0570eea9900065a065909124c472b08707` |
| Mamba `mamba_ssm/utils/determinism.py` | `cb6e1c30392c11200425c2a23ad9fa3d47f50b556d15e9b0caf79b7d483d6f1d` |
| Existing replay `m3_hjwm_compact/data.py` | `861cf76325cc9e5473e6fac837c1206657afc02bc4e32121e6d552055fb51929` |
| Existing checkpoint helper `m3_hjwm_compact/checkpoint.py` | `a6cb54e38811f7472e817206aed9ac6eea6d02d54432c8e91063660a36f33023` |
| Existing canonical Crafter wrapper | `a10083a7eb990f65b53955b7e79f5c2491572be8c3961d4717b3a66b309bc2ea` |
| Existing 40k transition replay | `c55257feb2f903d32806b2694dd35e049fcd48397d3525b505c9dd715c455dad` |
| Existing 20-episode held-out replay | `709e9646ce5ee1cf36ef4118f6b5d4482751a300b8c97186929af6f0271b27ad` |

The installed `.venv` copies of `mamba_ssm/modules/mamba2.py` and
`mamba_ssm/utils/determinism.py` are byte-identical to the pinned official
sources. The installed Crafter `env.py`, `engine.py`, and `worldgen.py` are
likewise byte-identical to the pinned checkout.

## Upstream code that remains authoritative

The following contracts are not locally reinterpreted unless a deviation is
entered in `DEVIATION_LEDGER.md` first:

- MMBench2 token layout and spatial attention masks;
- tokenizer patchification, masking, encoder, bottleneck, and decoder;
- shortcut signal and step embeddings;
- flow output parameterization;
- symlog/symexp reward representation;
- official Mamba-2 recurrence and cache tensors;
- Crafter action IDs, reward, termination, and achievement statistics.
