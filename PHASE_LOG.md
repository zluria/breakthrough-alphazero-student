# Phase Log

## Formal requirements - passed

- Read all supplied assignment pages and the relevant Breakthrough rules page.
- Recorded mandatory APIs, losses, PUCT fields/formula, 10,000-game pretraining, checkpointing, and evaluation requirements in `ASSIGNMENT_REQUIREMENTS.md`.
- Reconciled the absolute Player-1 value invariant with the assignment formula.

## Phase 1: rules and trusted baselines - passed locally

- Date: 2026-08-23
- Command: bundled Python 3.12 `unittest` discovery with `src` on the import path.
- Result: 30 tests passed except one expected TensorFlow-only skip; 1.140 seconds for the complete suite after data/evaluation tests were added.
- Sanity checks: independent rule generator, exact make/unmake restoration, captures and straight restrictions, terminal goal and no-reply handling, no terminal turn switch, all legal policy round trips, symmetry involutions/commutation, and alpha-beta versus brute-force solving.
- Review: board remains a flat list; no bitboards, padding, caches, or search-specific state leaked into the rules.

## Phase 2: neural boundary and supervised sanity - pending HPC gate

- Local fake-network tests pass canonical input identity, target/value conversions, policy reflection, and illegal-action masking.
- HPC smoke job 33965 was cancelled while pending because its dedicated RTX 3070 nodes were unavailable; it consumed no runtime.
- HPC smoke job 33966 exposed a Slurm working-directory bug before project imports. The script was corrected from `SLURM_SUBMIT_DIR` to the actual job working directory.
- Corrected smoke job 33967 completed on `HPC-RTX2080s-01` in 22 seconds with exit code 0. It ran all 30 tests, including real Keras save/load on TensorFlow 2.14, and generated two dummy-MCTS games (49 positions).
- Solver-supervised learning awaits the bounded Phase 2 job.

## Phase 3: dummy PUCT and required pretraining - pending HPC gate

- Local tests pass absolute backup, player-aware selection, parent-Q FPU, immediate wins for both players, swapped values, and input immutability.
- The required 10,000-game raw dataset awaits Phase 2 success.

## Phases 4-7 - pending

No playing-strength or research claim is accepted until its configured job and paired arena complete.
