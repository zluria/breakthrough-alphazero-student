# Phase Log

## Formal requirements - passed

- Read all supplied assignment pages and the relevant Breakthrough rules page.
- Recorded mandatory APIs, losses, PUCT fields/formula, 10,000-game pretraining, checkpointing, and evaluation requirements in `ASSIGNMENT_REQUIREMENTS.md`.
- Reconciled the absolute Player-1 value invariant with the assignment formula.

## Phase 1: rules and trusted baselines - passed locally

- Date: 2026-08-23
- Command: bundled Python 3.12 `unittest` discovery with `src` on the import path.
- Current result: 34 tests discovered; 32 pass locally and two TensorFlow-only tests skip because the bundled local runtime does not include TensorFlow. Keras save/load passed in the Slurm smoke gate; the later-added native 5x5/8x8 shape test must run in the next TensorFlow Slurm gate.
- Sanity checks: independent rule generator, exact make/unmake restoration, captures and straight restrictions, terminal goal and no-reply handling, no terminal turn switch, all legal policy round trips, symmetry involutions/commutation, and alpha-beta versus brute-force solving.
- Review: board remains a flat list; no bitboards, padding, caches, or search-specific state leaked into the rules.

## Phase 2: neural boundary and supervised sanity - passed with one tracked weakness

- Local fake-network tests pass canonical input identity, target/value conversions, policy reflection, and illegal-action masking.
- HPC smoke job 33965 was cancelled while pending because its dedicated RTX 3070 nodes were unavailable; it consumed no runtime.
- HPC smoke job 33966 exposed a Slurm working-directory bug before project imports. The script was corrected from `SLURM_SUBMIT_DIR` to the actual job working directory.
- Corrected smoke job 33967 completed on `HPC-RTX2080s-01` in 22 seconds with exit code 0. It ran all 30 tests, including real Keras save/load on TensorFlow 2.14, and generated two dummy-MCTS games (49 positions).
- Solver-supervised job 33968 stopped after 62 seconds as designed. Training losses approached zero while validation value loss stayed near 1.0; tactical value accuracy was 0.0 and policy accuracy 0.1. Color-swap error remained exactly zero, so the perspective boundary was not the cause.
- Diagnosis: Batch Normalization moving statistics were poorly estimated from only seven small batches per epoch, and the high-capacity CNN memorized 512 examples. The data also balanced absolute labels rather than the mover-relative labels actually seen by the value head, and several tactical positions accidentally contained an unrelated immediate win.
- Corrective action: remove Batch Normalization from the small CNN, balance mover-relative labels, discard ambiguous near-zero solver evaluations, increase the still-modest diagnostic set to 2,048 examples, add early stopping and explicit held-out accuracy, accept every solver-tied optimal policy action, and repair the tactical suite to include a real forced loss and a unique forced defense.
- Revised job 33969 ran corrective commit `1833e3abb1cae4a998ea45c65b2121e584f9fd76` on `HPC-RTX2080s-01` and completed in 8 minutes 15 seconds with exit code 0.
- Held-out accuracy: value 0.9024, policy 0.6000. Tactical accuracy: value 0.8333, policy 1.0000. Mean player-swap absolute error: exactly 0.0. Early stopping selected weights from a 16-epoch run.
- Tracked weakness: the network chose the unique correct forced-defense action for both colors but assigned the resulting position a pessimistic value sign. This is not hidden; later tactical alarms must show that it improves or at least does not deteriorate.
- Review: removing Batch Normalization made the native small CNN simpler and eliminated the train/inference statistics mismatch. The diagnostic checkpoint is evidence only and is not used as assignment pretraining.

## Phase 3: dummy PUCT and required pretraining - passed

- Local tests pass absolute backup, player-aware selection, parent-Q FPU, immediate wins for both players, swapped values, and input immutability.
- Slurm job 33970 ran the required 10,000-game dummy-PUCT generation on `HPC-RTX2080s-01` from source commit `31b3f8d31f89da58cfabe80f011ed6f61f135dee` (retained on the local audit branch after the no-reply metadata rewrite).
- The job completed in 57 minutes 28 seconds with exit code 0. Its full TensorFlow environment test suite passed before generation.
- Artifact validation: gzip integrity passed; 10,000 distinct games, 121,565 positions, mean game length 12.1565, and exactly 100 mean root visits. Player 1 won 5,577 games.
- The 9,077,590-byte raw corpus is preserved locally and on the HPC at `data/raw/pretraining-5x5-10000.jsonl.gz` with SHA-256 `fd5ce5b33eb3d673fcd1b57409d0f456c770e1413a56b2b84084ce94337d5c46`. It remains Git-ignored; the report, verification record, and Slurm log are tracked.

## Phases 4-7 - pending

No playing-strength or research claim is accepted until its configured job and paired arena complete.
