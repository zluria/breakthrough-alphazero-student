# Phase Log

## Formal requirements - passed

- Read all supplied assignment pages and the relevant Breakthrough rules page.
- Recorded mandatory APIs, losses, PUCT fields/formula, 10,000-game pretraining, checkpointing, and evaluation requirements in `ASSIGNMENT_REQUIREMENTS.md`.
- Reconciled the absolute Player-1 value invariant with the assignment formula.

## Phase 1: rules and trusted baselines - passed locally

- Date: 2026-08-23
- Command: bundled Python 3.12 `unittest` discovery with `src` on the import path.
- Current simplified-code result: 36 tests discovered; 34 pass locally and two TensorFlow-only tests skip because the bundled local runtime does not include TensorFlow. All 36 pass in Slurm job 33976 with TensorFlow 2.14 on an RTX 2080 SUPER, including Keras save/load and native 5x5/8x8 shapes.
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

## Phase 4: 5x5 AlphaZero learning - pretrained checkpoint passed; self-play learning pending

- Slurm job 33972 ran commit `a23d351cc9325c30fa47600203418b203c1f42f0` on `HPC-RTX2080s-01` and completed in 2 minutes 31 seconds with exit code 0.
- Its startup gate ran all 34 tests under TensorFlow 2.14 on an RTX 2080 SUPER. Keras save/load and separate native 75-action 5x5 / 192-action 8x8 shapes both passed.
- The 10,000-game corpus was split by whole games: 9,000 training and 1,000 validation. Symmetry conversion produced 218,565 training and 24,564 validation examples after exact mover-relative duplicates were removed.
- Across eight epochs, final training policy/value losses were 2.2304/0.7463 and validation policy/value losses were 2.2332/0.7462. Validation value loss was lowest one epoch earlier at 0.7402, so the small final increase is recorded rather than hidden.
- Tactical value-sign accuracy was 0.8333, tactical policy accuracy 1.0000, and player-swap error exactly 0.0. The previously tracked forced-defense position remained pessimistically valued (`-0.2213`) while the network still ranked its unique defense first for both colors.
- The 1,690,922-byte checkpoint is preserved locally and on the HPC at `checkpoints/pretrained-5x5.keras`, SHA-256 `2f2c4ab5e794c78b9ec769cc0083252ce836326e896573b7ba8a2e69dcf269eb`. It is eligible as a training initializer but is not yet claimed as a strong playable agent.
- Learning attempt 33973 was cancelled deliberately after its first saved metric. Iteration 0 produced 762 fresh positions, replay consumption 3.141, zero swap error, no arena failures, and a 9/12 score against its actor, but tactical value-sign accuracy declined from the pretrained 0.8333 to 0.6667.
- The attempt exposed an alarm blind spot: the loop initialized its tactical comparison after iteration 0, so it did not compare the first learned checkpoint with the pretrained actor. The cancelled weights and raw records remain quarantined under `results/phase4/attempt-33973` on the HPC; only the metric, log, and failure summary are tracked.
- Corrective action: every checkpoint is now compared directly with the actor that generated its data for both tactical value and policy accuracy, including iteration 0. A regression test reproduces the missed 0.833-to-0.667 decline. The canonical 5x5 loop reduces training from 32 to 8 batches per tranche, moving intended first-tranche example consumption from about 3.14 to 0.79 per newly added raw position without changing search, games, replay capacity, network, or optimizer.
- Corrected job 33974 passed the pre-refactor 35-test TensorFlow startup gate, then was cancelled after 4 minutes 54 seconds when the code-simplicity requirement was elevated to a hard gate. It is not a completed learning attempt and no weights from it are eligible for evaluation.
- Simplified-code job 33977 ran commit `098ed154fcfba31cac74d0fec09f784505bdcfb1`, passed all 36 startup tests, and stopped automatically after four completed iterations when tactical value accuracy declined from 0.8333 to 0.6667. It generated 256 games and 3,234 positions, presented 2,048 examples, ended at replay consumption 0.724, kept tactical policy accuracy at 1.0000 and swap error at 0.0, and scored 7/12 against its actor in the final small regression arena. Its iteration-3 weights are rejected.
- The raw labels were not viewpoint-skewed: mover-relative positive targets stayed between 52.1% and 52.4% in all four tranches, while root-value sign agreement with outcomes increased from 78.6% to 83.7%. On a fresh balanced set of 512 solver-labelled positions, pretrained value-sign accuracy was 0.9355; iterations 0-3 measured 0.9277, 0.9180, 0.8867, and 0.9219. This confirms unstable value retention rather than an absolute/relative conversion leak.
- Diagnostic jobs 33978 and 33979 replayed the same four raw tranches from the same pretrained checkpoint. Learning rates 0.001 and 0.0005 reproduced the tactical failure. Rates 0.00025 and 0.0001 retained 0.8333 tactical value, 1.0000 tactical policy, and zero swap error throughout. The 0.0001 run ended with the best solver value MSE (0.2149) and improved solver-policy accuracy from 0.3965 pretrained to 0.4316, so 0.0001 is the single measured correction for the next fresh run.

## Introductory-Python simplicity refactor - passed locally and on HPC

- Removed `from __future__`, all annotations, dataclasses, protocols, properties, class/static methods, custom decorators, callable magic methods, lambdas, `Path`, deques, and custom record/result classes from source and tests.
- Replaced moves and undo history with tuples; records, predictions, search results, games, metrics, and reports with dictionaries; symmetries with four `(swap, reflect)` tuples and ordinary functions; replay with a bounded list of `(record, iteration)` tuples.
- Removed arena agent factories and evaluator `__call__` methods. Arenas receive two ordinary agents; evaluators expose an explicit `evaluate` method.
- All 35 behavior tests retain their original coverage, and a 36th test prevents the banned language features from returning to the teaching source. All 36 passed in Slurm job 33976. The same job's two-game/four-visit end-to-end CLI smoke reproduced 49 positions and exactly four mean root visits. The preserved 10,000-game corpus still loads through the simplified dictionary format with the original 10,000 games, 121,565 positions, and 100 mean root visits.
- Job 33975 failed before importing the project because it was submitted from the HPC home directory and therefore could not find `tests/`. Job 33976 used the project directory explicitly and completed in nine seconds with exit code zero. This is recorded as an invocation failure, not a code-test failure.
- The full source was reviewed again after the refactor. No unnecessary indirection found in that pass blocks the introductory-course requirement. Learning was restarted only after this gate passed.

## Phases 5-7 - pending

No playing-strength or research claim is accepted until its configured job and paired arena complete.
