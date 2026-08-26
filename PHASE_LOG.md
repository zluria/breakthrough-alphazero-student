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

## Phase 4: 5x5 AlphaZero learning - passed

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
- Confirmation job 33980 showed that conclusion was premature. It ran commit `ea76ac6`, passed all 36 tests, and stopped after five iterations when the same material-edge value moved from an already marginal `+0.0278` pretrained to `-0.0354`. The unchanged alarm rejected the checkpoint after 320 games, 3,987 positions, 2,560 examples, replay consumption 0.740, 1.0000 tactical policy, zero swap error, and a 7/12 actor-regression score. Lowering the rate to 0.0001 delayed rather than fixed the crossing.
- Diagnostic job 33981 tested 0, 250, 500, and 1,000 retained pretraining positions at rate 0.0001. Small rehearsal samples prevented the observed tactical crossing, but larger samples slowed policy improvement. More importantly, unseeded broad solver accuracy improved even when the one marginal tactical sign crossed, so the evidence does not support calling this broad catastrophic forgetting.
- Factor-separated job 33982 tested rates 0.001, 0.0005, and 0.00025 with 0, 250, and 500 old replay positions on the same five tranches. Rehearsal did not reliably stabilize the two higher rates and was rejected. The unseeded 0.00025 condition alone kept every tactical and swap gate stable across both preserved datasets, improved fresh solver-policy accuracy from 0.3965 to 0.4570, held solver value accuracy at 0.9375, and improved value MSE from 0.2475 to 0.2027. The next fresh confirmation therefore uses 0.00025 with no replay seed; the alarm remains unchanged.
- Confirmation job 33983 ran commit `3529154`, passed all 36 startup tests, and completed one clean 0.00025 iteration before being cancelled after 11 minutes 36 seconds for a diagnostic-design correction. Its one checkpoint is not rejected for a training regression, but it is not eligible for evaluation because the governing test changed mid-run.
- Review showed that the old suite's six base positions were duplicated by color swap, so one marginal base sign crossing changed reported value accuracy by 2/12, or 16.7 percentage points. Binary accuracy also treated a `+0.01` to `-0.01` crossing like a high-confidence reversal. The replacement suite has 20 explicit, exactly solved base positions, ten mover wins and ten mover losses across the same five categories, plus 20 color-swapped partners. It reports sign accuracy, category breakdowns, policy accuracy, swap error, signed-value sum, and the continuous mean of `predicted absolute value * exact absolute outcome`. The retention alarm now uses that mean with a fixed 0.05 tolerance; sign accuracy remains descriptive.
- The expanded suite and exact outcomes pass locally as part of 37 discovered tests; the two Keras tests skip locally. Slurm job 33985 passed all 37 tests under TensorFlow/Keras and evaluated the actual historical checkpoints. The pretrained mean signed value was 0.687569. The unstable 0.001 run then changed by +0.099260, +0.103210, -0.053459, and -0.047258, so the 0.05 tolerance catches its first material decline. The 0.0001 run changed by +0.009275, +0.026593, +0.019285, +0.010530, and +0.005853, confirming that the old binary rejection was a diagnostic artifact. The first fresh 0.00025 update changed by +0.024354.
- The randomness audit found 64 unique games and 64 unique eight-ply prefixes within every archived 64-game iteration. It also found that separate fresh runs reproduced the same first iteration because they reused fixed random seeds. All fixed-seed plumbing has therefore been removed from live agents, search, self-play, replay, evaluation, commands, configurations, and jobs.
- Slurm job 33984 compared retained and fresh Adam state on the same five archived data tranches without a fixed seed. The loaded supervised checkpoint carried 13,664 optimizer steps, but resetting Adam did not improve the result: at 0.00025 the retained/fresh conditions ended at solver-policy accuracy 0.4551/0.4434 and tactical mean 0.7301/0.7240. The optimizer is therefore retained. The loop itself is unchanged for the next baseline run.
- Slurm job 33986 then completed all 20 configured iterations: 1,280 self-play games, 16,704 positions, and 10,240 examples presented over 7,474.6 seconds. Iteration 19 ended with tactical mean signed value 0.884107, 95% sign accuracy, 100% tactical policy accuracy, and zero color-swap error. The accepted checkpoint is `results/phase4/learn-5x5/checkpoints/iteration-0019.keras`, SHA-256 `71dccde76f7cf9274d63f084e27827563d480883ff557fa1ea147c393395355f`.
- Continuation job 33988 preserved the accepted weights and stopped after its iteration-2 tactical mean fell to 0.809, more than 0.05 below the fixed 0.884107 baseline. Those continuation weights remain as a rejected experiment; they were not silently substituted for the accepted checkpoint.

## Introductory-Python simplicity refactor - passed locally and on HPC

- Removed `from __future__`, all annotations, dataclasses, protocols, properties, class/static methods, custom decorators, callable magic methods, lambdas, `Path`, deques, and custom record/result classes from source and tests.
- Replaced moves and undo history with tuples; records, predictions, search results, games, metrics, and reports with dictionaries; symmetries with four `(swap, reflect)` tuples and ordinary functions; replay with a bounded list of `(record, iteration)` tuples.
- Removed arena agent factories and evaluator `__call__` methods. Arenas receive two ordinary agents; evaluators expose an explicit `evaluate` method.
- All 35 behavior tests retain their original coverage, and a 36th test prevents the banned language features from returning to the teaching source. All 36 passed in Slurm job 33976. The same job's two-game/four-visit end-to-end CLI smoke reproduced 49 positions and exactly four mean root visits. The preserved 10,000-game corpus still loads through the simplified dictionary format with the original 10,000 games, 121,565 positions, and 100 mean root visits.
- Job 33975 failed before importing the project because it was submitted from the HPC home directory and therefore could not find `tests/`. Job 33976 used the project directory explicitly and completed in nine seconds with exit code zero. This is recorded as an invocation failure, not a code-test failure.
- The full source was reviewed again after the refactor. No unnecessary indirection found in that pass blocks the introductory-course requirement. Learning was restarted only after this gate passed.

## Phase 5: formal 5x5 arena - passed

- Slurm job 33994 ran exact commit `24b0f1b` and completed in 7 minutes 50 seconds with exit code zero.
- Each comparison used 50 unseeded four-ply openings, paired by color, for 100 games at 0.1 seconds per move. All 600 games completed without failure.
- The accepted iteration-19 checkpoint scored 100% against random, 66% against alpha-beta, 86% against rollout PUCT, 65% against the pretrained checkpoint, and 63% against iteration 0. The corresponding 95% Wilson intervals all exclude 50%.
- It scored 46% against the rejected continuation iteration 2, with a 95% interval of 36.6%-55.7%. This is a statistical tie. The continuation remains rejected by the predetermined tactical-retention condition, not by a claim that it plays worse.

## Phase 6: native 8x8 transfer - complete

- Slurm job 33995 passed the entire native 8x8 smoke path in 34 seconds: two rollout-PUCT games, 40 positions, a tiny network training step, checkpoint save/load, and a four-game paired arena. The 2-2 arena score is a plumbing check, not a strength measurement.
- Full Slurm job 33996 started from a new native 8x8 network and native 8x8 dummy-PUCT data. It did not reuse 5x5 positions or weights. The 10,000 games produced 421,561 positions in 24 hours 40 minutes. Six AlphaZero iterations then produced 192 neural self-play games.
- The iteration-5 network scored 60-0 against random, 60-0 against rollout PUCT, 52-8 against its pretrained initializer, and 2-58 against alpha-beta in paired 0.1-second games. It learned substantially, but 192 neural games were not enough to approach alpha-beta.

## Phase 7 - replaced by the measured scaling check

The 1,000-game pretraining subset produced 42,182 positions. With the same
eight-epoch recipe, its network lost 24-76 to the 10,000-game pretrained network
in 100 paired games. Its validation loss also began worsening after about three
epochs. This establishes that the small corpus is useful but not 90% as good
under the unchanged recipe.

## Phase 8: serious native 8x8 AlphaZero continuation - ready

- Continue from the accepted phase-6 iteration-5 checkpoint and seed replay
  with its six existing neural self-play tranches.
- Batch leaf inference across 32 games and use progressive playout-cap
  randomization: 128/16, 192/24, then 256/32 full/fast simulations.
- Generate 256 games per iteration, retain 25,000 recent training positions,
  hold out 5% of whole games, augment sampled records by a random reflection,
  and perform two sample presentations per new training position.
- Every five elapsed hours, play 20 distinct six-ply openings twice with colors
  reversed. Rated search has 64 simulations, no Dirichlet noise, and no move
  sampling. A 55% score installs a new best checkpoint. Three consecutive
  checks without a new best stop the run; 50 hours is the hard compute budget.
- Smoke job 34025 failed before importing the project because `pytest` is not
  installed in the established TensorFlow environment. The launchers now use
  the standard-library `unittest` runner; no package was installed.
- Corrected smoke job 34026 ran commit `a04818a` on an RTX 2080 SUPER, passed all
  39 tests, and completed 32 batched games plus replay training in 17 seconds.
  Its deliberately tiny 8/2-simulation setting produced 0 Player-1 wins.
- Diagnostic array 34027 separated that result from batching. Sequential 8/2
  search also produced 0-32, while batched 64-simulation search produced 19-13.
  Actual first-stage job 34029 then used 128/16 simulations and produced a
  healthy 17-15 split, 529 full-search records, 1,476 fast turns, and a complete
  update in 82.7 seconds.
- Because measured batching throughput would finish 40 iterations far below the
  compute budget, the iteration count is only a generous ceiling. The 50-hour
  limit and three consecutive failed strength checks are the operative stops.
- Serious run 34030 was submitted from exact source commit `e8d08ba` in the
  isolated HPC directory `/home/zurlu/breakthrough-alphazero-e8d08ba`. It is
  using one RTX 2080 Ti, and its startup gate passed all 39 tests under
  TensorFlow before self-play began. Its phase-6 inputs are linked into the
  deployment and only read by the loop; all new records, checkpoints, metrics,
  and matches are written
  under that deployment's `results/phase8/serious-az-8x8` directory.
- The experiment was subsequently changed to use all 50 hours even if strength
  checks are flat. Running job 34030 was left untouched. Dependency job 34052,
  from commit `c379293`, will start only if 34030 exits successfully. It exits
  immediately if the budget is already exhausted; otherwise it restores the
  latest checkpoint, the best strength-reference checkpoint, and the seven most
  recent replay tranches, then trains for exactly the remaining budget without
  the three-stall stop. Five-hour paired matches continue in the second segment.
- The first two scheduled checks installed iteration 15 after a 39-1 result
  against the phase-6 iteration-5 starting model, then installed iteration 26
  after a 36-4 result against iteration 15.
- Slurm job 34098 evaluated accepted iteration 26 against the phase-6 alpha-beta
  baseline. It used 25 distinct random six-ply openings, played each with both
  color assignments, and allowed both agents 0.1 seconds per move. All 50 games
  completed without failure. Iteration 26 scored 49-1, including 24-1 as Player
  1 and 25-0 as Player 2; its 95% Wilson interval was 89.5%-99.6%.
- Job 34098 ran on an L40S, whereas the phase-6 2-58 result was obtained on an
  RTX 2080-class node. The arena software, time allowance, board settings, and
  paired-opening design were unchanged, but the score difference is not a
  hardware-controlled estimate of the improvement's exact size.
