# Breakthrough AlphaZero - Student Edition

A compact AlphaZero-style project for Breakthrough, written for students who have completed introductory Python. The code starts with trusted rules and baselines, isolates the neural perspective conversion, and then builds PUCT, replay, self-play training, and paired-opening evaluation on top.

The repository is intentionally small. It uses ordinary Python lists and direct algorithms before considering optimization.

This is a clean-room implementation derived from the supplied course handouts and the stated invariants. No implementation, checkpoint, training record, or infrastructure from the historical `breakthrough-zero` repository was copied.

## Python level

The mathematical ideas are more advanced than the Python used to express them. The implementation deliberately avoids type annotations, `from __future__`, dataclasses, protocols, properties, decorators, and callable or factory machinery.

- A move is a two-item tuple: `(from_square, to_square)`.
- An undo-history entry is a tuple containing the move and the old fields.
- A self-play position, search result, metric, and report is a plain dictionary.
- A symmetry is the tuple `(swap_players, reflect_left_right)` and is applied by ordinary functions.
- Classes are reserved for the game, agents, neural network, replay buffer, and PUCT node/player, where mutable state is genuinely useful.

Students should be able to follow the control flow using lists, tuples, dictionaries, loops, functions, simple classes, NumPy, and Keras.

## The game

Player 1 begins at the top and moves toward the final row. Player 2 begins at the bottom and moves toward row zero. A pawn moves one square straight forward into an empty square or one square diagonally forward into an empty or enemy square. A diagonal move captures an enemy pawn. The first pawn to reach the far side wins; a player with no legal reply also loses.

Development begins on a native 5x5 board with one starting row. The assignment deliverable also uses a separate native 8x8 board with two starting rows. No board padding, data reuse, or checkpoint reuse crosses those sizes.

## The most important invariant

The CNN is mover-relative, but the rest of the program is absolute:

- Input is exactly two planes: `my pawns` and `opponent pawns`.
- Player-2 positions rotate 180 degrees, so the mover always advances toward larger row numbers.
- Policy has `size * size * 3` outputs: forward-left, forward, and forward-right.
- The raw value predicts for the mover.
- `NeuralBoundary` alone converts raw values to absolute Player-1 values and converts absolute targets back to mover-relative targets.
- MCTS values, `Q`, replay outcomes, arenas, and reports are absolute for Player 1.
- Backup never changes a sign. At a parent node, selection maximizes `parent_player * Q + U`: Player 1 selects larger absolute `Q`, while Player 2 selects smaller absolute `Q`.

If you change the encoding or value convention, start with `tests/test_neural.py` and `tests/test_puct.py`.

## Code map

- `game.py` - flat-list board, tuple moves, tuple undo history, legal moves, terminal rules, action mapping.
- `agents.py` - random, tactical rollout, brute-force solver, readable alpha-beta.
- `neural.py` - two-plane canonicalization, the perspective boundary, native Keras CNN.
- `puct.py` - dummy rollout evaluator, neural evaluator, PUCT tree search.
- `data.py` - plain-dictionary gzip JSONL records with boards, counts, priors, root statistics, actions, and outcomes.
- `replay.py` - a bounded list and four tuple-defined symmetries with duplicate tensors removed.
- `diagnostics.py` - balanced alpha-beta supervision and color-paired tactical checks.
- `training.py` - assignment pretraining and synchronous `PLAY -> REPLAY -> TRAIN` iterations.
- `evaluation.py` - reproducible randomized opening prefixes, color pairing, Elo differences, intervals, and alarms.
- `cli.py` - small commands used by the Slurm scripts.

## Setup and tests

Python 3.9 or newer is supported. Rules and search tests need only NumPy. Neural training additionally needs TensorFlow.

```bash
python -m pip install -e ".[train,test]"
pytest
```

The tests independently regenerate legal moves, verify exact make/unmake restoration, exhaust action round trips, check symmetry algebra, compare alpha-beta with brute force, test neural perspective conversions, and lock down absolute-value PUCT behavior.

## Training workflow

Run the supervised diagnostic before any MCTS training:

```bash
breakthrough-zero diagnostic --output-dir results/phase2
```

Generate the assignment's 10,000 dummy-network games, then train the pretraining checkpoint:

```bash
breakthrough-zero pretrain-data \
  --config configs/pretrain-5x5.json \
  --output data/raw/pretraining-5x5-10000.jsonl.gz \
  --report results/phase3/pretraining-data-report.json

breakthrough-zero pretrain-network \
  --data data/raw/pretraining-5x5-10000.jsonl.gz \
  --output checkpoints/pretrained-5x5.keras \
  --report results/phase4/pretraining-network-report.json
```

Continue with the ungated synchronous loop. The latest saved checkpoint always becomes the next actor; arenas diagnose progress but never accept or reject candidates.

```bash
breakthrough-zero learn \
  --config configs/learn-5x5.json \
  --run-dir results/phase4/learn-5x5 \
  --checkpoint checkpoints/pretrained-5x5.keras \
  --report results/phase4/learn-5x5-report.json
```

Raw records preserve the position, player, legal relative actions, visit counts, priors, root value and visit total, search effort and time, played action, seed, and final absolute outcome. This is enough to rebuild alternative policy temperatures, value targets, and symmetry schemes.

## Fair evaluation

Rated search has no Dirichlet noise. Each seeded opening prefix is played twice with agent colors reversed. Different search algorithms receive the same wall-clock move budget.

```bash
breakthrough-zero arena \
  --agent-a neural --a-checkpoint results/phase4/learn-5x5/checkpoints/latest.keras \
  --agent-b alphabeta \
  --openings 50 --prefix-plies 4 --move-seconds 0.1 \
  --output results/phase5/neural-v-alphabeta.json
```

Reports include game counts, failures, scores, Elo differences, 95% intervals, duplicate-game rates, timings, and automatic alarms.

## HPC workflow

The `scripts/slurm` directory contains deliberately direct jobs. The intended order is:

1. `00_smoke.sbatch`
2. `10_phase2_diagnostic.sbatch`
3. `20_phase3_pretraining_data.sbatch`
4. `30_phase4_pretrain_network.sbatch`
5. `40_phase4_learn5.sbatch`
6. `50_phase5_arena5.sbatch`
7. `60_phase6_8x8.sbatch`, only after the 5x5 gate passes

For every submitted job, record the Git commit, command, configuration, seed, inputs, outputs, and Slurm job ID in `PHASE_LOG.md`. Inspect startup once and the completed output once; repeated monitoring is unnecessary.

## Documents and results

- `ASSIGNMENT_REQUIREMENTS.md` is the formal compliance checklist and conflict record.
- `PHASE_LOG.md` records phase gates and reproducibility metadata.
- `RESULTS.md` is the human-readable Elo and diagnostic summary.
- `RESEARCH_CONCLUSIONS.md` contains only lessons supported by completed experiments.
- `docs/REPORT.md` and `docs/TEACHERS_TIPS.md` are the sources for the final PDFs.
- `docs/LITERATURE_SURVEY.md` records mechanisms, evidence, complexity, and adopt/test/reject decisions for post-AlphaZero techniques.

Checkpoints and generated raw data are ignored by Git because they may be large. Published releases should attach only verified data/checkpoints, never an invalid playable model.
