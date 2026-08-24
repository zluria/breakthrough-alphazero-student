# Breakthrough AlphaZero

A compact AlphaZero-style project for Breakthrough. The code starts with tested rules and baselines, isolates the neural perspective conversion, and then adds PUCT, replay, self-play training, and paired-opening evaluation.

The implementation uses direct Python and keeps the mathematical steps visible.

This is a clean-room implementation derived from the supplied course handouts and the stated invariants. No implementation, checkpoint, training record, or infrastructure from the historical `breakthrough-zero` repository was copied.

## Program structure

The implementation uses lists, tuples, dictionaries, loops, functions, classes, NumPy, and Keras.

- A move is a two-item tuple: `(from_square, to_square)`.
- An undo-history entry is the tuple `(move, captured_piece)`.
- A self-play position, search result, metric, and report is a dictionary.
- A symmetry is the tuple `(swap_players, reflect_left_right)` and is applied by functions.
- Classes hold genuinely mutable state: the game, agents, neural network,
  replay buffer, PUCT node/player, and local GUI window.

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
- `agents.py` - random rollout, brute-force solver, and alpha-beta.
- `neural.py` - two-plane canonicalization, the perspective boundary, native Keras CNN.
- `puct.py` - rollout evaluator and PUCT tree search.
- `data.py` - gzip JSONL records with boards, visit counts, and outcomes.
- `replay.py` - a bounded list with left-right reflection augmentation.
- `diagnostics.py` - balanced alpha-beta supervision and exact tactical value checks.
- `training.py` - assignment pretraining and synchronous `PLAY -> REPLAY -> TRAIN` iterations.
- `evaluation.py` - randomized opening prefixes, color pairing, scores, and intervals.
- `cli.py` - commands used by the Slurm scripts.

## Setup and tests

Python 3.9 or newer is supported. Rules and search tests need only NumPy. Neural training uses Keras with the TensorFlow backend.

```bash
python -m pip install -e ".[train,test]"
pytest
```

The tests independently regenerate legal moves, verify exact make/unmake restoration, exhaust action round trips, check symmetry algebra, compare alpha-beta with brute force, verify 20 balanced tactical outcomes by exact solving, test neural perspective conversions, and lock down absolute-value PUCT behavior.

The tactical report scores the 20 base positions with the continuous mean of `predicted absolute value * exact absolute outcome`. Sign accuracy is a secondary summary. Color-swapped copies test the perspective invariant without counting as additional tactical observations.

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

Continue with the synchronous loop. Each trained checkpoint produces the next iteration's self-play. A direct comparison with the initial pretrained tactical score stops a clear value regression; there is no candidate-promotion arena inside training.

```bash
breakthrough-zero learn \
  --config configs/learn-5x5.json \
  --run-dir results/phase4/learn-5x5 \
  --checkpoint checkpoints/pretrained-5x5.keras \
  --report results/phase4/learn-5x5-report.json
```

Run the local graphical board to play either color against a trained checkpoint.
After each network move, the table shows the final PUCT prior, visit count,
absolute Player-1 Q, exploration bonus, and selection score for every legal move.
On Windows, `play_gui.bat` launches the installed local checkpoint directly.

```bash
breakthrough-zero gui \
  --checkpoint results/phase4/learn-5x5/checkpoints/iteration-0019-inference.h5
```

Raw records preserve the position, player, legal relative actions, visit counts, and final absolute outcome. That is enough to reconstruct the policy and value targets.

## Fair evaluation

Rated search has no Dirichlet noise. Each random opening prefix is played twice with agent colors reversed. Different search algorithms receive the same wall-clock move budget.

```bash
breakthrough-zero arena \
  --agent-a neural --a-checkpoint results/phase4/learn-5x5/checkpoints/iteration-0019.keras \
  --agent-b alphabeta \
  --openings 50 --prefix-plies 4 --move-seconds 0.1 \
  --output results/phase5/neural-v-alphabeta.json
```

Reports include game counts, failures, scores, 95% Wilson intervals, timings, and the complete games.

## HPC workflow

The `scripts/slurm` directory contains the jobs in this order:

1. `00_smoke.sbatch`
2. `10_phase2_diagnostic.sbatch`
3. `20_phase3_pretraining_data.sbatch`
4. `30_phase4_pretrain_network.sbatch`
5. `40_phase4_learn5.sbatch`
6. `50_phase5_arena5.sbatch`
7. `60_phase6_8x8.sbatch`, only after the 5x5 gate passes

For every submitted job, record the Git commit, command, configuration, inputs, outputs, and Slurm job ID in `PHASE_LOG.md`.

## Documents and results

- `ASSIGNMENT_REQUIREMENTS.md` is the formal compliance checklist and conflict record.
- `PHASE_LOG.md` records phase gates and reproducibility metadata.
- `RESULTS.md` is the human-readable evaluation and diagnostic summary.
- `RESEARCH_CONCLUSIONS.md` contains only lessons supported by completed experiments.
- `docs/REPORT.md` and `docs/TEACHERS_TIPS.md` are the sources for the final PDFs.
- `docs/LITERATURE_SURVEY.md` records mechanisms, evidence, complexity, and adopt/test/reject decisions for post-AlphaZero techniques.

Checkpoints and generated raw data are ignored by Git because they may be large. Published releases should attach only verified data/checkpoints, never an invalid playable model.
