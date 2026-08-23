# AlphaZero-Style Breakthrough Engine

## Abstract

This project develops a compact AlphaZero-style engine for Breakthrough. Correctness is established in stages: a flat-list rules implementation is checked against an independent move generator; alpha-beta is checked against brute force; a two-plane mover-relative CNN is isolated behind one absolute-value conversion boundary; PUCT is tested with an assignment-prescribed dummy evaluator before being connected to the CNN. Training uses root visit counts for the policy target and final outcomes for the value target. The final study compares agents under reproducible, color-paired opening prefixes and equal wall-clock move budgets.

The final numerical summary will be inserted only from completed JSON and Slurm artifacts. At the current verified gate, the local suite discovers 34 tests (32 pass and two TensorFlow-only tests skip locally), Keras save/load has passed in the TensorFlow HPC environment, the later-added native-shape test awaits the next Slurm gate, and the revised solver-supervised diagnostic reached 90.2% held-out value accuracy, 60.0% held-out policy accuracy, 83.3% tactical value accuracy, 100% tactical policy accuracy, and exact player-swap consistency.

## 1. Assignment and scope

The formal deliverable requires a Python game class, a `GameNetwork` with policy and value heads, PUCT nodes storing `P`, `Q`, and `N`, 10,000 self-play games without neural guidance for pretraining, self-play learning with saved weights, and evaluation against a simple MCTS or humans.

The implementation begins with a native 5x5, one-row diagnostic game and later transfers the verified process to a separate native 8x8, two-row model. Five-by-five tensors are never padded to 8x8; checkpoints and derived data are never transferred between sizes.

## 2. Breakthrough rules and state

The board is an ordinary row-major Python list. Player 1 starts at the top and advances toward the last row; Player 2 starts at the bottom and advances toward row zero. A pawn may step straight forward into an empty square, step diagonally forward into an empty square, or capture an opponent by a diagonal step. Reaching the opposite edge wins. A player unable to reply loses.

Moves use a pair of compact square indices. `make_move` records only the moved square, captured piece, previous player, and previous winner. `unmake_move` restores those fields exactly. A terminal move deliberately leaves `player_to_move` equal to the mover, which prevents terminal state reporting from pretending that a losing reply turn began.

The public API contains the assignment names: `make_move`, `unmake_move`, `clone`, `encode`, `decode`, `status`, `outcome`, and `legal_moves`.

## 3. Policy and neural perspective

The policy has `board_size * board_size * 3` outputs. An action selects a mover-relative origin square and one of forward-left, forward, or forward-right. Player-2 positions and moves rotate by 180 degrees before encoding, so both players always appear to advance toward increasing row numbers. Every legal move round-trips through action encoding and decoding.

The CNN sees exactly two binary planes: the mover's pawns and the opponent's pawns. Its raw `tanh` value is mover-relative. `NeuralBoundary` is the only component allowed to perform these conversions:

```text
absolute Player-1 value = raw mover value * player_to_move
raw mover target         = absolute Player-1 target * player_to_move
```

All tree values, replay outcomes, arena scores, and reports outside that boundary are absolute Player-1 values.

## 4. Trusted baselines

The random agent samples a legal move from a seeded generator. The tactical rollout agent checks immediate wins, then captures, then falls back to a seeded random choice. Alpha-beta uses a short, explicit evaluation combining material, forward progress, and mobility. It maximizes the one absolute score for Player 1 and minimizes it for Player 2.

The rule gate compares legal moves with a separately written coordinate generator across full random games. Other checks cover captures, illegal straight captures, terminal goal moves, no-legal-reply outcomes, exact restoration, symmetry algebra, policy action round trips, and alpha-beta agreement with brute-force solving on tractable positions.

## 5. PUCT with absolute values

Each `PUCTNode` stores a prior, a visit count, an absolute value sum, and children. Its mean `Q` is the absolute value sum divided by visits. Unvisited children use their parent's `Q` as first-play urgency.

The handout's exploration term is retained:

```text
U = cpuct * P(s,a) * sqrt(N(s)) / (1 + N(s,a))
```

Because `Q` is absolute rather than mover-relative, selection maximizes:

```text
player_to_move * Q(s,a) + U(s,a)
```

Player 1 therefore prefers larger absolute `Q`, while Player 2 prefers smaller absolute `Q`. Backup adds the same absolute outcome to every node on the path and never flips a sign.

The assignment's dummy evaluator gives every legal action a uniform prior and obtains an absolute value from a seeded random rollout. It is used to generate the mandatory pretraining records before any neural PUCT self-play.

## 6. Data and training

Every raw position record preserves the absolute board, mover, legal relative actions, visit counts, priors, root value, root visits, requested and completed search effort, elapsed search time, played action, seed, and final absolute outcome. Alternative policy temperatures, symmetry choices, and value targets can therefore be reconstructed later.

The four exact game transformations combine player swap with left-right reflection. Player swap can create a neural tensor identical to the original canonical tensor; exact per-position hashing removes that duplicate so it is not accidentally overweighted.

The learned loop is synchronous:

```text
PLAY a bounded tranche with the latest checkpoint
-> reserve complete games for validation
-> add the remaining raw positions to bounded replay
-> TRAIN for a bounded number of batches
-> save model weights and optimizer state
-> compare the new checkpoint with its actor
-> use the new checkpoint for the next tranche
```

There is no candidate acceptance gate. A paired mini-arena is an alarm: if its uncertainty interval demonstrates regression, training stops with the evidence saved.

Iteration metrics include fresh games and positions, validation and replay positions, replay capacity and age, examples presented, replay consumption, training and validation losses for both heads, policy KL, throughput, tactical accuracy, player-swap consistency, a regression arena, and alarms.

## 7. Evaluation protocol

Rated search does not use Dirichlet noise. Seeded random legal prefixes create distinct openings. Every prefix is played twice with agent colors exchanged. Algorithms of different kinds receive the same wall-clock move budget. Reports retain every move sequence and record failures, score, Wilson score interval, pairwise Elo difference, Elo interval, mean game time, duplicate-game fraction, and alarms. A connected Bradley-Terry fit produces an anchored Elo table with approximate uncertainty.

The required agent set is random, alpha-beta, rollout PUCT, pretrained neural PUCT, the first learned checkpoint, and the latest learned checkpoint. Native 8x8 agents are rated separately.

## 8. Verified diagnostic history

The first solver-supervised CNN attempt is retained as a failed experiment. With 512 examples and Batch Normalization, training losses approached zero while validation value loss stayed near 1.0; tactical value accuracy was 0%. Exact color-swap consistency showed that the perspective conversion was not responsible. The failure was diagnosed as small-batch moving-statistics mismatch plus memorization and an incorrectly balanced target set.

The corrected CNN removed Batch Normalization, balanced the mover-relative labels actually seen by the value head, discarded near-zero solver evaluations, expanded the still-small dataset to 2,048 examples, used early stopping, and repaired the tactical suite. The rerun passed. It remained pessimistic on the forced-defense value, although its top policy action was the unique correct defense for both colors; that weakness is tracked rather than concealed.

## 9. Results

The 10,000-game pretraining, learned 5x5 checkpoints, paired arenas, native 8x8 transfer, and two-factor research screen are in progress. Their tables will be generated from the completed artifacts; placeholders are not reported as measurements.

## 10. Planned limited extensions

The post-AlphaZero survey supports a narrow screen, not a platform rewrite. The planned factors are modest root Dirichlet noise versus none and the 48-filter/3-block CNN versus a 24-filter/2-block CNN. Each begins from the same appropriate raw data, uses matched self-play settings, and is evaluated at equal wall-clock search time. Gumbel AlphaZero remains conditional on profiling evidence that very low simulation counts are the limiting factor.

## 11. Reproducibility and limitations

Every HPC phase records a Git commit, Slurm job ID, command, configuration, seed, inputs, outputs, and pass/fail decision. Failed reports remain in the repository. Large raw data and checkpoints are saved beside the project but ignored by Git.

The main remaining risk is statistical power: a course-scale experiment cannot establish tiny playing-strength differences. Conclusions therefore require uncertainty intervals and avoid treating an exact split or a small point estimate as evidence. The native 8x8 work deliberately validates only a few settings rather than reopening a large tuning campaign.
