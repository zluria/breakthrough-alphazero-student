# AlphaZero-Style Breakthrough Engine

## Abstract

This project develops a compact AlphaZero-style engine for Breakthrough. Correctness is established in stages: a flat-list rules implementation is checked against an independent move generator; alpha-beta is checked against brute force; a two-plane mover-relative CNN is isolated behind one absolute-value conversion boundary; PUCT is tested with an assignment-prescribed dummy evaluator before being connected to the CNN. Training uses root visit counts for the policy target and final outcomes for the value target. The final study compares agents under random, color-paired opening prefixes and equal wall-clock move budgets.

The final numerical summary is inserted only from completed JSON and Slurm artifacts. The rules, search, replay, tactical, perspective, and network save/load invariants are covered by the test suite. Before the current architecture cleanup, the revised solver-supervised diagnostic reached 90.2% held-out value accuracy and 60.0% held-out policy accuracy. The assignment's dummy-PUCT stage completed 10,000 games and 121,565 reconstructable positions. Neural pretraining on a whole-game split finished with validation policy/value losses of 2.233/0.746. The 20-base-position tactical suite gives that historical checkpoint a mean signed value of 0.6876, 95% sign accuracy, and zero swap error.

## 1. Assignment and scope

The formal deliverable requires a Python game class, a `GameNetwork` with policy and value heads, PUCT nodes storing `P`, `Q`, and `N`, 10,000 self-play games without neural guidance for pretraining, self-play learning with saved weights, and evaluation against a simple MCTS or humans.

The implementation begins with a native 5x5, one-row diagnostic game and later transfers the verified process to a separate native 8x8, two-row model. Five-by-five tensors are never padded to 8x8; checkpoints and derived data are never transferred between sizes.

## 2. Breakthrough rules and state

The board is an ordinary row-major Python list. Player 1 starts at the top and advances toward the last row; Player 2 starts at the bottom and advances toward row zero. A pawn may step straight forward into an empty square, step diagonally forward into an empty square, or capture an opponent by a diagonal step. Reaching the opposite edge wins. A player unable to reply loses.

Moves are two-item tuples of compact square indices. `make_move` saves only the move and captured piece. The pawn on the destination square identifies the mover during `unmake_move`; a legal move always begins from a nonterminal state, so the restored winner is `None`. A terminal move leaves `player_to_move` equal to the mover, so terminal state reporting does not begin a reply turn.

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

The random agent and rollout evaluator sample legal moves. Alpha-beta uses an evaluation combining material, forward progress, and mobility. It maximizes the one absolute score for Player 1 and minimizes it for Player 2. Without a time limit it searches the requested depth once; with a time limit it uses iterative deepening until timeout.

The rule gate compares legal moves with a separately written coordinate generator across full random games. Other checks cover captures, illegal straight captures, terminal goal moves, no-legal-reply outcomes, exact restoration, symmetry algebra, policy action round trips, and alpha-beta agreement with brute-force solving on tractable positions.

## 5. PUCT with absolute values

Each `PUCTNode` stores a prior, a parent, a visit count, an absolute mean `Q`, and children. A new child inherits its parent's current `Q` as first-play urgency. Backup follows parent pointers and updates the running mean directly.

The handout's exploration term is retained:

```text
U = cpuct * P(s,a) * sqrt(N(s)) / (1 + N(s,a))
```

Because `Q` is absolute rather than mover-relative, selection maximizes:

```text
parent_player * Q(s,a) + U(s,a)
```

Here `parent_player` is the player choosing the edge at the parent state. Player 1 therefore prefers larger absolute `Q`, while Player 2 prefers smaller absolute `Q`. Backup applies the same absolute outcome at every ancestor and never flips a sign.

The assignment's dummy evaluator gives every legal action a uniform prior and obtains an absolute value from a random rollout. It is used to generate the mandatory pretraining records before any neural PUCT self-play.

## 6. Data and training

Every raw position is a dictionary preserving the game number, board size, starting rows, absolute board, mover, legal relative actions, visit counts, and final absolute outcome. These are exactly the fields required to reconstruct neural inputs and targets. Extra fields in the existing 10,000-game corpus are ignored.

The mover-relative representation makes player-swap augmentation duplicate the original neural example. Training therefore uses only the original position and its left-right reflection. Player swapping remains in the tests for the perspective and value-sign invariant.

The learned loop is synchronous:

```text
PLAY a bounded tranche with the latest checkpoint
-> add every raw position to bounded replay
-> TRAIN for a bounded number of batches
-> measure the 20-position tactical value score
-> save the checkpoint and compact metrics
-> use the new checkpoint for the next tranche
```

There is no candidate acceptance gate. On 5x5, each iteration's tactical mean is compared directly with the initial pretrained mean, so several small degradations cannot evade the check. The standalone arena is reserved for post-training evaluation.

Iteration metrics contain the iteration, new-position count, replay size, training losses, tactical value score and sign accuracy, color-swap error, and checkpoint path.

## 7. Evaluation protocol

Rated search does not use Dirichlet noise. Random legal prefixes create distinct openings. Every prefix is played twice with agent colors exchanged. Algorithms of different kinds receive the same wall-clock move budget. Reports retain every move sequence and record failures, score, Wilson score interval, and mean game time.

The required agent set is random, alpha-beta, rollout PUCT, pretrained neural PUCT, the first learned checkpoint, and the latest learned checkpoint. Native 8x8 agents are rated separately.

## 8. Verified diagnostic history

The first solver-supervised CNN attempt is retained as a failed experiment. With 512 examples and Batch Normalization, training losses approached zero while validation value loss stayed near 1.0; tactical value accuracy was 0%. Exact color-swap consistency showed that the perspective conversion was not responsible. The failure was diagnosed as small-batch moving-statistics mismatch plus memorization and an incorrectly balanced target set.

The corrected CNN removed Batch Normalization, balanced the mover-relative labels actually seen by the value head, discarded near-zero solver evaluations, expanded the still-small dataset to 2,048 examples, used early stopping, and repaired the tactical suite. The rerun passed. It remained pessimistic on the forced-defense value, although its top policy action was the unique correct defense for both colors; that weakness is tracked rather than concealed.

## 9. Results

The required 5x5 dummy-PUCT corpus completed in Slurm job 33970: 10,000 games, 121,565 positions, mean game length 12.1565, and 5,577 Player-1 wins. Gzip integrity and the local/remote SHA-256 `fd5ce5b33eb3d673fcd1b57409d0f456c770e1413a56b2b84084ce94337d5c46` agree.

Neural pretraining completed in Slurm job 33972 using 9,000 training games and 1,000 validation games. Final validation policy/value losses were 2.2332/0.7462; tactical value-sign accuracy was 83.3%, policy accuracy 100%, and player-swap error exactly zero. The value head remained pessimistic on the unique-defense example despite selecting its correct action, so later self-play metrics continue to track that weakness. Learned 5x5 checkpoints, paired arenas, native 8x8 transfer, and the two-factor research screen remain gated on their own completed artifacts; placeholders are not reported as measurements.

The first synchronous-learning attempt was stopped after iteration 0 when tactical value-sign accuracy fell from 83.3% to 66.7%. That debugging period motivated extensive actor snapshots, per-iteration arenas, replay telemetry, and policy alarms. The final loop removes that machinery. Its one retention check compares the continuous 20-position tactical mean directly with the initial pretrained mean, preventing a sequence of small declines from escaping comparison with the starting model.

## 10. Planned limited extensions

The post-AlphaZero survey supports a narrow screen, not a platform rewrite. The planned factors are modest root Dirichlet noise versus none and the 48-filter/3-block CNN versus a 24-filter/2-block CNN. Each begins from the same appropriate raw data, uses matched self-play settings, and is evaluated at equal wall-clock search time. Gumbel AlphaZero remains conditional on profiling evidence that very low simulation counts are the limiting factor.

## 11. Reproducibility and limitations

Every HPC phase records a Git commit, Slurm job ID, command, configuration, inputs, outputs, and pass/fail decision. Failed reports remain in the repository. Large raw data and checkpoints are saved beside the project but ignored by Git.

The main remaining risk is statistical power: the available compute cannot establish tiny playing-strength differences. Conclusions therefore require uncertainty intervals and avoid treating an exact split or a small point estimate as evidence. The native 8x8 work validates only a few settings rather than reopening a large tuning campaign.
