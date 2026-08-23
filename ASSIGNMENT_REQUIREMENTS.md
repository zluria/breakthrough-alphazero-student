# Assignment Requirements and Reconciliation

This checklist was extracted from the four course handouts dated August 11, 2026. It is the acceptance contract for this repository.

## Mandatory game work

- Choose an approved game and learn its rules. This project uses Breakthrough.
- Implement a Python game class with `__init__`, `make_move`, `unmake_move`, `clone`, `encode`, `decode`, `status`, `outcome`, and `legal_moves`.
- Include every decision-relevant feature in the encoding and give every possible move a unique policy index.
- Debug the rules with complete simulated games and edge-case tests.
- Standard Breakthrough is an 8x8 board with two starting rows (16 pawns per player). A pawn steps one square forward or diagonally forward into an empty square and captures an opponent diagonally. Reaching the opposite side wins.

## Mandatory neural and search work

- Provide a class named `GameNetwork`.
- Provide a value head estimating win probability and a policy head predicting action probabilities.
- Match neural input and policy output dimensions to the game encoding.
- Keep the network small enough to train economically.
- Use value MSE plus policy cross-entropy; weight regularization is optional.
- Save and load neural weights. This project saves full `.keras` checkpoints, including optimizer state, and also exposes weight-only methods.
- Provide `PUCTNode` and `PUCTPlayer` classes with priors `P`, mean values `Q`, and visits `N`.
- Use the PUCT exploration term `cpuct * P(s,a) * sqrt(N(s)) / (1 + N(s,a))`.
- Replace rollout evaluation with neural evaluation for learned PUCT.

## Mandatory data, training, and evaluation work

- Generate 10,000 self-play games without neural guidance for pretraining.
- Train the value head on outcomes and the policy head on root visit counts.
- Continue training by self-play and save checkpoints between sessions.
- Test the trained model against humans or a simple MCTS. This repository uses reproducible paired-opening arenas against rollout MCTS as the formal test and also reports random and alpha-beta baselines.

## Reconciliation with the project specification

The 5x5, one-row game is an explicit diagnostic stage. It does not replace the mandatory native 8x8, two-row implementation or its fresh data and model.

The handout writes PUCT as `Q + U`, which assumes all `Q` values use the maximizing player's viewpoint. This project stores every tree value in the absolute Player-1 viewpoint. The mathematically equivalent selection score is therefore `player_to_move * Q + U`: Player 1 prefers larger absolute `Q`, while Player 2 prefers smaller absolute `Q`. Backup never flips signs. Tests lock down this deliberate adaptation.

The course handout calls the neural value a win probability, while the requested invariant needs a signed value compatible with absolute Player-1 outcomes. The Keras value head uses `tanh` and trains on `-1/+1`; `(value + 1) / 2` is the corresponding mover win probability. This preserves MSE training and makes losses symmetric.

The wording of the supplied Breakthrough rules permits diagonal steps into empty squares as well as diagonal captures. The implementation follows that wording. Straight moves into occupied squares remain illegal.

