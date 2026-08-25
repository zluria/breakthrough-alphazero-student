# The 8x8 AlphaZero training loop

The main 8x8 run continues from the accepted iteration-5 network. Its values,
MCTS statistics, replay outcomes, and arena results remain absolute Player-1
values. At a parent where Player 1 moves, PUCT exploits `Q`; at a parent where
Player 2 moves, it exploits `-Q`. Backup never changes the sign.

## Parallel self-play

Thirty-two games advance together. At each simulation, PUCT independently
selects one leaf from every active tree. The leaf boards are stacked and passed
through Keras in one call. Expansion and backup then proceed separately in each
tree. Batching changes the amount of parallel work given to the GPU; it does not
change which statistics PUCT stores or how it selects and backs up a path.

Self-play uses playout-cap randomization. On 25% of turns it performs a full
search, applies root Dirichlet noise, and saves the resulting visit counts as a
policy target. The other turns use a fast search with no root noise and are not
saved. Fast turns still choose moves in the real game, so they affect the final
outcome assigned to every saved position from that game.

The search schedule is progressive:

| Iterations | Full search | Fast search |
|---|---:|---:|
| 0-5 | 128 simulations | 16 simulations |
| 6-13 | 192 simulations | 24 simulations |
| 14 onward | 256 simulations | 32 simulations |

Moves are sampled from visit counts for the first 12 plies. Later moves choose
the largest visit count. No fixed random seeds are used.

## Replay and optimization

Five percent of complete self-play games are assigned to validation before
their positions enter replay. The other positions enter a 25,000-position FIFO
window. The six earlier neural self-play tranches seed this window and then age
out naturally; rollout-pretraining targets do not enter the AlphaZero replay.

Each training batch samples 128 raw positions. Every sampled position is either
used as recorded or reflected left-to-right, chosen independently at sampling
time. The batch therefore contains 128 distinct replay records rather than 64
records followed by their 64 reflections.

The number of optimizer steps is proportional to the new training data:

```text
steps = ceil(2 * new training positions / 128)
```

The loop keeps the checkpoint's Adam state and uses learning rate 0.00025.
Training and validation policy/value losses are logged, but playing strength is
the deciding measurement.

## Five-hour strength checks

Approximately every five elapsed hours, the current checkpoint plays the best
checkpoint seen so far. The check uses 20 distinct random, nonterminal six-ply
opening prefixes. Every prefix is played twice with the network colors reversed,
giving 40 paired games.

The opening prefixes provide diversity. Once a rated position begins, both
agents use deterministic 64-simulation PUCT: no Dirichlet noise and no move
sampling. The match report preserves every opening and complete game.

A score of at least 55% makes the current checkpoint the new reference and
resets the stall count. A lower score leaves the reference unchanged. Three
consecutive checks without a new reference stop training. This is deliberately
a repeated practical rule, not a claim that one 40-game match proves a small
strength difference.

The run also stops after 50 training hours. Every iteration writes its raw
records, checkpoint, losses, throughput, search counts, and elapsed time before
either stopping condition is considered. The final report identifies both the
latest checkpoint and the best checkpoint selected by the strength checks.

## Why this loop

Playout-cap randomization and batched inference follow the most portable
compute-efficiency lessons from KataGo and MiniZero. Progressive simulation
counts and a recent replay window are also consistent with OLIVAW's low-resource
Othello training. The loop does not add OLIVAW's off-trajectory value targets or
Gumbel AlphaZero's completed-Q machinery, because either change would alter the
learning targets and needs its own controlled experiment.

- [KataGo methods](https://github.com/lightvector/KataGo/blob/master/docs/KataGoMethods.md)
- [KataGo paper](https://arxiv.org/abs/1902.10565)
- [MiniZero](https://github.com/rlglab/minizero)
- [OLIVAW](https://arxiv.org/abs/2103.17228)
- [Gumbel AlphaZero](https://openreview.net/forum?id=bERaNdoegnO)
