# Research Conclusions

This document intentionally begins nearly empty. It will contain general lessons only after the baseline and controlled comparisons finish.

## Established by correctness tests

- A single explicit neural boundary makes mover-relative inference compatible with absolute Player-1 MCTS values without sign-flipping backup.
- Player-aware PUCT selection is necessary when the tree stores one absolute value convention.
- Player-swap canonicalization can duplicate neural tensors during augmentation; exact per-position deduplication prevents accidental extra weight.

## Experimental conclusions

- On this native CNN and a 512-position diagnostic set, Batch Normalization allowed training losses to collapse while inference/validation stayed near chance. Removing it, increasing the diagnostic set to 2,048 positions, and using early stopping produced a large held-out improvement. This supports omitting Batch Normalization from this baseline; it does not claim that Batch Normalization is generally harmful.
- Labels must be balanced in the viewpoint the network actually predicts. Balancing absolute Player-1 wins does not guarantee balanced mover-relative value targets.
- Tactical policy and value diagnostics reveal different failures. The revised model selected the unique forced defense but still valued that position pessimistically, so aggregate loss alone would have hidden a useful, specific weakness.
- Symmetry deduplication is materially necessary with mover-relative canonicalization, not just a theoretical precaution. On the 9,000-game pretraining split, four exact transformations yielded 218,565 retained examples while 218,567 exact canonical duplicates were removed; the 1,000-game validation split retained and removed 24,564 each.
- Hold out complete games rather than random positions from the same trajectories. The dummy-PUCT pretraining run used 9,000 training games and 1,000 validation games; its closely matched final training and validation losses are therefore evidence across unseen games, not neighboring positions from a shared game.
- Fixed random seeds made separate fresh runs reproduce the same first 64 games even though every game within each run was distinct. Fixed seeds were removed from the live training and evaluation paths.
- Binary sign accuracy on six tactical positions was too coarse to govern training. On 20 balanced exact positions and their swaps, mean signed value identified a 0.0535 decline in the unstable 0.001 run while increasing through all five 0.0001 updates, including the update rejected by the old sign test.
- Resetting the supervised Adam state before reinforcement learning did not improve the controlled five-tranche result. At learning rate 0.00025, the retained and reset conditions ended with solver-policy accuracy 45.5% and 44.3%, respectively, so the optimizer state is retained.

Playing-strength conclusions remain pending the 5x5 baseline, equal-wall-clock arenas, native 8x8 transfer, and limited variant screen.
