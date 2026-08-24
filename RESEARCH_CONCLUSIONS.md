# Research Conclusions

This document intentionally begins nearly empty. It will contain general lessons only after the baseline and controlled comparisons finish.

## Established by correctness tests

- A single explicit neural boundary makes mover-relative inference compatible with absolute Player-1 MCTS values without sign-flipping backup.
- Player-aware PUCT selection is necessary when the tree stores one absolute value convention.
- Mover-relative canonicalization makes player-swap augmentation redundant. Training uses the original position and left-right reflection only.

## Experimental conclusions

- On this native CNN and a 512-position diagnostic set, Batch Normalization allowed training losses to collapse while inference/validation stayed near chance. Removing it, increasing the diagnostic set to 2,048 positions, and using early stopping produced a large held-out improvement. This supports omitting Batch Normalization from this baseline; it does not claim that Batch Normalization is generally harmful.
- Labels must be balanced in the viewpoint the network actually predicts. Balancing absolute Player-1 wins does not guarantee balanced mover-relative value targets.
- Tactical policy and value diagnostics reveal different failures. The revised model selected the unique forced defense but still valued that position pessimistically, so aggregate loss alone would have hidden a useful, specific weakness.
- The historical four-transformation conversion yielded almost exactly one player-swap duplicate for every retained example. This is direct evidence to omit player swapping from training augmentation rather than generate and hash duplicates.
- Hold out complete games rather than random positions from the same trajectories. The dummy-PUCT pretraining run used 9,000 training games and 1,000 validation games; its closely matched final training and validation losses are therefore evidence across unseen games, not neighboring positions from a shared game.
- Fixed random seeds made separate fresh runs reproduce the same first 64 games even though every game within each run was distinct. Fixed seeds were removed from the live training and evaluation paths.
- Binary sign accuracy on six tactical positions was too coarse to govern training. On 20 balanced exact positions and their swaps, mean signed value identified a 0.0535 decline in the unstable 0.001 run while increasing through all five 0.0001 updates, including the update rejected by the old sign test.
- Resetting the supervised Adam state before reinforcement learning did not improve the controlled five-tranche result. At learning rate 0.00025, the retained and reset conditions ended with solver-policy accuracy 45.5% and 44.3%, respectively, so the optimizer state is retained.
- Twenty synchronous 5x5 learning iterations produced a checkpoint that outscored random, alpha-beta, rollout PUCT, the pretrained network, and the first learned checkpoint in 100-game paired-opening comparisons. The measured scores were 100%, 66%, 86%, 65%, and 63%, respectively; each Wilson interval excluded 50%.
- The tactical-retention alarm and playing strength answer different questions. A continuation checkpoint rejected for falling below the fixed tactical threshold scored 54% against the accepted checkpoint, but the accepted checkpoint's 46% interval included 50%. The evidence therefore supports treating the alarm as a knowledge-retention constraint, not as a proxy for arena strength or an assertion that every rejected checkpoint plays worse.

The 5x5 baseline and equal-time arena are complete. Native 8x8 transfer and the limited variant screen remain pending.
