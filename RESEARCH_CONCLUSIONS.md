# Research Conclusions

This document intentionally begins nearly empty. It will contain general lessons only after the baseline and controlled comparisons finish.

## Established by correctness tests

- A single explicit neural boundary makes mover-relative inference compatible with absolute Player-1 MCTS values without sign-flipping backup.
- Player-aware PUCT selection is necessary when the tree stores one absolute value convention.
- Player-swap canonicalization can duplicate neural tensors during augmentation; exact per-position deduplication prevents accidental extra weight.

## Experimental conclusions

Pending the 5x5 baseline, equal-wall-clock arenas, native 8x8 transfer, and the limited variant screen.

