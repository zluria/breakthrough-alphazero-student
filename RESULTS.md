# Results

Only completed, reproducible measurements belong in this file. Generated JSON reports contain the full game-level evidence.

## Test and diagnostic gates

| Gate | Board | Result | Evidence |
|---|---:|---|---|
| Rules/search unit tests | 5x5 and 8x8 | 30 pass, 1 TensorFlow-only skip | Local run, 2026-08-23 |
| Keras save/load | 5x5 | Pass | Slurm job 33967, TensorFlow 2.14 on RTX 2080 SUPER |
| Dummy-MCTS data smoke | 5x5 | 2 games, 49 positions, 4 visits/root | Slurm job 33967 |
| Solver-supervised tactical diagnostic | 5x5 | Pending | `results/phase2/diagnostic-report.json` |

## Playing strength

| Agent A | Agent B | Board | Move budget | Games | Score | Elo difference (95% interval) |
|---|---|---:|---:|---:|---:|---:|
| Pending | Pending | 5x5 | 0.1 s | - | - | - |

## Training progress

Iteration-level fresh positions, replay ages, examples presented, replay consumption, losses, policy KL, throughput, tactical accuracy, color-swap consistency, and alarms are written to `results/phase4/learn-5x5/metrics.jsonl`.
