# Results

Only completed, reproducible measurements belong in this file. Generated JSON reports contain the full game-level evidence.

## Test and diagnostic gates

| Gate | Board | Result | Evidence |
|---|---:|---|---|
| Rules/search/data/evaluation unit tests | 5x5 and 8x8 | 35 discovered: 33 pass locally, 2 TensorFlow-only skips; all 35 pass on HPC | Local and Slurm runs, 2026-08-23 |
| Keras save/load | 5x5 | Pass | Slurm job 33967, TensorFlow 2.14 on RTX 2080 SUPER |
| Dummy-MCTS data smoke | 5x5 | 2 games, 49 positions, 4 visits/root | Slurm job 33967 |
| Solver-supervised diagnostic, attempt 1 | 5x5 | Rejected: memorized training data, 0.0 tactical value accuracy | Slurm job 33968 |
| Solver-supervised diagnostic, revised | 5x5 | Pass: held-out value 90.2%, policy 60.0%; tactical value 83.3%, policy 100%; swap error 0 | Slurm job 33969 |
| Required dummy-PUCT corpus | 5x5 | 10,000 games, 121,565 positions, 100 visits/root; gzip and SHA-256 verified | Slurm job 33970 |
| Dummy-PUCT neural pretraining | 5x5 | 1,000 held-out games; final val policy/value loss 2.233/0.746; tactical value 83.3%, policy 100%; swap error 0 | Slurm job 33972 |
| AlphaZero learning, attempt 1 | 5x5 | Stopped: iteration-0 tactical value accuracy fell 83.3%→66.7%; alarm baseline bug found and fixed | Slurm job 33973 |

## Playing strength

| Agent A | Agent B | Board | Move budget | Games | Score | Elo difference (95% interval) |
|---|---|---:|---:|---:|---:|---:|
| Pending | Pending | 5x5 | 0.1 s | - | - | - |

## Training progress

Iteration-level fresh positions, replay ages, examples presented, replay consumption, losses, policy KL, throughput, tactical accuracy, color-swap consistency, and alarms are written to `results/phase4/learn-5x5/metrics.jsonl`.
