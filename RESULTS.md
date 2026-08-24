# Results

Only completed, reproducible measurements belong in this file. Generated JSON reports contain the full game-level evidence.

## Test and diagnostic gates

| Gate | Board | Result | Evidence |
|---|---:|---|---|
| Rules/search/data/evaluation/style tests | 5x5 and 8x8 | 37/37 pass on HPC; 35 pass and 2 Keras tests skip locally | Slurm jobs 33976 and 33985 |
| Keras save/load | 5x5 | Pass after simplicity rewrite | Slurm job 33976, TensorFlow 2.14 on RTX 2080 SUPER |
| Dummy-MCTS data smoke | 5x5 | 2 games, 49 positions, 4 visits/root | Slurm job 33967 |
| Solver-supervised diagnostic, attempt 1 | 5x5 | Rejected: memorized training data, 0.0 tactical value accuracy | Slurm job 33968 |
| Solver-supervised diagnostic, revised | 5x5 | Pass: held-out value 90.2%, policy 60.0%; tactical value 83.3%, policy 100%; swap error 0 | Slurm job 33969 |
| Required dummy-PUCT corpus | 5x5 | 10,000 games, 121,565 positions, 100 visits/root; gzip and SHA-256 verified | Slurm job 33970 |
| Dummy-PUCT neural pretraining | 5x5 | 1,000 held-out games; final val policy/value loss 2.233/0.746; tactical value 83.3%, policy 100%; swap error 0 | Slurm job 33972 |
| AlphaZero learning, attempt 1 | 5x5 | Stopped: iteration-0 tactical value accuracy fell 83.3%→66.7%; alarm baseline bug found and fixed | Slurm job 33973 |
| AlphaZero learning, attempt 2 | 5x5 | Stopped after iteration 3: tactical value 83.3%→66.7%; policy 100%, swap error 0, replay consumption 0.724 | Slurm job 33977 |
| Continuation learning-rate screen | 5x5 | Short screen selected `0.0001`, but fresh confirmation showed it only delayed the marginal value crossing | Slurm jobs 33978-33980 |
| AlphaZero learning, attempt 3 | 5x5 | Stopped after iteration 4: tactical value 83.3%→66.7%; policy 100%, swap error 0, replay consumption 0.740 | Slurm job 33980 |
| Replay × learning-rate screen | 5x5 | Replay rehearsal rejected; unseeded `0.00025` retained all gates and improved solver policy 39.6%→45.7%, value MSE 0.247→0.203 | Slurm jobs 33981-33982 |
| Tactical retention redesign | 5x5 | 20 balanced exact base positions + 20 swaps; mean signed value catches the unstable run's -0.0535 decline while accepting every 0.0001 step and the fresh 0.00025 step | Slurm job 33985 |
| Training-loop audit | 5x5 | Fixed randomness removed; fresh Adam rejected; 0.00025 retained; loop otherwise unchanged | Slurm jobs 33984-33985 |

## Playing strength

| Agent A | Agent B | Board | Move budget | Games | Score | Elo difference (95% interval) |
|---|---|---:|---:|---:|---:|---:|
| Pending | Pending | 5x5 | 0.1 s | - | - | - |

## Training progress

Iteration-level fresh positions, replay ages, examples presented, replay consumption, losses, policy KL, throughput, tactical accuracy, color-swap consistency, and alarms are written to `results/phase4/learn-5x5/metrics.jsonl`.
