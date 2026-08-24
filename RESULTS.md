# Results

Only completed, reproducible measurements belong in this file. Generated JSON reports contain the full game-level evidence.

## Test and diagnostic gates

| Gate | Board | Result | Evidence |
|---|---:|---|---|
| Pre-cleanup regression suite | 5x5 and 8x8 | 37/37 pass on HPC; 35 pass and 2 Keras tests skip locally | Slurm jobs 33976 and 33985 |
| Current implementation | 5x5 and 8x8 | 34/34 pass locally and on the HPC, including Keras save/load and native board shapes | Local run and Slurm job 33993 |
| Historical Keras save/load | 5x5 | Pass before the current architecture cleanup | Slurm job 33976, TensorFlow 2.14 on RTX 2080 SUPER |
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

| Agent A | Agent B | Board | Move budget | Games | Score (95% interval) |
|---|---|---:|---:|---:|---:|
| Accepted iteration 19 | Random | 5x5 | 0.1 s | 100 | 100% (96.3%-100.0%) |
| Accepted iteration 19 | Alpha-beta | 5x5 | 0.1 s | 100 | 66% (56.3%-74.5%) |
| Accepted iteration 19 | Rollout PUCT | 5x5 | 0.1 s | 100 | 86% (77.9%-91.5%) |
| Accepted iteration 19 | Pretrained neural PUCT | 5x5 | 0.1 s | 100 | 65% (55.3%-73.6%) |
| Accepted iteration 19 | First learned checkpoint | 5x5 | 0.1 s | 100 | 63% (53.2%-71.8%) |
| Accepted iteration 19 | Rejected continuation iteration 2 | 5x5 | 0.1 s | 100 | 46% (36.6%-55.7%) |

Each comparison in Slurm job 33994 used 50 unseeded four-ply opening prefixes, played once with each color assignment. All 600 games completed without failure. The final comparison does not establish a playing-strength difference: its interval contains 50%.

## Training progress

Slurm job 33986 completed all 20 configured 5x5 iterations in 2 hours 4 minutes. It generated 1,280 self-play games and 16,704 positions. The accepted iteration-19 checkpoint ended with tactical mean signed value 0.8841, 95% value-sign accuracy, 100% tactical policy accuracy, and zero color-swap error. Its SHA-256 is `71dccde76f7cf9274d63f084e27827563d480883ff557fa1ea147c393395355f`.

Native 8x8 smoke job 33995 completed successfully. Full from-scratch native 8x8 job 33996 is running; no 8x8 strength result is reported before it finishes.
