# Concise Post-AlphaZero Survey and Decisions

This survey was conducted before selecting any research extension. The canonical implementation stays unchanged until the baseline passes its tactical and arena gates.

| Work | Mechanism | Claimed benefit | Compute regime and evidence | Complexity cost here | Decision |
|---|---|---|---|---|---|
| [OLIVAW: Mastering Othello without Human Knowledge, nor a Fortune](https://arxiv.org/abs/2103.17228) | Uses progressive MCTS budgets, a recent-generation replay window, and highly explored unplayed tree positions. | Strong Othello play using commodity hardware and free cloud services. | About 50,000 games over 20 generations, evaluated against Edax, online opponents, and elite human players. | The off-trajectory positions need search-Q value targets and were not isolated in an ablation. | **Adopt progressive search and recent replay. Postpone off-trajectory targets.** |
| [KataGo: Accelerating Self-Play Learning in Go](https://arxiv.org/abs/1902.10565) | Includes playout-cap randomization: some turns receive a full search and become policy targets, while other turns use a much smaller search and are not recorded. | Reports a 1.37x time-to-strength improvement for this mechanism in its ablation. | Large Go experiments, with separate ablations for several improvements. | The mechanism needs two search limits and one sampling decision per turn, but no new network heads. | **Adopt playout-cap randomization.** Keep Go-specific auxiliary targets out of this project. |
| [Gumbel AlphaZero: Policy Improvement by Planning with Gumbel](https://openreview.net/forum?id=bERaNdoegnO) | Samples root actions without replacement using Gumbel top-k, uses sequential halving at the root, and completes action values for policy targets. | Better-founded policy improvement and stronger performance when simulations are scarce. | Evaluated on Go and chess (and Gumbel MuZero on Atari); the main attraction here is the low-simulation regime. | Replaces several easy-to-explain PUCT mechanisms and adds target construction machinery. | **Candidate after baseline.** Test only if profiling shows that low simulation counts limit training; compare at equal wall-clock time. |
| [PCZero: Efficient Learning for AlphaZero via Path Consistency](https://proceedings.mlr.press/v162/zhao22h.html) | Adds a path-consistency objective using historical trajectories and MCTS-scouted search paths. | Improves self-play efficiency and offline learning in reported Hex, Othello, and Gomoku experiments. | The paper reports 900K self-play games for its principal Hex result, far beyond the available budget. | Requires new stored path targets, an extra loss, and multiple coupled choices. | **Reject for this project.** The implementation and scale do not fit the baseline or a one-factor local screen. |
| [Train on Small, Play the Large: Scaling Up Board Games with AlphaZero and GNN](https://arxiv.org/abs/2107.08387) | Uses a graph neural network and incremental small-board training so one model can scale to larger boards. | Reports that training on smaller Othello boards can outperform a large-board model trained much longer. | Multi-day Othello experiments comparing small-to-large transfer with direct large-board training. | The graph model and shared-size transfer directly conflict with this project's explicit native 5x5 and native 8x8 CNN/checkpoint separation. | **Reject the transfer mechanism.** Use 5x5 only to guide a few 8x8 hyperparameters, never to reuse weights or derived data. |
| [Scaling Scaling Laws with Board Games](https://arxiv.org/abs/2104.03113) | Trains many low-resource AlphaZero Hex models across board sizes and compute budgets, then relates compute to attainable strength. | Makes compute/strength tradeoffs measurable instead of assuming one architecture or budget scales uniformly. | Many small-board experiments against perfect play provide controlled scaling evidence. | No single drop-in trick; applying the lesson needs disciplined measurement rather than new model code. | **Adopt the methodology.** Measure strength, uncertainty, wall-clock search, replay use, and throughput before increasing compute or model size. |

## Native 8x8 continuation

The native 8x8 baseline learned from only 192 neural self-play games but remained
far behind alpha-beta. The next run therefore adopts the well-supported
efficiency changes that leave the two-head network and PUCT value convention
unchanged:

1. batch neural leaves from 32 simultaneous games;
2. use KataGo-style full and fast turns, recording only full-search targets;
3. increase the full/fast simulation limits in three declared stages;
4. retain a recent 25,000-position replay window and train in proportion to new
   records;
5. compare the current and best checkpoints every five hours with noise-free
   search from varied paired openings.

Gumbel search, auxiliary heads, a larger network, resignation, and OLIVAW's
off-trajectory targets remain separate experiments. They are not mixed into the
main run without evidence from this simpler continuation.

