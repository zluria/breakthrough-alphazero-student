"""Command-line entry points for local smoke tests and bounded HPC jobs."""

from __future__ import annotations

import argparse
from dataclasses import fields
import json
from pathlib import Path

from .agents import AlphaBetaAgent, RandomAgent
from .data import read_records, summarize_records
from .diagnostics import run_supervised_diagnostic
from .evaluation import evaluate_pair, fit_elo_table
from .neural import GameNetwork, NeuralBoundary
from .puct import NeuralEvaluator, PUCTPlayer, RolloutEvaluator
from .training import (
    LoopConfig,
    PretrainingConfig,
    generate_pretraining_data,
    run_learning_loop,
    train_pretrained_network,
)


def _load_config(path: str | Path, config_type):
    values = json.loads(Path(path).read_text(encoding="utf-8"))
    allowed = {field.name for field in fields(config_type)}
    unknown = set(values) - allowed
    if unknown:
        raise ValueError(f"unknown configuration keys: {sorted(unknown)}")
    return config_type(**values)


def _write_json(path: str | Path, value: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def _agent_factory(
    name: str,
    *,
    board_size: int,
    simulations: int,
    move_seconds: float,
    checkpoint: str | None,
    seed: int,
):
    counter = {"value": 0}
    network = GameNetwork.load(checkpoint) if name == "neural" and checkpoint else None

    def factory():
        current_seed = seed + counter["value"]
        counter["value"] += 1
        if name == "random":
            return RandomAgent(current_seed)
        if name == "alphabeta":
            return AlphaBetaAgent(depth=4, time_limit_s=move_seconds)
        if name == "rollout":
            return PUCTPlayer(
                RolloutEvaluator(current_seed),
                simulations=simulations,
                move_time_s=move_seconds,
                seed=current_seed,
            )
        if name == "neural":
            if network is None:
                raise ValueError("a neural agent requires --a-checkpoint/--b-checkpoint")
            if network.board_size != board_size:
                raise ValueError("checkpoint board size does not match the arena")
            return PUCTPlayer(
                NeuralEvaluator(NeuralBoundary(network)),
                simulations=simulations,
                move_time_s=move_seconds,
                seed=current_seed,
            )
        raise ValueError(f"unknown agent: {name}")

    return factory


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="breakthrough-zero")
    subparsers = parser.add_subparsers(dest="command", required=True)

    diagnostic = subparsers.add_parser("diagnostic", help="run solver-supervised sanity training")
    diagnostic.add_argument("--output-dir", required=True)
    diagnostic.add_argument("--examples", type=int, default=2048)
    diagnostic.add_argument("--epochs", type=int, default=24)
    diagnostic.add_argument("--batch-size", type=int, default=64)
    diagnostic.add_argument("--seed", type=int, default=20260811)

    generate = subparsers.add_parser("pretrain-data", help="generate dummy-MCTS records")
    generate.add_argument("--config", required=True)
    generate.add_argument("--output", required=True)
    generate.add_argument("--report", required=True)

    pretrain = subparsers.add_parser("pretrain-network", help="fit a CNN to MCTS records")
    pretrain.add_argument("--data", required=True)
    pretrain.add_argument("--output", required=True)
    pretrain.add_argument("--report", required=True)
    pretrain.add_argument("--epochs", type=int, default=8)
    pretrain.add_argument("--batch-size", type=int, default=128)
    pretrain.add_argument("--seed", type=int, default=20260811)
    pretrain.add_argument("--filters", type=int, default=48)
    pretrain.add_argument("--residual-blocks", type=int, default=3)

    learn = subparsers.add_parser("learn", help="run synchronous AlphaZero learning")
    learn.add_argument("--config", required=True)
    learn.add_argument("--run-dir", required=True)
    learn.add_argument("--checkpoint")
    learn.add_argument("--report", required=True)

    inspect = subparsers.add_parser("inspect-data", help="summarize raw records")
    inspect.add_argument("--data", required=True)

    arena = subparsers.add_parser("arena", help="run a paired-opening arena")
    arena.add_argument("--agent-a", choices=("random", "alphabeta", "rollout", "neural"), required=True)
    arena.add_argument("--agent-b", choices=("random", "alphabeta", "rollout", "neural"), required=True)
    arena.add_argument("--a-name")
    arena.add_argument("--b-name")
    arena.add_argument("--a-checkpoint")
    arena.add_argument("--b-checkpoint")
    arena.add_argument("--board-size", type=int, default=5)
    arena.add_argument("--starting-rows", type=int, default=1)
    arena.add_argument("--openings", type=int, default=50)
    arena.add_argument("--prefix-plies", type=int, default=4)
    arena.add_argument("--move-seconds", type=float, default=0.1)
    arena.add_argument("--simulations", type=int, default=1000000)
    arena.add_argument("--seed", type=int, default=20260811)
    arena.add_argument("--output", required=True)

    elo = subparsers.add_parser("elo-table", help="fit a connected Elo table from arena reports")
    elo.add_argument("--reports", nargs="+", required=True)
    elo.add_argument("--anchor", required=True)
    elo.add_argument("--output", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "diagnostic":
        report = run_supervised_diagnostic(
            args.output_dir,
            examples=args.examples,
            seed=args.seed,
            epochs=args.epochs,
            batch_size=args.batch_size,
        )
        print(json.dumps(report, indent=2))
        return 0
    if args.command == "pretrain-data":
        config = _load_config(args.config, PretrainingConfig)
        report = generate_pretraining_data(config, args.output)
        _write_json(args.report, report)
        print(json.dumps(report, indent=2))
        return 0
    if args.command == "pretrain-network":
        report = train_pretrained_network(
            args.data,
            args.output,
            epochs=args.epochs,
            batch_size=args.batch_size,
            seed=args.seed,
            filters=args.filters,
            residual_blocks=args.residual_blocks,
        )
        _write_json(args.report, report)
        print(json.dumps(report, indent=2))
        return 0
    if args.command == "learn":
        config = _load_config(args.config, LoopConfig)
        report = run_learning_loop(config, args.run_dir, initial_checkpoint=args.checkpoint)
        _write_json(args.report, report)
        print(json.dumps(report, indent=2))
        return 0
    if args.command == "inspect-data":
        print(json.dumps(summarize_records(read_records(args.data)), indent=2))
        return 0
    if args.command == "arena":
        factory_a = _agent_factory(
            args.agent_a,
            board_size=args.board_size,
            simulations=args.simulations,
            move_seconds=args.move_seconds,
            checkpoint=args.a_checkpoint,
            seed=args.seed,
        )
        factory_b = _agent_factory(
            args.agent_b,
            board_size=args.board_size,
            simulations=args.simulations,
            move_seconds=args.move_seconds,
            checkpoint=args.b_checkpoint,
            seed=args.seed + 1_000_000,
        )
        report = evaluate_pair(
            factory_a,
            factory_b,
            agent_a_name=args.a_name or args.agent_a,
            agent_b_name=args.b_name or args.agent_b,
            opening_count=args.openings,
            prefix_plies=args.prefix_plies,
            board_size=args.board_size,
            starting_rows=args.starting_rows,
            seed=args.seed,
        )
        _write_json(args.output, report)
        print(json.dumps({key: value for key, value in report.items() if key != "games"}, indent=2))
        return 0
    if args.command == "elo-table":
        reports = [json.loads(Path(path).read_text(encoding="utf-8")) for path in args.reports]
        table = fit_elo_table(reports, anchor=args.anchor)
        _write_json(args.output, table)
        print(json.dumps(table, indent=2))
        return 0
    raise AssertionError("unreachable command")


if __name__ == "__main__":
    raise SystemExit(main())
