"""Command-line entry points used by the Slurm scripts."""

import argparse
import json
import os

from .agents import AlphaBetaAgent, RandomAgent
from .data import read_records
from .diagnostics import run_supervised_diagnostic
from .evaluation import evaluate_pair
from .neural import NeuralBoundary, load_network
from .puct import PUCTPlayer, RolloutEvaluator
from .training import (
    generate_pretraining_data,
    run_learning_loop,
    train_pretrained_network,
)


def read_json(path):
    with open(path, "r", encoding="utf-8") as stream:
        return json.load(stream)


def write_json(path, value):
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(path, "w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2)


def build_agent(
    name,
    board_size,
    simulations,
    move_seconds,
    checkpoint,
):
    if name == "neural":
        if checkpoint is None:
            raise ValueError("a neural agent needs a checkpoint")
        network = load_network(checkpoint)
        if network.board_size != board_size:
            raise ValueError("checkpoint board size does not match arena")
        return PUCTPlayer(
            NeuralBoundary(network),
            simulations,
            1.5,
            move_seconds,
        )
    if name == "random":
        return RandomAgent()
    if name == "alphabeta":
        return AlphaBetaAgent(4, move_seconds)
    if name == "rollout":
        return PUCTPlayer(
            RolloutEvaluator(),
            simulations,
            1.5,
            move_seconds,
        )
    raise ValueError("unknown agent: " + name)


def build_parser():
    parser = argparse.ArgumentParser(prog="breakthrough-zero")
    commands = parser.add_subparsers(dest="command", required=True)

    diagnostic = commands.add_parser("diagnostic")
    diagnostic.add_argument("--output-dir", required=True)
    diagnostic.add_argument("--examples", type=int, default=2048)
    diagnostic.add_argument("--epochs", type=int, default=24)
    diagnostic.add_argument("--batch-size", type=int, default=64)

    data = commands.add_parser("pretrain-data")
    data.add_argument("--config", required=True)
    data.add_argument("--output", required=True)
    data.add_argument("--report", required=True)

    pretrain = commands.add_parser("pretrain-network")
    pretrain.add_argument("--data", required=True)
    pretrain.add_argument("--output", required=True)
    pretrain.add_argument("--report", required=True)
    pretrain.add_argument("--epochs", type=int, default=8)
    pretrain.add_argument("--batch-size", type=int, default=128)
    pretrain.add_argument("--filters", type=int, default=48)
    pretrain.add_argument("--residual-blocks", type=int, default=3)

    learn = commands.add_parser("learn")
    learn.add_argument("--config", required=True)
    learn.add_argument("--run-dir", required=True)
    learn.add_argument("--checkpoint")
    learn.add_argument("--report", required=True)

    inspect = commands.add_parser("inspect-data")
    inspect.add_argument("--data", required=True)

    arena = commands.add_parser("arena")
    choices = ("random", "alphabeta", "rollout", "neural")
    arena.add_argument("--agent-a", choices=choices, required=True)
    arena.add_argument("--agent-b", choices=choices, required=True)
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
    arena.add_argument("--output", required=True)

    return parser


def main(arguments=None):
    args = build_parser().parse_args(arguments)

    if args.command == "diagnostic":
        report = run_supervised_diagnostic(
            args.output_dir,
            args.examples,
            args.epochs,
            args.batch_size,
        )
        print(json.dumps(report, indent=2))
        return 0

    if args.command == "pretrain-data":
        report = generate_pretraining_data(read_json(args.config), args.output)
        write_json(args.report, report)
        print(json.dumps(report, indent=2))
        return 0

    if args.command == "pretrain-network":
        report = train_pretrained_network(
            args.data,
            args.output,
            args.epochs,
            args.batch_size,
            args.filters,
            args.residual_blocks,
        )
        write_json(args.report, report)
        print(json.dumps(report, indent=2))
        return 0

    if args.command == "learn":
        report = run_learning_loop(
            read_json(args.config), args.run_dir, args.checkpoint
        )
        write_json(args.report, report)
        print(json.dumps(report, indent=2))
        return 0

    if args.command == "inspect-data":
        records = read_records(args.data)
        games = set()
        for record in records:
            games.add(record["game_index"])
        report = {"positions": len(records), "games": len(games)}
        print(json.dumps(report, indent=2))
        return 0

    if args.command == "arena":
        agent_a = build_agent(
            args.agent_a,
            args.board_size,
            args.simulations,
            args.move_seconds,
            args.a_checkpoint,
        )
        agent_b = build_agent(
            args.agent_b,
            args.board_size,
            args.simulations,
            args.move_seconds,
            args.b_checkpoint,
        )
        report = evaluate_pair(
            agent_a,
            agent_b,
            args.a_name or args.agent_a,
            args.b_name or args.agent_b,
            args.openings,
            args.prefix_plies,
            args.board_size,
            args.starting_rows,
        )
        write_json(args.output, report)
        summary = dict(report)
        del summary["games"]
        print(json.dumps(summary, indent=2))
        return 0

    raise AssertionError("unknown command")


if __name__ == "__main__":
    raise SystemExit(main())
