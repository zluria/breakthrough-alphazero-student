"""The synchronous PLAY, REPLAY, TRAIN loop.

At each iteration the current network produces self-play search targets, trains
from the replay window, and becomes the network used for the next iteration.
"""

import hashlib
import json
import os
import time

import numpy as np

from .data import (
    append_records,
    play_self_play_game,
    read_records,
    write_records,
)
from .diagnostics import evaluate_tactical_suite
from .neural import GameNetwork, NeuralBoundary, load_network
from .puct import PUCTPlayer, RolloutEvaluator
from .replay import ReplayBuffer, records_to_training_arrays


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        while True:
            chunk = stream.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def generate_pretraining_data(config, output_path):
    """Generate search targets using rollouts in place of a neural evaluator."""

    if os.path.exists(output_path):
        raise FileExistsError("refusing to overwrite raw data: " + output_path)

    evaluator = RolloutEvaluator()
    search = PUCTPlayer(
        evaluator,
        config["simulations"],
        config["cpuct"],
    )
    started = time.perf_counter()
    positions = 0
    player_1_wins = 0

    for game_index in range(config["games"]):
        records = play_self_play_game(
            search,
            config["board_size"],
            config["starting_rows"],
            game_index,
            config["temperature"],
            config["temperature_plies"],
            False,
        )
        positions += append_records(output_path, records)
        if records[0]["final_outcome"] == 1:
            player_1_wins += 1

    report = dict(config)
    report["positions"] = positions
    report["player1_wins"] = player_1_wins
    report["elapsed_s"] = time.perf_counter() - started
    report["output"] = output_path
    report["output_sha256"] = file_sha256(output_path)
    return report


def train_pretrained_network(
    data_path,
    output_path,
    epochs=8,
    batch_size=128,
    filters=48,
    residual_blocks=3,
):
    """Fit the first policy/value network to rollout-MCTS targets."""

    if os.path.exists(output_path):
        raise FileExistsError("refusing to overwrite checkpoint: " + output_path)
    records = read_records(data_path)
    if not records:
        raise ValueError("pretraining data is empty")

    board_size = records[0]["board_size"]
    for record in records:
        if record["board_size"] != board_size:
            raise ValueError("one data file must contain one board size")

    # Split whole games, rather than individual positions, so nearby states from
    # one trajectory cannot appear in both training and validation data.
    game_numbers = set()
    for record in records:
        game_numbers.add(record["game_index"])
    game_numbers = np.array(sorted(game_numbers))
    np.random.shuffle(game_numbers)
    validation_game_count = max(1, len(game_numbers) // 10)
    validation_games = set(game_numbers[:validation_game_count])

    training_records = []
    validation_records = []
    for record in records:
        if record["game_index"] in validation_games:
            validation_records.append(record)
        else:
            training_records.append(record)

    inputs, policies, values = records_to_training_arrays(training_records)
    validation = records_to_training_arrays(validation_records)
    validation_inputs = validation[0]
    validation_policies = validation[1]
    validation_values = validation[2]

    order = np.random.permutation(len(inputs))
    inputs = inputs[order]
    policies = policies[order]
    values = values[order]

    network = GameNetwork(board_size, filters, residual_blocks)
    validation_data = (
        validation_inputs,
        {"policy": validation_policies, "value": validation_values},
    )
    history = network.fit(
        inputs,
        policies,
        values,
        validation_data=validation_data,
        epochs=epochs,
        batch_size=batch_size,
        verbose=2,
    )
    network.save(output_path)

    history_values = {}
    for name in history.history:
        history_values[name] = []
        for value in history.history[name]:
            history_values[name].append(float(value))
    if board_size == 5:
        tactics = evaluate_tactical_suite(network)
    else:
        tactics = None
    return {
        "data": data_path,
        "checkpoint": output_path,
        "checkpoint_sha256": file_sha256(output_path),
        "records": len(records),
        "network": {"filters": filters, "residual_blocks": residual_blocks},
        "training_games": len(game_numbers) - validation_game_count,
        "validation_games": validation_game_count,
        "history": history_values,
        "tactical": tactics,
    }


def run_learning_loop(config, run_dir, initial_checkpoint=None):
    """Alternate self-play and replay training."""

    raw_dir = os.path.join(run_dir, "raw")
    checkpoint_dir = os.path.join(run_dir, "checkpoints")
    metric_path = os.path.join(run_dir, "metrics.jsonl")
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)
    if os.path.exists(metric_path):
        raise FileExistsError("choose a fresh run directory: " + run_dir)

    if initial_checkpoint is None:
        network = GameNetwork(config["board_size"])
    else:
        network = load_network(initial_checkpoint)
    network.model.optimizer.learning_rate.assign(config["learning_rate"])

    replay = ReplayBuffer(config["replay_capacity"])
    run_started = time.perf_counter()
    if config["board_size"] == 5:
        initial_tactics = evaluate_tactical_suite(network)
        initial_tactical_score = initial_tactics["mean_signed_value"]
    else:
        initial_tactical_score = None

    for iteration in range(config["iterations"]):
        boundary = NeuralBoundary(network)
        search = PUCTPlayer(
            boundary,
            config["simulations"],
            config["cpuct"],
            None,
            config["dirichlet_alpha"],
            config["dirichlet_fraction"],
        )
        fresh = []
        for offset in range(config["games_per_iteration"]):
            game_index = iteration * config["games_per_iteration"] + offset
            records = play_self_play_game(
                search,
                config["board_size"],
                config["starting_rows"],
                game_index,
                config["temperature"],
                config["temperature_plies"],
                True,
            )
            fresh.extend(records)

        raw_path = os.path.join(raw_dir, "iteration-%04d.jsonl.gz" % iteration)
        if os.path.exists(raw_path):
            raise FileExistsError("iteration data already exists: " + raw_path)
        write_records(raw_path, fresh)
        replay.add(fresh)

        loss_totals = {}
        records_per_batch = max(1, config["batch_size"] // 2)

        for unused_step in range(config["train_steps"]):
            sample_count = min(records_per_batch, len(replay.data))
            sampled = replay.sample(sample_count)
            inputs, policies, values = records_to_training_arrays(sampled)
            result = network.model.train_on_batch(
                inputs,
                {"policy": policies, "value": values},
                return_dict=True,
            )
            for name in result:
                if name not in loss_totals:
                    loss_totals[name] = 0.0
                loss_totals[name] += float(result[name])

        checkpoint = os.path.join(
            checkpoint_dir, "iteration-%04d.keras" % iteration
        )
        if config["board_size"] == 5:
            tactics = evaluate_tactical_suite(network)
        else:
            tactics = None
        network.save(checkpoint)

        metric = {
            "iteration": iteration,
            "new_positions": len(fresh),
            "replay_size": len(replay.data),
            "loss": loss_totals.get("loss", 0.0) / config["train_steps"],
            "policy_loss": loss_totals.get("policy_loss", 0.0)
            / config["train_steps"],
            "value_loss": loss_totals.get("value_loss", 0.0)
            / config["train_steps"],
            "checkpoint": checkpoint,
        }
        if tactics is not None:
            metric["tactical_value_score"] = tactics["mean_signed_value"]
            metric["tactical_value_accuracy"] = tactics["value_accuracy"]
            metric["color_swap_error"] = tactics[
                "mean_color_swap_absolute_error"
            ]
        with open(metric_path, "a", encoding="utf-8", newline="\n") as stream:
            stream.write(json.dumps(metric, separators=(",", ":")) + "\n")
        if tactics is not None:
            if tactics["mean_signed_value"] + 0.05 < initial_tactical_score:
                raise RuntimeError(
                    "tactical value declined from the initial %.3f to %.3f"
                    % (initial_tactical_score, tactics["mean_signed_value"])
                )

    return {
        "config": config,
        "iterations_completed": config["iterations"],
        "elapsed_s": time.perf_counter() - run_started,
        "latest_checkpoint": checkpoint,
        "latest_checkpoint_sha256": file_sha256(checkpoint),
        "metrics": metric_path,
    }
