"""The synchronous PLAY, REPLAY, TRAIN loop.

At each iteration the current network produces self-play search targets, trains
from the replay window, and becomes the network used for the next iteration.
"""

import hashlib
import json
import math
import os
import random
import time

import numpy as np

from .data import (
    append_records,
    play_parallel_self_play_games,
    play_self_play_game,
    read_records,
    write_records,
)
from .diagnostics import evaluate_tactical_suite
from .evaluation import evaluate_pair
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


def search_limits(config, iteration):
    """Return the full and fast search sizes for one training iteration."""

    if "search_schedule" not in config:
        simulations = config["simulations"]
        return simulations, simulations
    for stage in config["search_schedule"]:
        if iteration < stage["until_iteration"]:
            return stage["full_simulations"], stage["fast_simulations"]
    raise ValueError("search schedule does not cover every iteration")


def split_training_and_validation(records, validation_fraction):
    if validation_fraction <= 0:
        return records, []

    game_numbers = set()
    for record in records:
        game_numbers.add(record["game_index"])
    game_numbers = list(game_numbers)
    random.shuffle(game_numbers)
    validation_game_count = max(
        1,
        int(round(len(game_numbers) * validation_fraction)),
    )
    validation_games = set(game_numbers[:validation_game_count])

    training_records = []
    validation_records = []
    for record in records:
        if record["game_index"] in validation_games:
            record["validation"] = True
            validation_records.append(record)
        else:
            record["validation"] = False
            training_records.append(record)
    return training_records, validation_records


def run_strength_match(
    config,
    network,
    reference_network,
    current_name,
    reference_name,
):
    simulations = config["strength_simulations"]
    current = PUCTPlayer(
        NeuralBoundary(network),
        simulations,
        config["cpuct"],
    )
    reference = PUCTPlayer(
        NeuralBoundary(reference_network),
        simulations,
        config["cpuct"],
    )
    return evaluate_pair(
        current,
        reference,
        current_name,
        reference_name,
        config["strength_openings"],
        config["strength_prefix_plies"],
        config["board_size"],
        config["starting_rows"],
    )


def run_learning_loop(config, run_dir, initial_checkpoint=None):
    """Run self-play, replay training, and periodic strength checks."""

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
    next_game_index = 0
    replay_seed_records = 0
    for path in config.get("replay_seed_paths", []):
        old_records = read_records(path)
        replay.add(old_records)
        replay_seed_records += len(old_records)
        for record in old_records:
            next_game_index = max(next_game_index, record["game_index"] + 1)

    validation_records = []
    run_started = time.perf_counter()
    if config["board_size"] == 5:
        initial_tactics = evaluate_tactical_suite(network)
        initial_tactical_score = initial_tactics["mean_signed_value"]
    else:
        initial_tactical_score = None

    strength_interval_s = config.get("strength_check_hours", 0) * 3600
    next_strength_match_s = strength_interval_s
    strength_stalls = 0
    matches_completed = 0
    best_checkpoint = initial_checkpoint
    best_network = None
    if strength_interval_s > 0:
        if initial_checkpoint is None:
            best_checkpoint = os.path.join(run_dir, "starting-network.keras")
            network.save(best_checkpoint)
        best_network = load_network(best_checkpoint)

    max_seconds = config.get("max_hours", 0) * 3600
    stop_reason = "iterations completed"
    completed_iterations = 0
    latest_checkpoint = initial_checkpoint

    for iteration in range(config["iterations"]):
        if max_seconds > 0:
            if time.perf_counter() - run_started >= max_seconds:
                stop_reason = "time budget reached"
                break
        iteration_started = time.perf_counter()
        full_simulations, fast_simulations = search_limits(config, iteration)
        boundary = NeuralBoundary(network)
        search = PUCTPlayer(
            boundary,
            full_simulations,
            config["cpuct"],
            None,
            config["dirichlet_alpha"],
            config["dirichlet_fraction"],
        )
        fresh = []
        player_1_wins = 0
        game_length_total = 0.0
        full_searches = 0
        fast_searches = 0
        games_finished = 0
        parallel_games = config.get("parallel_games", 1)
        while games_finished < config["games_per_iteration"]:
            games_in_batch = min(
                parallel_games,
                config["games_per_iteration"] - games_finished,
            )
            self_play = play_parallel_self_play_games(
                search,
                games_in_batch,
                next_game_index + games_finished,
                config["board_size"],
                config["starting_rows"],
                full_simulations,
                fast_simulations,
                config.get("full_search_probability", 1.0),
                config["temperature"],
                config["temperature_plies"],
            )
            fresh.extend(self_play["records"])
            games_finished += self_play["games"]
            player_1_wins += self_play["player1_wins"]
            game_length_total += (
                self_play["mean_game_length"] * self_play["games"]
            )
            full_searches += self_play["full_searches"]
            fast_searches += self_play["fast_searches"]
            print(
                "iteration %d: %d/%d self-play games"
                % (iteration, games_finished, config["games_per_iteration"]),
                flush=True,
            )
        next_game_index += config["games_per_iteration"]

        training_fresh, validation_fresh = split_training_and_validation(
            fresh,
            config.get("validation_fraction", 0),
        )

        raw_path = os.path.join(raw_dir, "iteration-%04d.jsonl.gz" % iteration)
        if os.path.exists(raw_path):
            raise FileExistsError("iteration data already exists: " + raw_path)
        write_records(raw_path, fresh)
        replay.add(training_fresh)
        validation_records.extend(validation_fresh)
        validation_capacity = config.get("validation_capacity", 5000)
        if len(validation_records) > validation_capacity:
            validation_records = validation_records[-validation_capacity:]

        loss_totals = {}
        random_reflection = config.get("random_reflection", False)
        if random_reflection:
            records_per_batch = config["batch_size"]
        else:
            records_per_batch = max(1, config["batch_size"] // 2)

        if "training_reuse" in config:
            train_steps = math.ceil(
                config["training_reuse"]
                * len(training_fresh)
                / records_per_batch
            )
            train_steps = max(1, train_steps)
        else:
            train_steps = config["train_steps"]

        for unused_step in range(train_steps):
            sample_count = min(records_per_batch, len(replay.data))
            sampled = replay.sample(sample_count)
            inputs, policies, values = records_to_training_arrays(
                sampled,
                random_reflection,
            )
            result = network.model.train_on_batch(
                inputs,
                {"policy": policies, "value": values},
                return_dict=True,
            )
            for name in result:
                if name not in loss_totals:
                    loss_totals[name] = 0.0
                loss_totals[name] += float(result[name])

        validation_result = {}
        if validation_records:
            validation_count = min(
                config.get("validation_sample_size", 256),
                len(validation_records),
            )
            validation_sample = random.sample(
                validation_records,
                validation_count,
            )
            inputs, policies, values = records_to_training_arrays(
                validation_sample
            )
            validation_result = network.model.test_on_batch(
                inputs,
                {"policy": policies, "value": values},
                return_dict=True,
            )

        checkpoint = os.path.join(
            checkpoint_dir, "iteration-%04d.keras" % iteration
        )
        if config["board_size"] == 5:
            tactics = evaluate_tactical_suite(network)
        else:
            tactics = None
        network.save(checkpoint)
        latest_checkpoint = checkpoint
        completed_iterations += 1

        strength_summary = None
        elapsed_s = time.perf_counter() - run_started
        if strength_interval_s > 0 and elapsed_s >= next_strength_match_s:
            match_name = "match-%04d" % matches_completed
            match = run_strength_match(
                config,
                network,
                best_network,
                "iteration-%04d" % iteration,
                "best-so-far",
            )
            if match["failures"]:
                raise RuntimeError("a strength-check arena game failed")
            required_score = config["strength_score_required"]
            stronger = match["agent_a_score_rate"] >= required_score
            old_best_checkpoint = best_checkpoint
            if stronger:
                best_checkpoint = checkpoint
                best_network = load_network(checkpoint)
                strength_stalls = 0
            else:
                strength_stalls += 1

            match["candidate_checkpoint"] = checkpoint
            match["reference_checkpoint"] = old_best_checkpoint
            match["required_score_rate"] = required_score
            match["stronger"] = stronger
            match["consecutive_stalls"] = strength_stalls
            match_dir = os.path.join(run_dir, "matches")
            os.makedirs(match_dir, exist_ok=True)
            match_path = os.path.join(match_dir, match_name + ".json")
            with open(match_path, "w", encoding="utf-8") as stream:
                json.dump(match, stream, indent=2)
            strength_summary = {
                "score": match["agent_a_score"],
                "games": match["games_completed"],
                "score_rate": match["agent_a_score_rate"],
                "stronger": stronger,
                "consecutive_stalls": strength_stalls,
                "reference_checkpoint": old_best_checkpoint,
                "report": match_path,
            }
            matches_completed += 1
            elapsed_s = time.perf_counter() - run_started
            while next_strength_match_s <= elapsed_s:
                next_strength_match_s += strength_interval_s

        metric = {
            "iteration": iteration,
            "new_positions": len(fresh),
            "training_positions": len(training_fresh),
            "validation_positions": len(validation_fresh),
            "self_play_games": config["games_per_iteration"],
            "player1_wins": player_1_wins,
            "mean_game_length": game_length_total / games_finished,
            "full_searches": full_searches,
            "fast_searches": fast_searches,
            "full_simulations": full_simulations,
            "fast_simulations": fast_simulations,
            "replay_size": len(replay.data),
            "train_steps": train_steps,
            "loss": loss_totals.get("loss", 0.0) / train_steps,
            "policy_loss": loss_totals.get("policy_loss", 0.0)
            / train_steps,
            "value_loss": loss_totals.get("value_loss", 0.0)
            / train_steps,
            "validation_loss": float(validation_result.get("loss", 0.0)),
            "validation_policy_loss": float(
                validation_result.get("policy_loss", 0.0)
            ),
            "validation_value_loss": float(
                validation_result.get("value_loss", 0.0)
            ),
            "checkpoint": checkpoint,
            "iteration_s": time.perf_counter() - iteration_started,
            "elapsed_s": time.perf_counter() - run_started,
        }
        if strength_summary is not None:
            metric["strength_match"] = strength_summary
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

        if strength_stalls >= config.get("max_strength_stalls", math.inf):
            stop_reason = "three strength checks without improvement"
            break

    if strength_interval_s <= 0:
        best_checkpoint = latest_checkpoint

    return {
        "config": config,
        "iterations_completed": completed_iterations,
        "elapsed_s": time.perf_counter() - run_started,
        "stop_reason": stop_reason,
        "replay_seed_records": replay_seed_records,
        "strength_matches": matches_completed,
        "best_checkpoint": best_checkpoint,
        "best_checkpoint_sha256": file_sha256(best_checkpoint),
        "latest_checkpoint": latest_checkpoint,
        "latest_checkpoint_sha256": file_sha256(latest_checkpoint),
        "metrics": metric_path,
    }
