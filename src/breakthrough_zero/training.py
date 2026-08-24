"""The synchronous PLAY, REPLAY, TRAIN loop.

At each iteration the current network is frozen as the actor, the actor produces
self-play search targets, and the network trains from the replay window. The
updated network is then compared with that frozen actor before the next cycle.
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
from .evaluation import evaluate_pair
from .neural import GameNetwork, NeuralBoundary, load_network
from .puct import NeuralEvaluator, PUCTPlayer, RolloutEvaluator
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

    evaluator = RolloutEvaluator(config["tactical_rollouts"])
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

    inputs, policies, values, training_conversion = records_to_training_arrays(
        training_records, True
    )
    validation = records_to_training_arrays(validation_records, True)
    validation_inputs = validation[0]
    validation_policies = validation[1]
    validation_values = validation[2]
    validation_conversion = validation[3]

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
        "training_conversion": training_conversion,
        "validation_conversion": validation_conversion,
        "training_games": len(game_numbers) - validation_game_count,
        "validation_games": validation_game_count,
        "history": history_values,
        "tactical": tactics,
    }


def mean_loss(losses, name):
    values = []
    for loss in losses:
        if name in loss:
            values.append(float(loss[name]))
    if not values:
        return None
    return sum(values) / len(values)


def policy_kl(network, inputs, targets):
    """Measure how far the network policy is from the sampled MCTS policy."""

    logits = network.model(inputs, training=False)["policy"].numpy()
    logits = logits - logits.max(axis=1, keepdims=True)
    probabilities = np.exp(logits)
    probabilities = probabilities / probabilities.sum(axis=1, keepdims=True)
    safe_targets = np.clip(targets, 0.000000000001, 1.0)
    safe_predictions = np.clip(probabilities, 0.000000000001, 1.0)
    rows = safe_targets * np.log(safe_targets / safe_predictions)
    return float(np.mean(np.sum(rows, axis=1)))


def tactical_decline_alarms(
    current,
    actor,
    value_tolerance=0.05,
    policy_tolerance=0.10,
):
    """Report meaningful declines from the actor's tactical measurements."""

    alarms = []
    current_value = current["mean_signed_value"]
    actor_value = actor["mean_signed_value"]
    if current_value + value_tolerance < actor_value:
        text = "tactical mean signed value declined from %.3f to %.3f"
        alarms.append(text % (actor_value, current_value))
    current_policy = current["policy_accuracy"]
    actor_policy = actor["policy_accuracy"]
    if current_policy + policy_tolerance < actor_policy:
        text = "tactical policy accuracy declined from %.3f to %.3f"
        alarms.append(text % (actor_policy, current_policy))
    if current["mean_color_swap_absolute_error"] > 0.00001:
        alarms.append("color-swap consistency invariant failed")
    return alarms


def run_learning_loop(config, run_dir, initial_checkpoint=None):
    """Alternate self-play and replay training, stopping on calibrated alarms."""

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
        # Loading a Keras checkpoint also restores Adam's momentum and variance
        # estimates. Only the step size is changed for self-play continuation.
        network = load_network(initial_checkpoint)
    network.model.optimizer.learning_rate.assign(config["learning_rate"])

    replay = ReplayBuffer(config["replay_capacity"])
    total_fresh_positions = 0
    total_examples_presented = 0
    run_started = time.perf_counter()

    for iteration in range(config["iterations"]):
        iteration_started = time.perf_counter()
        # This frozen checkpoint is the exact actor that generates the iteration.
        # It is also the reference used to judge the update below.
        actor_path = os.path.join(
            checkpoint_dir, "actor-%04d.keras" % iteration
        )
        network.save(actor_path)

        boundary = NeuralBoundary(network)
        search = PUCTPlayer(
            NeuralEvaluator(boundary),
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

        # Hold out complete fresh games before adding the remaining records to
        # replay. Validation positions are never sampled for this update.
        validation_games = max(1, config["games_per_iteration"] // 8)
        validation_start = (
            iteration * config["games_per_iteration"]
            + config["games_per_iteration"]
            - validation_games
        )
        validation_records = []
        replay_records = []
        for record in fresh:
            if record["game_index"] >= validation_start:
                validation_records.append(record)
            else:
                replay_records.append(record)
        replay.add(replay_records, iteration)
        total_fresh_positions += len(fresh)

        losses = []
        duplicate_count = 0
        iteration_examples = 0
        records_per_batch = max(1, config["batch_size"] // 2)
        last_inputs = None
        last_policies = None

        for unused_step in range(config["train_steps"]):
            # Symmetry augmentation can yield up to four examples per record.
            # Starting with half a batch limits the common case to one batch.
            sampled = replay.sample(records_per_batch)
            converted = records_to_training_arrays(sampled, True)
            inputs, policies, values, conversion = converted
            duplicate_count += conversion["symmetry_duplicates_removed"]
            if len(inputs) > config["batch_size"]:
                inputs = inputs[: config["batch_size"]]
                policies = policies[: config["batch_size"]]
                values = values[: config["batch_size"]]
            result = network.model.train_on_batch(
                inputs,
                {"policy": policies, "value": values},
                return_dict=True,
            )
            simple_result = {}
            for name in result:
                simple_result[name] = float(result[name])
            losses.append(simple_result)
            total_examples_presented += len(inputs)
            iteration_examples += len(inputs)
            last_inputs = inputs
            last_policies = policies

        converted = records_to_training_arrays(validation_records, True)
        validation_inputs = converted[0]
        validation_policies = converted[1]
        validation_values = converted[2]
        validation_loss = network.model.test_on_batch(
            validation_inputs,
            {"policy": validation_policies, "value": validation_values},
            return_dict=True,
        )

        checkpoint = os.path.join(
            checkpoint_dir, "iteration-%04d.keras" % iteration
        )
        latest_checkpoint = os.path.join(checkpoint_dir, "latest.keras")
        network.save(checkpoint)
        network.save(latest_checkpoint)

        # Diagnostics compare the trained network with the actor before any new
        # self-play is generated, isolating the effect of this training update.
        actor_network = load_network(actor_path)
        if config["board_size"] == 5:
            tactics = evaluate_tactical_suite(network)
            actor_tactics = evaluate_tactical_suite(actor_network)
        else:
            tactics = None
            actor_tactics = None

        latest_agent = PUCTPlayer(
            NeuralEvaluator(NeuralBoundary(network)),
            min(24, config["simulations"]),
            config["cpuct"],
        )
        actor_agent = PUCTPlayer(
            NeuralEvaluator(NeuralBoundary(actor_network)),
            min(24, config["simulations"]),
            config["cpuct"],
        )

        # Every opening is played with both color assignments, reducing first-
        # player and opening effects in this small regression arena.
        arena = evaluate_pair(
            latest_agent,
            actor_agent,
            "iteration-%04d" % iteration,
            "actor-%04d" % iteration,
            6,
            2 if config["board_size"] == 5 else 4,
            config["board_size"],
            config["starting_rows"],
        )
        arena_summary = dict(arena)
        del arena_summary["games"]

        alarms = []
        ratio = total_examples_presented / max(1, replay.total_added)
        if ratio < 0.25 or ratio > 8.0:
            alarms.append("replay consumption ratio %.3f is out of range" % ratio)
        if arena["score_95_interval"][1] < 0.5:
            # Alarm only when even the upper confidence bound says the update is
            # weaker; a noisy point estimate below 50% is not enough by itself.
            alarms.append("new checkpoint is weaker than its actor")
        if arena["failures"]:
            alarms.append("regression arena had failed games")
        alarms.extend(arena["alarms"])
        if tactics is not None:
            alarms.extend(tactical_decline_alarms(tactics, actor_tactics))

        elapsed = time.perf_counter() - iteration_started
        metric = {
            "iteration": iteration,
            "fresh_games": config["games_per_iteration"],
            "fresh_positions": len(fresh),
            "validation_positions": len(validation_records),
            "replay_positions_added": len(replay_records),
            "total_fresh_positions": total_fresh_positions,
            "examples_presented": total_examples_presented,
            "examples_presented_this_iteration": iteration_examples,
            "learning_rate": config["learning_rate"],
            "replay_consumption_ratio": ratio,
            "replay": replay.metrics(iteration),
            "loss": mean_loss(losses, "loss"),
            "policy_loss": mean_loss(losses, "policy_loss"),
            "value_loss": mean_loss(losses, "value_loss"),
            "validation_loss": float(validation_loss.get("loss", 0.0)),
            "validation_policy_loss": float(
                validation_loss.get("policy_loss", 0.0)
            ),
            "validation_value_loss": float(
                validation_loss.get("value_loss", 0.0)
            ),
            "policy_kl": policy_kl(network, last_inputs, last_policies),
            "symmetry_duplicates_removed": duplicate_count,
            "positions_per_second": len(fresh) / max(elapsed, 0.000000001),
            "actor_tactical": actor_tactics,
            "tactical": tactics,
            "regression_arena": arena_summary,
            "alarms": alarms,
            "checkpoint": checkpoint,
        }
        with open(metric_path, "a", encoding="utf-8", newline="\n") as stream:
            stream.write(json.dumps(metric, separators=(",", ":")) + "\n")
        # Preserve the full metric and checkpoint before stopping, so a failed
        # update remains available for diagnosis rather than disappearing.
        if alarms:
            raise RuntimeError("; ".join(alarms))

    return {
        "config": config,
        "iterations_completed": config["iterations"],
        "elapsed_s": time.perf_counter() - run_started,
        "latest_checkpoint": latest_checkpoint,
        "latest_checkpoint_sha256": file_sha256(latest_checkpoint),
        "metrics": metric_path,
    }
