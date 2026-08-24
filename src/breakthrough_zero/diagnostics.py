"""Solver-labelled supervision and tactical measurements."""

import json
import math
import os
import random

import numpy as np

from .agents import AlphaBetaAgent, evaluate_position
from .game import Breakthrough, PLAYER_1, game_from_rows
from .neural import GameNetwork, NeuralBoundary, canonical_planes
from .symmetry import transform_state


def tactical_suite():
    """Return 20 exact positions and their 20 color-swapped partners.

    The categories cover immediate wins, forced defense, longer forced results,
    material traps, and pawn races. Color-swapped pairs test the absolute-value
    convention independently of tactical strength.
    """

    definitions = [
        (
            "immediate_win_center",
            "immediate wins",
            [".....", "....2", ".....", "..1..", "....."],
            1,
        ),
        (
            "immediate_win_left",
            "immediate wins",
            ["1....", "..2..", ".....", "1....", "....2"],
            1,
        ),
        (
            "immediate_win_capture",
            "immediate wins",
            ["....1", ".....", "..2..", "....1", "...2."],
            1,
        ),
        (
            "must_defend_center",
            "immediate threats requiring defense",
            [".1...", "..2..", ".....", ".....", "....2"],
            1,
        ),
        (
            "must_defend_double_left",
            "immediate threats requiring defense",
            ["1...1", ".2.2.", ".....", ".....", "....."],
            -1,
        ),
        (
            "must_defend_double_right",
            "immediate threats requiring defense",
            [".1.1.", "2...2", "..1..", ".....", "....."],
            -1,
        ),
        (
            "must_defend_right",
            "immediate threats requiring defense",
            ["...1.", "....2", ".1...", ".....", "2...."],
            1,
        ),
        (
            "forced_race",
            "forced wins and losses",
            ["1....", ".....", ".1.2.", ".....", "....2"],
            1,
        ),
        (
            "forced_loss",
            "forced wins and losses",
            ["....1", ".2..1", ".....", ".....", "....2"],
            -1,
        ),
        (
            "forced_loss_center",
            "forced wins and losses",
            ["1...1", ".....", ".2...", "...2.", "....."],
            -1,
        ),
        (
            "forced_loss_wide",
            "forced wins and losses",
            ["..1..", ".....", "2....", "....2", "....."],
            -1,
        ),
        (
            "forced_loss_advanced",
            "forced wins and losses",
            ["....1", ".....", "2....", ".2...", "....."],
            -1,
        ),
        (
            "material_edge",
            "material advantages",
            ["11...", "..1..", ".....", "...2.", "....2"],
            1,
        ),
        (
            "material_center",
            "material advantages",
            ["11...", "...1.", ".....", "..2..", "....2"],
            1,
        ),
        (
            "material_trap_split",
            "material advantages",
            ["1.1.1", ".2.2.", ".....", ".....", "....."],
            -1,
        ),
        (
            "material_trap_wide",
            "material advantages",
            ["11.11", "2...2", ".....", ".....", "....."],
            -1,
        ),
        (
            "passed_pawn",
            "advanced passed pawns",
            ["1....", ".....", ".1...", "...2.", "....2"],
            1,
        ),
        (
            "passed_pawn_center",
            "advanced passed pawns",
            ["....1", ".....", "..1..", "2....", "....2"],
            1,
        ),
        (
            "passed_pawn_loses_race",
            "advanced passed pawns",
            ["1....", "....2", "..1..", "2....", "....."],
            -1,
        ),
        (
            "passed_pawn_too_slow",
            "advanced passed pawns",
            ["....1", ".2...", "...1.", ".....", "2...."],
            -1,
        ),
    ]

    positions = []
    for name, category, rows, outcome in definitions:
        positions.append(
            {
                "name": name,
                "category": category,
                "game": game_from_rows(rows, PLAYER_1),
                "outcome": outcome,
            }
        )

    paired = list(positions)
    for position in positions:
        paired.append(
            {
                "name": position["name"] + "_swapped",
                "category": position["category"],
                "game": transform_state(position["game"], (True, False)),
                "outcome": -position["outcome"],
            }
        )
    return paired


def generate_solver_examples(count, search_depth=6):
    """Create a balanced set of positions labelled by alpha-beta search."""

    if count < 2:
        raise ValueError("count must be at least two")
    target_per_label = count // 2
    solver = AlphaBetaAgent(search_depth)
    boundary = NeuralBoundary()
    # Balance the labels seen by the mover-relative value head. An absolute set
    # balanced by winner could still be badly imbalanced after perspective
    # conversion if one player is usually to move.
    buckets = {1: [], -1: []}
    attempts = 0

    while min(len(buckets[1]), len(buckets[-1])) < target_per_label:
        attempts += 1
        if attempts > count * 200:
            raise RuntimeError("could not build a balanced solver dataset")
        game = Breakthrough(5, 1)
        skip = random.randint(4, 14)
        for unused_move in range(skip):
            if game.status() is not None:
                break
            move = random.choice(game.legal_moves())
            game.make_move(move)
        if game.status() is not None or len(game.legal_moves()) < 2:
            continue

        value, move = solver.search(game, search_depth)
        if abs(value) < 0.25:
            continue
        if value > 0:
            absolute_label = 1
        else:
            absolute_label = -1
        relative_label = absolute_label * game.player_to_move
        if len(buckets[relative_label]) >= target_per_label:
            continue

        policy = np.zeros(game.action_size, dtype=np.float32)
        policy[game.encode_move(move)] = 1.0
        value_target = boundary.relative_target(
            absolute_label, game.player_to_move
        )
        example = (canonical_planes(game), policy, value_target)
        buckets[relative_label].append(example)

    examples = buckets[1][:target_per_label] + buckets[-1][:target_per_label]
    random.shuffle(examples)
    inputs = []
    policies = []
    values = []
    for planes, policy, value in examples:
        inputs.append(planes)
        policies.append(policy)
        values.append(value)
    metrics = {
        "examples": len(examples),
        "positive": target_per_label,
        "negative": target_per_label,
    }
    value_array = np.array(values, dtype=np.float32).reshape(-1, 1)
    return np.stack(inputs), np.stack(policies), value_array, metrics


def solver_optimal_actions(game, depth):
    """Return the best absolute value and every action attaining it."""

    values = {}
    for move in game.legal_moves():
        action = game.encode_move(move)
        game.make_move(move)
        if game.status() is not None:
            value = float(game.status())
        elif depth <= 1:
            value = evaluate_position(game)
        else:
            solver = AlphaBetaAgent(depth - 1)
            value, unused_move = solver.search(game, depth - 1)
        game.unmake_move(move)
        values[action] = value

    # Values remain absolute: Player 1 chooses the maximum and Player 2 chooses
    # the minimum, just as in PUCT's player-aware exploitation term.
    if game.player_to_move == 1:
        best_value = max(values.values())
    else:
        best_value = min(values.values())
    best_actions = []
    for action in values:
        if math.isclose(values[action], best_value):
            best_actions.append(action)
    return best_value, best_actions


def evaluate_tactical_suite(network, solver_depth=8):
    """Measure value calibration, policy choices, and color-swap consistency."""

    boundary = NeuralBoundary(network)
    rows = []
    for position in tactical_suite():
        game = position["game"]
        unused_value, best_actions = solver_optimal_actions(
            game.clone(), solver_depth
        )
        expected_outcome = position["outcome"]
        prediction = boundary.predict(game)
        priors = prediction["priors"]
        predicted_action = max(priors, key=priors.get)
        # The product v*z is a continuous correctness score: +1 is confidently
        # correct, zero is uncertain, and -1 is confidently wrong.
        signed_value = prediction["value"] * expected_outcome
        rows.append(
            {
                "name": position["name"],
                "category": position["category"],
                "player_to_move": game.player_to_move,
                "expected_absolute_outcome": expected_outcome,
                "predicted_absolute_value": prediction["value"],
                "signed_value": signed_value,
                "value_correct": signed_value > 0,
                "solver_actions": best_actions,
                "top_policy_action": predicted_action,
                "policy_correct": predicted_action in best_actions,
            }
        )

    by_name = {}
    for row in rows:
        by_name[row["name"]] = row
    # Swapping the players must negate an absolute value prediction. The sum of
    # each pair should therefore be zero even when the prediction itself is bad.
    swap_errors = []
    for row in rows:
        if row["name"].endswith("_swapped"):
            original_name = row["name"][:-8]
            original = by_name[original_name]
            error = abs(
                original["predicted_absolute_value"]
                + row["predicted_absolute_value"]
            )
            swap_errors.append(error)

    correct_values = 0
    correct_policies = 0
    signed_value_sum = 0.0
    category_results = {}
    for row in rows:
        correct_values += int(row["value_correct"])
        correct_policies += int(row["policy_correct"])
        signed_value_sum += row["signed_value"]

        category = row["category"]
        if category not in category_results:
            category_results[category] = {
                "positions": 0,
                "correct_values": 0,
                "correct_policies": 0,
                "signed_value_sum": 0.0,
            }
        result = category_results[category]
        result["positions"] += 1
        result["correct_values"] += int(row["value_correct"])
        result["correct_policies"] += int(row["policy_correct"])
        result["signed_value_sum"] += row["signed_value"]

    by_category = {}
    for category in category_results:
        result = category_results[category]
        count = result["positions"]
        by_category[category] = {
            "positions": count,
            "value_accuracy": result["correct_values"] / count,
            "mean_signed_value": result["signed_value_sum"] / count,
            "policy_accuracy": result["correct_policies"] / count,
        }
    return {
        "positions": rows,
        "value_accuracy": correct_values / len(rows),
        "signed_value_sum": signed_value_sum,
        "mean_signed_value": signed_value_sum / len(rows),
        "policy_accuracy": correct_policies / len(rows),
        "mean_color_swap_absolute_error": float(np.mean(swap_errors)),
        "by_category": by_category,
    }


def run_supervised_diagnostic(
    output_dir,
    examples=2048,
    epochs=24,
    batch_size=64,
):
    """Verify that the policy/value network can learn solver-labelled data."""

    import keras

    os.makedirs(output_dir, exist_ok=True)
    inputs, policies, values, dataset_metrics = generate_solver_examples(
        examples, 6
    )
    split = int(0.8 * len(inputs))
    network = GameNetwork(5)
    early_stopping = keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True,
    )
    validation = (
        inputs[split:],
        {"policy": policies[split:], "value": values[split:]},
    )
    history = network.fit(
        inputs[:split],
        policies[:split],
        values[:split],
        validation_data=validation,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stopping],
        verbose=2,
    )
    model_path = os.path.join(output_dir, "diagnostic-5x5.keras")
    network.save(model_path)

    prediction = network.model(inputs[split:], training=False)
    predicted_policy = np.array(prediction["policy"])
    predicted_value = np.array(prediction["value"])
    value_accuracy = np.mean(np.sign(predicted_value) == values[split:])
    policy_accuracy = np.mean(
        np.argmax(predicted_policy, axis=1)
        == np.argmax(policies[split:], axis=1)
    )
    heldout = {
        "examples": len(inputs) - split,
        "value_accuracy": float(value_accuracy),
        "policy_accuracy": float(policy_accuracy),
    }
    tactics = evaluate_tactical_suite(network)

    history_values = {}
    for name in history.history:
        history_values[name] = []
        for value in history.history[name]:
            history_values[name].append(float(value))
    passed = (
        heldout["value_accuracy"] >= 0.7
        and heldout["policy_accuracy"] >= 0.35
        and tactics["value_accuracy"] >= 0.75
        and tactics["policy_accuracy"] >= 0.5
        and tactics["mean_color_swap_absolute_error"] <= 0.00001
    )
    report = {
        "dataset": dataset_metrics,
        "history": history_values,
        "heldout": heldout,
        "tactical": tactics,
        "model": model_path,
        "passed": passed,
    }
    report_path = os.path.join(output_dir, "diagnostic-report.json")
    with open(report_path, "w", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2)
    if not passed:
        raise RuntimeError("supervised diagnostic failed; inspect its report")
    return report
