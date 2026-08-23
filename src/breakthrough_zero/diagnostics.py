"""Solver-labelled supervision and elementary tactical checks."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import random
import math

import numpy as np

from .agents import AlphaBetaAgent
from .game import Breakthrough, PLAYER_1
from .neural import GameNetwork, NeuralBoundary, canonical_planes
from .symmetry import Symmetry


@dataclass(frozen=True)
class DiagnosticPosition:
    name: str
    category: str
    game: Breakthrough


def tactical_suite() -> list[DiagnosticPosition]:
    """Native 5x5 positions, paired with their color-swapped equivalents."""

    bases = [
        DiagnosticPosition(
            "immediate_win",
            "immediate wins",
            Breakthrough.from_rows(
                [".....", "....2", ".....", "..1..", "....."],
                player_to_move=PLAYER_1,
            ),
        ),
        DiagnosticPosition(
            "must_defend",
            "immediate threats requiring defense",
            Breakthrough.from_rows(
                [".1...", "..2..", ".....", ".....", "....2"],
                player_to_move=PLAYER_1,
            ),
        ),
        DiagnosticPosition(
            "forced_race",
            "forced wins and losses",
            Breakthrough.from_rows(
                ["1....", ".....", ".1.2.", ".....", "....2"],
                player_to_move=PLAYER_1,
            ),
        ),
        DiagnosticPosition(
            "forced_loss",
            "forced wins and losses",
            Breakthrough.from_rows(
                ["....1", ".2..1", ".....", ".....", "....2"],
                player_to_move=PLAYER_1,
            ),
        ),
        DiagnosticPosition(
            "material_edge",
            "material advantages",
            Breakthrough.from_rows(
                ["11...", "..1..", ".....", "...2.", "....2"],
                player_to_move=PLAYER_1,
            ),
        ),
        DiagnosticPosition(
            "passed_pawn",
            "advanced passed pawns",
            Breakthrough.from_rows(
                ["1....", ".....", ".1...", "...2.", "....2"],
                player_to_move=PLAYER_1,
            ),
        ),
    ]
    swap = Symmetry(swap_players=True)
    paired = list(bases)
    paired.extend(
        DiagnosticPosition(f"{item.name}_swapped", item.category, swap.state(item.game))
        for item in bases
    )
    return paired


def generate_solver_examples(
    count: int,
    *,
    seed: int = 0,
    search_depth: int = 6,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, int]]:
    """Create a balanced native-5x5 diagnostic set labelled by alpha-beta.

    These labels are intentionally separate from the required MCTS pretraining
    data. Positions are sampled from seeded random games after the opening.
    """

    if count < 2:
        raise ValueError("count must be at least two")
    target_per_label = count // 2
    rng = random.Random(seed)
    solver = AlphaBetaAgent(depth=search_depth)
    # Balance the labels that the mover-relative value head actually sees.
    buckets: dict[int, list[tuple[np.ndarray, np.ndarray, float]]] = {1: [], -1: []}
    attempts = 0
    while min(len(buckets[1]), len(buckets[-1])) < target_per_label:
        attempts += 1
        if attempts > count * 200:
            raise RuntimeError("could not build a balanced solver dataset")
        game = Breakthrough(5, 1)
        skip = rng.randint(4, 14)
        for _ in range(skip):
            if game.status() is not None:
                break
            game.make_move(rng.choice(game.legal_moves()))
        if game.status() is not None or len(game.legal_moves()) < 2:
            continue
        value, move = solver.search(game, search_depth)
        if abs(value) < 0.25:
            continue
        label = 1 if value > 0 else -1
        relative_label = label * game.player_to_move
        if len(buckets[relative_label]) >= target_per_label:
            continue
        policy = np.zeros(game.action_size, dtype=np.float32)
        policy[game.encode_move(move)] = 1.0
        relative_value = NeuralBoundary.relative_target(label, game.player_to_move)
        buckets[relative_label].append((canonical_planes(game), policy, relative_value))

    examples = buckets[1][:target_per_label] + buckets[-1][:target_per_label]
    rng.shuffle(examples)
    x, p, z = zip(*examples)
    return (
        np.stack(x),
        np.stack(p),
        np.asarray(z, dtype=np.float32)[:, None],
        {"examples": len(examples), "positive": target_per_label, "negative": target_per_label},
    )


def _solver_optimal_actions(
    game: Breakthrough, depth: int
) -> tuple[float, list[int]]:
    values: dict[int, float] = {}
    for move in game.legal_moves():
        action = game.encode_move(move)
        game.make_move(move)
        if game.status() is not None:
            value = float(game.status())
        elif depth <= 1:
            value = AlphaBetaAgent.evaluate(game)
        else:
            solver = AlphaBetaAgent(depth=depth - 1)
            value, _ = solver.search(game, depth - 1)
        game.unmake_move(move)
        values[action] = value
    best = max(values.values()) if game.player_to_move == 1 else min(values.values())
    optimal = [action for action, value in values.items() if math.isclose(value, best)]
    return best, optimal


def evaluate_tactical_suite(
    network: GameNetwork,
    *,
    solver_depth: int = 8,
) -> dict:
    boundary = NeuralBoundary(network)
    rows: list[dict] = []
    for item in tactical_suite():
        expected_value, optimal_actions = _solver_optimal_actions(
            item.game.clone(), solver_depth
        )
        expected_outcome = 1 if expected_value > 0 else -1
        prediction = boundary.predict(item.game)
        predicted_action = max(prediction.priors, key=prediction.priors.get)
        rows.append(
            {
                "name": item.name,
                "category": item.category,
                "player_to_move": item.game.player_to_move,
                "expected_absolute_outcome": expected_outcome,
                "predicted_absolute_value": prediction.value,
                "value_correct": prediction.value * expected_outcome > 0,
                "solver_actions": optimal_actions,
                "top_policy_action": predicted_action,
                "policy_correct": predicted_action in optimal_actions,
            }
        )
    paired_consistency = []
    by_name = {row["name"]: row for row in rows}
    for row in rows:
        if row["name"].endswith("_swapped"):
            base = by_name[row["name"].removesuffix("_swapped")]
            paired_consistency.append(
                abs(base["predicted_absolute_value"] + row["predicted_absolute_value"])
            )
    return {
        "positions": rows,
        "value_accuracy": sum(row["value_correct"] for row in rows) / len(rows),
        "policy_accuracy": sum(row["policy_correct"] for row in rows) / len(rows),
        "mean_color_swap_absolute_error": float(np.mean(paired_consistency)),
    }


def run_supervised_diagnostic(
    output_dir: str | Path,
    *,
    examples: int = 2048,
    seed: int = 0,
    epochs: int = 24,
    batch_size: int = 64,
) -> dict:
    """Train the diagnostic CNN and refuse to pass weak elementary learning."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    x, p, z, dataset_metrics = generate_solver_examples(examples, seed=seed)
    split = int(0.8 * len(x))
    GameNetwork._tf().keras.utils.set_random_seed(seed)
    network = GameNetwork(5)
    early_stopping = network._tf().keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=5, restore_best_weights=True
    )
    history = network.fit(
        x[:split],
        p[:split],
        z[:split],
        validation_data=(x[split:], {"policy": p[split:], "value": z[split:]}),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stopping],
        verbose=2,
    )
    model_path = output_dir / "diagnostic-5x5.keras"
    network.save(model_path)
    heldout_prediction = network.model(x[split:], training=False)
    heldout_policy = np.asarray(heldout_prediction["policy"])
    heldout_value = np.asarray(heldout_prediction["value"])
    heldout = {
        "examples": len(x) - split,
        "value_accuracy": float(np.mean(np.sign(heldout_value) == z[split:])),
        "policy_accuracy": float(
            np.mean(np.argmax(heldout_policy, axis=1) == np.argmax(p[split:], axis=1))
        ),
    }
    tactics = evaluate_tactical_suite(network)
    report = {
        "dataset": dataset_metrics,
        "history": {key: [float(v) for v in values] for key, values in history.history.items()},
        "heldout": heldout,
        "tactical": tactics,
        "model": str(model_path),
        "passed": heldout["value_accuracy"] >= 0.7
        and heldout["policy_accuracy"] >= 0.35
        and tactics["value_accuracy"] >= 0.75
        and tactics["policy_accuracy"] >= 0.5
        and tactics["mean_color_swap_absolute_error"] <= 1e-5,
    }
    report_path = output_dir / "diagnostic-report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    if not report["passed"]:
        raise RuntimeError(
            "supervised diagnostic failed; inspect diagnostic-report.json before MCTS training"
        )
    return report
