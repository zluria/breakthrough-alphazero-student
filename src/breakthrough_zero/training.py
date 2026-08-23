"""Simple synchronous PLAY -> REPLAY -> TRAIN loop."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from .data import append_records, play_self_play_game, read_records, write_records
from .diagnostics import evaluate_tactical_suite
from .evaluation import evaluate_pair
from .game import Breakthrough
from .neural import GameNetwork, NeuralBoundary
from .puct import NeuralEvaluator, PUCTPlayer, RolloutEvaluator
from .replay import ReplayBuffer, records_to_training_arrays


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class PretrainingConfig:
    board_size: int = 5
    starting_rows: int = 1
    games: int = 10_000
    simulations: int = 100
    cpuct: float = 1.5
    seed: int = 20260811
    tactical_rollouts: bool = False
    temperature: float = 1.0
    temperature_plies: int = 8


def generate_pretraining_data(config: PretrainingConfig, output_path: str | Path) -> dict:
    """Generate the assignment's exact dummy-network MCTS data."""

    output_path = Path(output_path)
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite raw data: {output_path}")
    evaluator = RolloutEvaluator(config.seed, tactical=config.tactical_rollouts)
    search = PUCTPlayer(
        evaluator,
        simulations=config.simulations,
        cpuct=config.cpuct,
        seed=config.seed,
    )
    started = time.perf_counter()
    positions = 0
    p1_wins = 0
    for game_index in range(config.games):
        records = play_self_play_game(
            search,
            board_size=config.board_size,
            starting_rows=config.starting_rows,
            game_index=game_index,
            seed=config.seed + game_index,
            temperature=config.temperature,
            temperature_plies=config.temperature_plies,
            add_root_noise=False,
        )
        positions += append_records(output_path, records)
        p1_wins += int(records[0].final_outcome == 1)
    return {
        **asdict(config),
        "positions": positions,
        "player1_wins": p1_wins,
        "elapsed_s": time.perf_counter() - started,
        "output": str(output_path),
        "output_sha256": _sha256(output_path),
    }


def train_pretrained_network(
    data_path: str | Path,
    output_path: str | Path,
    *,
    epochs: int = 8,
    batch_size: int = 128,
    seed: int = 20260811,
    filters: int = 48,
    residual_blocks: int = 3,
) -> dict:
    """Train MSE value and cross-entropy policy heads from dummy-MCTS data."""

    output_path = Path(output_path)
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite a checkpoint: {output_path}")
    records = list(read_records(data_path))
    if not records:
        raise ValueError("pretraining data is empty")
    board_size = records[0].board_size
    if any(record.board_size != board_size for record in records):
        raise ValueError("one pretraining file must contain a single board size")
    rng = np.random.default_rng(seed)
    game_ids = np.asarray(sorted({record.game_index for record in records}))
    rng.shuffle(game_ids)
    validation_game_count = max(1, len(game_ids) // 10)
    validation_games = set(int(game) for game in game_ids[:validation_game_count])
    training_records = [r for r in records if r.game_index not in validation_games]
    validation_records = [r for r in records if r.game_index in validation_games]
    x, p, z, training_conversion = records_to_training_arrays(
        training_records, augment=True
    )
    validation_x, validation_p, validation_z, validation_conversion = (
        records_to_training_arrays(validation_records, augment=True)
    )
    order = rng.permutation(len(x))
    x, p, z = x[order], p[order], z[order]
    GameNetwork._tf().keras.utils.set_random_seed(seed)
    network = GameNetwork(
        board_size, filters=filters, residual_blocks=residual_blocks
    )
    history = network.fit(
        x,
        p,
        z,
        validation_data=(
            validation_x,
            {"policy": validation_p, "value": validation_z},
        ),
        epochs=epochs,
        batch_size=batch_size,
        verbose=2,
    )
    network.save(output_path)
    return {
        "data": str(data_path),
        "checkpoint": str(output_path),
        "checkpoint_sha256": _sha256(output_path),
        "records": len(records),
        "network": {"filters": filters, "residual_blocks": residual_blocks},
        "training_conversion": training_conversion,
        "validation_conversion": validation_conversion,
        "training_games": len(game_ids) - validation_game_count,
        "validation_games": validation_game_count,
        "history": {key: [float(v) for v in values] for key, values in history.history.items()},
        "tactical": evaluate_tactical_suite(network) if board_size == 5 else None,
    }


@dataclass(frozen=True)
class LoopConfig:
    board_size: int = 5
    starting_rows: int = 1
    iterations: int = 10
    games_per_iteration: int = 32
    simulations: int = 100
    cpuct: float = 1.5
    train_steps: int = 32
    batch_size: int = 64
    replay_capacity: int = 20_000
    seed: int = 20260811
    dirichlet_alpha: float | None = 0.3
    dirichlet_fraction: float = 0.25
    temperature: float = 1.0
    temperature_plies: int = 8


def _append_metric(path: Path, metric: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(metric, separators=(",", ":")) + "\n")


def _mean_loss(losses: list[dict], key: str) -> float | None:
    values = [float(loss[key]) for loss in losses if key in loss]
    return sum(values) / len(values) if values else None


def _policy_kl(network: GameNetwork, x: np.ndarray, target: np.ndarray) -> float:
    prediction = network.model(x, training=False)["policy"].numpy()
    prediction -= prediction.max(axis=1, keepdims=True)
    probabilities = np.exp(prediction)
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    clipped_target = np.clip(target, 1e-12, 1.0)
    clipped_prediction = np.clip(probabilities, 1e-12, 1.0)
    return float(
        np.mean(np.sum(clipped_target * np.log(clipped_target / clipped_prediction), axis=1))
    )


def run_learning_loop(
    config: LoopConfig,
    run_dir: str | Path,
    *,
    initial_checkpoint: str | Path | None = None,
) -> dict:
    """Run synchronous self-play and training; the latest network is always actor."""

    run_dir = Path(run_dir)
    raw_dir = run_dir / "raw"
    checkpoint_dir = run_dir / "checkpoints"
    metric_path = run_dir / "metrics.jsonl"
    raw_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    if metric_path.exists():
        raise FileExistsError(
            f"run directory already contains metrics; choose a fresh run_dir: {metric_path}"
        )
    GameNetwork._tf().keras.utils.set_random_seed(config.seed)
    if initial_checkpoint:
        network = GameNetwork.load(initial_checkpoint)
    else:
        network = GameNetwork(config.board_size)
    replay = ReplayBuffer(config.replay_capacity, config.seed)
    total_fresh_positions = 0
    total_examples_presented = 0
    previous_tactical_accuracy: float | None = None
    run_started = time.perf_counter()

    for iteration in range(config.iterations):
        iteration_started = time.perf_counter()
        actor_checkpoint = checkpoint_dir / f"actor-{iteration:04d}.keras"
        network.save(actor_checkpoint)
        boundary = NeuralBoundary(network)
        search = PUCTPlayer(
            NeuralEvaluator(boundary),
            simulations=config.simulations,
            cpuct=config.cpuct,
            seed=config.seed + iteration,
            dirichlet_alpha=config.dirichlet_alpha,
            dirichlet_fraction=config.dirichlet_fraction,
        )
        fresh = []
        for offset in range(config.games_per_iteration):
            game_index = iteration * config.games_per_iteration + offset
            fresh.extend(
                play_self_play_game(
                    search,
                    board_size=config.board_size,
                    starting_rows=config.starting_rows,
                    game_index=game_index,
                    seed=config.seed + game_index,
                    temperature=config.temperature,
                    temperature_plies=config.temperature_plies,
                    add_root_noise=True,
                )
            )
        raw_path = raw_dir / f"iteration-{iteration:04d}.jsonl.gz"
        if raw_path.exists():
            raise FileExistsError(f"refusing to append duplicate iteration data: {raw_path}")
        write_records(raw_path, fresh)
        validation_game_count = max(1, config.games_per_iteration // 8)
        validation_start = (
            iteration * config.games_per_iteration
            + config.games_per_iteration
            - validation_game_count
        )
        validation_records = [
            record for record in fresh if record.game_index >= validation_start
        ]
        replay_records = [
            record for record in fresh if record.game_index < validation_start
        ]
        replay.add(replay_records, iteration)
        total_fresh_positions += len(fresh)

        losses: list[dict] = []
        duplicate_count = 0
        iteration_examples_presented = 0
        last_x = last_p = None
        records_per_batch = max(1, config.batch_size // 2)
        for _ in range(config.train_steps):
            sampled = replay.sample(records_per_batch)
            x, p, z, conversion = records_to_training_arrays(sampled, augment=True)
            duplicate_count += conversion["symmetry_duplicates_removed"]
            if len(x) > config.batch_size:
                x, p, z = x[: config.batch_size], p[: config.batch_size], z[: config.batch_size]
            result = network.model.train_on_batch(
                x, {"policy": p, "value": z}, return_dict=True
            )
            losses.append({key: float(value) for key, value in result.items()})
            total_examples_presented += len(x)
            iteration_examples_presented += len(x)
            last_x, last_p = x, p

        validation_x, validation_p, validation_z, _ = records_to_training_arrays(
            validation_records, augment=True
        )
        validation_loss = network.model.test_on_batch(
            validation_x,
            {"policy": validation_p, "value": validation_z},
            return_dict=True,
        )

        checkpoint = checkpoint_dir / f"iteration-{iteration:04d}.keras"
        network.save(checkpoint)
        network.save(checkpoint_dir / "latest.keras")
        tactics = evaluate_tactical_suite(network) if config.board_size == 5 else None
        previous_network = GameNetwork.load(actor_checkpoint)

        def latest_factory():
            return PUCTPlayer(
                NeuralEvaluator(NeuralBoundary(network)),
                simulations=min(24, config.simulations),
                cpuct=config.cpuct,
                seed=config.seed + iteration,
            )

        def previous_factory():
            return PUCTPlayer(
                NeuralEvaluator(NeuralBoundary(previous_network)),
                simulations=min(24, config.simulations),
                cpuct=config.cpuct,
                seed=config.seed + iteration + 100_000,
            )

        regression_arena = evaluate_pair(
            latest_factory,
            previous_factory,
            agent_a_name=f"iteration-{iteration:04d}",
            agent_b_name=f"actor-{iteration:04d}",
            opening_count=6,
            prefix_plies=2 if config.board_size == 5 else 4,
            board_size=config.board_size,
            starting_rows=config.starting_rows,
            seed=config.seed + iteration,
        )
        regression_summary = {
            key: value for key, value in regression_arena.items() if key != "games"
        }
        alarms: list[str] = []
        replay_metrics = replay.metrics(iteration)
        ratio = total_examples_presented / max(1, replay.total_added)
        if not 0.25 <= ratio <= 8.0:
            alarms.append(f"replay consumption ratio {ratio:.3f} is outside [0.25, 8.0]")
        if regression_arena["score_95_interval"][1] < 0.5:
            alarms.append("new checkpoint is demonstrably weaker than its actor checkpoint")
        if regression_arena["failures"]:
            alarms.append(
                f"regression arena had {regression_arena['failures']} failed games"
            )
        alarms.extend(regression_arena["alarms"])
        if tactics is not None:
            current_accuracy = tactics["value_accuracy"]
            if previous_tactical_accuracy is not None and current_accuracy + 0.1 < previous_tactical_accuracy:
                alarms.append("tactical value accuracy declined by more than 0.10")
            if tactics["mean_color_swap_absolute_error"] > 1e-5:
                alarms.append("color-swap consistency invariant failed")
            previous_tactical_accuracy = current_accuracy

        elapsed = time.perf_counter() - iteration_started
        metric = {
            "iteration": iteration,
            "fresh_games": config.games_per_iteration,
            "fresh_positions": len(fresh),
            "validation_positions": len(validation_records),
            "replay_positions_added": len(replay_records),
            "total_fresh_positions": total_fresh_positions,
            "examples_presented": total_examples_presented,
            "examples_presented_this_iteration": iteration_examples_presented,
            "replay_consumption_ratio": ratio,
            "replay": replay_metrics,
            "loss": _mean_loss(losses, "loss"),
            "policy_loss": _mean_loss(losses, "policy_loss"),
            "value_loss": _mean_loss(losses, "value_loss"),
            "validation_loss": float(validation_loss.get("loss", 0.0)),
            "validation_policy_loss": float(
                validation_loss.get("policy_loss", 0.0)
            ),
            "validation_value_loss": float(
                validation_loss.get("value_loss", 0.0)
            ),
            "policy_kl": _policy_kl(network, last_x, last_p),
            "symmetry_duplicates_removed": duplicate_count,
            "positions_per_second": len(fresh) / max(elapsed, 1e-9),
            "tactical": tactics,
            "regression_arena": regression_summary,
            "alarms": alarms,
            "checkpoint": str(checkpoint),
        }
        _append_metric(metric_path, metric)
        if alarms:
            raise RuntimeError("; ".join(alarms))

    latest_checkpoint = checkpoint_dir / "latest.keras"
    return {
        "config": asdict(config),
        "iterations_completed": config.iterations,
        "elapsed_s": time.perf_counter() - run_started,
        "latest_checkpoint": str(latest_checkpoint),
        "latest_checkpoint_sha256": _sha256(latest_checkpoint),
        "metrics": str(metric_path),
    }
