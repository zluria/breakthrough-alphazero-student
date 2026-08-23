"""Paired-opening arenas, score intervals, and Elo differences."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import random
import time
from typing import Callable, Protocol

from .game import Breakthrough, Move, PLAYER_1


class Agent(Protocol):
    def choose_move(self, game: Breakthrough) -> Move: ...


AgentFactory = Callable[[], Agent]


@dataclass(frozen=True)
class PlayedGame:
    opening_index: int
    agent_a_player: int
    winner: int | None
    agent_a_score: float
    moves: list[tuple[int, int]]
    elapsed_s: float
    failure: str | None = None


def randomized_openings(
    count: int,
    *,
    board_size: int,
    starting_rows: int,
    prefix_plies: int,
    seed: int,
) -> list[list[Move]]:
    rng = random.Random(seed)
    openings: list[list[Move]] = []
    seen: set[tuple[tuple[int, int], ...]] = set()
    attempts = 0
    while len(openings) < count:
        attempts += 1
        if attempts > count * 100:
            raise RuntimeError("could not produce enough distinct opening prefixes")
        game = Breakthrough(board_size, starting_rows)
        moves: list[Move] = []
        for _ in range(prefix_plies):
            if game.status() is not None:
                break
            move = rng.choice(game.legal_moves())
            moves.append(move)
            game.make_move(move)
        if game.status() is not None:
            continue
        key = tuple((move.from_sq, move.to_sq) for move in moves)
        if key not in seen:
            seen.add(key)
            openings.append(moves)
    return openings


def play_arena_game(
    agent_a: Agent,
    agent_b: Agent,
    *,
    agent_a_player: int,
    opening: list[Move],
    opening_index: int,
    board_size: int,
    starting_rows: int,
) -> PlayedGame:
    game = Breakthrough(board_size, starting_rows)
    move_log: list[tuple[int, int]] = []
    started = time.perf_counter()
    try:
        for move in opening:
            game.make_move(move)
            move_log.append((move.from_sq, move.to_sq))
        while game.status() is None:
            current = agent_a if game.player_to_move == agent_a_player else agent_b
            move = current.choose_move(game)
            game.make_move(move)
            move_log.append((move.from_sq, move.to_sq))
        winner = game.status()
        score = 1.0 if winner == agent_a_player else 0.0
        failure = None
    except Exception as error:  # arena failures are recorded, not hidden
        winner = None
        score = 0.0
        failure = f"{type(error).__name__}: {error}"
    return PlayedGame(
        opening_index,
        agent_a_player,
        winner,
        score,
        move_log,
        time.perf_counter() - started,
        failure,
    )


def wilson_interval(successes: float, games: int, z: float = 1.96) -> tuple[float, float]:
    if games < 1:
        return 0.0, 1.0
    p = successes / games
    denominator = 1 + z * z / games
    center = (p + z * z / (2 * games)) / denominator
    margin = z * math.sqrt(p * (1 - p) / games + z * z / (4 * games * games)) / denominator
    return max(0.0, center - margin), min(1.0, center + margin)


def score_to_elo(score: float) -> float:
    score = min(1 - 1e-6, max(1e-6, score))
    return 400 * math.log10(score / (1 - score))


def evaluate_pair(
    agent_a_factory: AgentFactory,
    agent_b_factory: AgentFactory,
    *,
    agent_a_name: str,
    agent_b_name: str,
    opening_count: int = 50,
    prefix_plies: int = 4,
    board_size: int = 5,
    starting_rows: int = 1,
    seed: int = 20260811,
) -> dict:
    """Play every deterministic opening twice with agent colors reversed."""

    openings = randomized_openings(
        opening_count,
        board_size=board_size,
        starting_rows=starting_rows,
        prefix_plies=prefix_plies,
        seed=seed,
    )
    games: list[PlayedGame] = []
    for opening_index, opening in enumerate(openings):
        games.append(
            play_arena_game(
                agent_a_factory(),
                agent_b_factory(),
                agent_a_player=1,
                opening=opening,
                opening_index=opening_index,
                board_size=board_size,
                starting_rows=starting_rows,
            )
        )
        games.append(
            play_arena_game(
                agent_a_factory(),
                agent_b_factory(),
                agent_a_player=-1,
                opening=opening,
                opening_index=opening_index,
                board_size=board_size,
                starting_rows=starting_rows,
            )
        )

    successful = [game for game in games if game.failure is None]
    score = sum(game.agent_a_score for game in successful)
    rate = score / len(successful) if successful else 0.0
    interval = wilson_interval(score, len(successful))
    sequences = [tuple(game.moves) for game in successful]
    duplicate_fraction = 1 - len(set(sequences)) / max(1, len(sequences))
    alarms: list[str] = []
    if duplicate_fraction > 0.25:
        alarms.append(f"duplicate game fraction is {duplicate_fraction:.3f}")
    if len(successful) >= 40 and rate == 0.5:
        alarms.append("suspiciously exact 50/50 split; inspect paired games")
    return {
        "agent_a": agent_a_name,
        "agent_b": agent_b_name,
        "board_size": board_size,
        "opening_count": opening_count,
        "games_requested": len(games),
        "games_completed": len(successful),
        "failures": len(games) - len(successful),
        "agent_a_score": score,
        "agent_a_score_rate": rate,
        "score_95_interval": interval,
        "elo_difference": score_to_elo(rate) if successful else None,
        "elo_95_interval": (
            score_to_elo(interval[0]),
            score_to_elo(interval[1]),
        )
        if successful
        else None,
        "mean_game_seconds": sum(game.elapsed_s for game in successful)
        / max(1, len(successful)),
        "duplicate_game_fraction": duplicate_fraction,
        "alarms": alarms,
        "games": [asdict(game) for game in games],
    }
