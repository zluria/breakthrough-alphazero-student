"""Raw, reconstructable self-play records.

The on-disk format keeps absolute boards and absolute Player-1 outcomes. Neural
targets are derived only when training, so policy temperatures, symmetries, and
value targets can be changed later without regenerating games.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import gzip
import json
from pathlib import Path
import random
from typing import Iterable, Iterator

import numpy as np

from .game import Breakthrough
from .puct import PUCTPlayer, SearchResult


@dataclass(frozen=True)
class PositionRecord:
    game_index: int
    ply: int
    board_size: int
    starting_rows: int
    board: list[int]
    player_to_move: int
    legal_actions: list[int]
    visit_counts: list[int]
    priors: list[float]
    root_value: float
    root_visits: int
    simulations: int
    search_elapsed_s: float
    played_action: int
    final_outcome: int | None
    seed: int

    def state(self) -> Breakthrough:
        return Breakthrough(
            self.board_size,
            self.starting_rows,
            board=self.board,
            player_to_move=self.player_to_move,
        )

    @classmethod
    def from_search(
        cls,
        game: Breakthrough,
        search: SearchResult,
        *,
        game_index: int,
        ply: int,
        played_action: int,
        seed: int,
    ) -> "PositionRecord":
        legal_actions = sorted(search.visit_counts)
        return cls(
            game_index=game_index,
            ply=ply,
            board_size=game.board_size,
            starting_rows=game.starting_rows,
            board=game.board.copy(),
            player_to_move=game.player_to_move,
            legal_actions=legal_actions,
            visit_counts=[search.visit_counts[action] for action in legal_actions],
            priors=[search.priors[action] for action in legal_actions],
            root_value=search.root_value,
            root_visits=sum(search.visit_counts.values()),
            simulations=search.simulations,
            search_elapsed_s=search.elapsed_s,
            played_action=played_action,
            final_outcome=None,
            seed=seed,
        )


def choose_action(
    result: SearchResult,
    rng: np.random.Generator,
    temperature: float,
) -> int:
    actions = np.asarray(sorted(result.visit_counts), dtype=np.int64)
    counts = np.asarray([result.visit_counts[int(a)] for a in actions], dtype=np.float64)
    if temperature <= 1e-8:
        return int(actions[np.argmax(counts)])
    weights = np.power(np.maximum(counts, 1e-12), 1.0 / temperature)
    weights /= weights.sum()
    return int(rng.choice(actions, p=weights))


def play_self_play_game(
    search: PUCTPlayer,
    *,
    board_size: int,
    starting_rows: int,
    game_index: int,
    seed: int,
    temperature: float = 1.0,
    temperature_plies: int = 8,
    add_root_noise: bool = False,
) -> list[PositionRecord]:
    game = Breakthrough(board_size, starting_rows)
    rng = np.random.default_rng(seed)
    records: list[PositionRecord] = []
    ply = 0
    while game.status() is None:
        result = search.search(game, add_root_noise=add_root_noise)
        move_temperature = temperature if ply < temperature_plies else 0.0
        action = choose_action(result, rng, move_temperature)
        records.append(
            PositionRecord.from_search(
                game,
                result,
                game_index=game_index,
                ply=ply,
                played_action=action,
                seed=seed,
            )
        )
        game.make_move(game.decode(action))
        ply += 1
    outcome = int(game.status())
    return [replace(record, final_outcome=outcome) for record in records]


def write_records(path: str | Path, records: Iterable[PositionRecord]) -> int:
    """Write gzip JSON lines atomically enough for bounded batch jobs."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with gzip.open(path, "wt", encoding="utf-8", newline="\n") as stream:
        for record in records:
            stream.write(json.dumps(asdict(record), separators=(",", ":")) + "\n")
            count += 1
    return count


def append_records(path: str | Path, records: Iterable[PositionRecord]) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with gzip.open(path, "at", encoding="utf-8", newline="\n") as stream:
        for record in records:
            stream.write(json.dumps(asdict(record), separators=(",", ":")) + "\n")
            count += 1
    return count


def read_records(path: str | Path) -> Iterator[PositionRecord]:
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                yield PositionRecord(**json.loads(line))


def summarize_records(records: Iterable[PositionRecord]) -> dict[str, float | int]:
    records = list(records)
    if not records:
        return {"positions": 0, "games": 0}
    games = {record.game_index for record in records}
    return {
        "positions": len(records),
        "games": len(games),
        "mean_game_length": len(records) / len(games),
        "mean_search_seconds": sum(r.search_elapsed_s for r in records) / len(records),
        "mean_root_visits": sum(r.root_visits for r in records) / len(records),
        "player1_outcome_fraction": sum(r.final_outcome == 1 for r in records)
        / len(records),
    }

