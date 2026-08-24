"""Reconstructable self-play records saved as gzip JSON lines.

Each record keeps the position, MCTS visit counts, and final absolute outcome.
The neural training targets can therefore be reconstructed without playing the
game again. Readers ignore extra fields in older corpus records.
"""

import gzip
import json
import os

import numpy as np

from .game import Breakthrough
from .puct import best_action


def state_from_record(record):
    return Breakthrough(
        record["board_size"],
        record["starting_rows"],
        record["board"],
        record["player_to_move"],
    )


def make_record(game, result, game_index):
    # The parallel action and count lists have an explicit correspondence.
    actions = list(result["visit_counts"])
    counts = []
    for action in actions:
        counts.append(result["visit_counts"][action])

    return {
        "game_index": game_index,
        "board_size": game.board_size,
        "starting_rows": game.starting_rows,
        "board": list(game.board),
        "player_to_move": game.player_to_move,
        "legal_actions": actions,
        "visit_counts": counts,
        "final_outcome": None,
    }


def choose_action(result, temperature):
    """Sample from visit counts raised to ``1 / temperature``.

    Positive temperature encourages varied opening play. At zero temperature,
    the move with the largest visit count is selected.
    """

    counts = result["visit_counts"]
    actions = np.array(sorted(counts), dtype=np.int64)
    if temperature <= 0.00000001:
        return best_action(result)

    weights = []
    for action in actions:
        count = counts[int(action)]
        weights.append(count ** (1.0 / temperature))
    weights = np.array(weights, dtype=np.float64)
    weights = weights / weights.sum()
    return int(np.random.choice(actions, p=weights))


def play_self_play_game(
    search,
    board_size,
    starting_rows,
    game_index,
    temperature=1.0,
    temperature_plies=8,
    add_root_noise=False,
):
    game = Breakthrough(board_size, starting_rows)
    records = []
    ply = 0

    while game.status() is None:
        # Root noise changes which moves MCTS explores. Temperature then controls
        # how sharply the played move follows the resulting visit counts.
        result = search.search(game, add_root_noise)
        if ply < temperature_plies:
            move_temperature = temperature
        else:
            move_temperature = 0.0
        action = choose_action(result, move_temperature)
        record = make_record(game, result, game_index)
        records.append(record)
        game.make_move(game.decode(action))
        ply += 1

    # The value target is the eventual winner, measured in the fixed absolute
    # convention: 1 for Player 1 and -1 for Player 2.
    for record in records:
        record["final_outcome"] = game.status()
    return records


def write_records(path, records):
    directory = os.path.dirname(str(path))
    if directory:
        os.makedirs(directory, exist_ok=True)
    count = 0
    with gzip.open(path, "wt", encoding="utf-8", newline="\n") as stream:
        for record in records:
            stream.write(json.dumps(record, separators=(",", ":")) + "\n")
            count += 1
    return count


def append_records(path, records):
    directory = os.path.dirname(str(path))
    if directory:
        os.makedirs(directory, exist_ok=True)
    count = 0
    with gzip.open(path, "at", encoding="utf-8", newline="\n") as stream:
        for record in records:
            stream.write(json.dumps(record, separators=(",", ":")) + "\n")
            count += 1
    return count


def read_records(path):
    records = []
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                records.append(json.loads(line))
    return records
