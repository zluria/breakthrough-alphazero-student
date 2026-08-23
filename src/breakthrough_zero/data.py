"""Plain-dictionary self-play records saved as gzip JSON lines."""

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


def make_record(game, result, game_index, ply, played_action, seed):
    actions = sorted(result["visit_counts"])
    counts = []
    priors = []
    for action in actions:
        counts.append(result["visit_counts"][action])
        priors.append(result["priors"][action])

    return {
        "game_index": game_index,
        "ply": ply,
        "board_size": game.board_size,
        "starting_rows": game.starting_rows,
        "board": list(game.board),
        "player_to_move": game.player_to_move,
        "legal_actions": actions,
        "visit_counts": counts,
        "priors": priors,
        "root_value": result["root_value"],
        "root_visits": sum(counts),
        "simulations": result["simulations"],
        "search_elapsed_s": result["elapsed_s"],
        "played_action": played_action,
        "final_outcome": None,
        "seed": seed,
    }


def choose_action(result, random_generator, temperature):
    counts = result["visit_counts"]
    actions = np.array(sorted(counts), dtype=np.int64)
    if temperature <= 0.00000001:
        return best_action(result)

    weights = []
    for action in actions:
        count = max(counts[int(action)], 0.000000000001)
        weights.append(count ** (1.0 / temperature))
    weights = np.array(weights, dtype=np.float64)
    weights = weights / weights.sum()
    return int(random_generator.choice(actions, p=weights))


def play_self_play_game(
    search,
    board_size,
    starting_rows,
    game_index,
    seed,
    temperature=1.0,
    temperature_plies=8,
    add_root_noise=False,
):
    game = Breakthrough(board_size, starting_rows)
    random_generator = np.random.default_rng(seed)
    records = []
    ply = 0

    while game.status() is None:
        result = search.search(game, add_root_noise)
        if ply < temperature_plies:
            move_temperature = temperature
        else:
            move_temperature = 0.0
        action = choose_action(result, random_generator, move_temperature)
        record = make_record(game, result, game_index, ply, action, seed)
        records.append(record)
        game.make_move(game.decode(action))
        ply += 1

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


def summarize_records(records):
    if not records:
        return {"positions": 0, "games": 0}

    game_numbers = set()
    total_search_seconds = 0.0
    total_root_visits = 0
    player_1_positions = 0
    for record in records:
        game_numbers.add(record["game_index"])
        total_search_seconds += record["search_elapsed_s"]
        total_root_visits += record["root_visits"]
        if record["final_outcome"] == 1:
            player_1_positions += 1

    return {
        "positions": len(records),
        "games": len(game_numbers),
        "mean_game_length": len(records) / len(game_numbers),
        "mean_search_seconds": total_search_seconds / len(records),
        "mean_root_visits": total_root_visits / len(records),
        "player1_outcome_fraction": player_1_positions / len(records),
    }
