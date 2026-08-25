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


def play_parallel_self_play_games(
    search,
    game_count,
    first_game_index,
    board_size,
    starting_rows,
    full_simulations,
    fast_simulations,
    full_search_probability,
    temperature=1.0,
    temperature_plies=8,
):
    """Play several games together so their neural leaves form GPU batches.

    A full search supplies a policy target and receives root noise. A fast
    search only chooses a move. Its position is not saved, because a small
    search should not teach the policy head as if it were a strong target.
    Every saved position still receives the final outcome of its actual game.
    """

    if game_count < 1:
        raise ValueError("game count must be positive")
    if full_search_probability <= 0 or full_search_probability > 1:
        raise ValueError("full search probability must be in (0, 1]")

    games = []
    game_records = []
    plies = []
    for unused_game in range(game_count):
        games.append(Breakthrough(board_size, starting_rows))
        game_records.append([])
        plies.append(0)

    active = list(range(game_count))
    full_searches = 0
    fast_searches = 0
    while active:
        active_games = []
        simulation_counts = []
        use_root_noise = []
        full_search = []

        for game_index in active:
            active_games.append(games[game_index])
            is_full = np.random.random() < full_search_probability
            full_search.append(is_full)
            if is_full:
                simulation_counts.append(full_simulations)
                use_root_noise.append(True)
                full_searches += 1
            else:
                simulation_counts.append(fast_simulations)
                use_root_noise.append(False)
                fast_searches += 1

        results = search.search_batch(
            active_games,
            simulation_counts,
            use_root_noise,
        )
        still_active = []
        for batch_index in range(len(active)):
            game_index = active[batch_index]
            game = games[game_index]
            result = results[batch_index]
            if plies[game_index] < temperature_plies:
                move_temperature = temperature
            else:
                move_temperature = 0.0
            action = choose_action(result, move_temperature)

            if full_search[batch_index]:
                record = make_record(
                    game,
                    result,
                    first_game_index + game_index,
                )
                game_records[game_index].append(record)

            game.make_move(game.decode(action))
            plies[game_index] += 1
            if game.status() is None:
                still_active.append(game_index)
            else:
                for record in game_records[game_index]:
                    record["final_outcome"] = game.status()
        active = still_active

    records = []
    player_1_wins = 0
    for game_index in range(game_count):
        records.extend(game_records[game_index])
        if games[game_index].status() == 1:
            player_1_wins += 1
    return {
        "records": records,
        "games": game_count,
        "positions": len(records),
        "player1_wins": player_1_wins,
        "mean_game_length": sum(plies) / game_count,
        "full_searches": full_searches,
        "fast_searches": fast_searches,
    }


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
