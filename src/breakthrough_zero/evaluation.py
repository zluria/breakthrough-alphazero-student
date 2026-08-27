"""Paired-opening games, score intervals, and Elo estimates."""

import math
import random
import time

from .game import Breakthrough


def randomized_openings(
    count,
    board_size=5,
    starting_rows=1,
    prefix_plies=4,
):
    """Generate distinct nonterminal opening prefixes for paired games."""

    openings = []
    seen = set()
    attempts = 0

    while len(openings) < count:
        attempts += 1
        if attempts > count * 100:
            raise RuntimeError("could not produce enough distinct openings")
        game = Breakthrough(board_size, starting_rows)
        moves = []
        for unused_ply in range(prefix_plies):
            if game.status() is not None:
                break
            move = random.choice(game.legal_moves())
            moves.append(move)
            game.make_move(move)
        if game.status() is not None:
            continue
        key = tuple(moves)
        if key not in seen:
            seen.add(key)
            openings.append(moves)
    return openings


def play_arena_game(
    agent_a,
    agent_b,
    agent_a_player,
    opening,
    opening_index,
    board_size,
    starting_rows,
):
    game = Breakthrough(board_size, starting_rows)
    move_log = []
    started = time.perf_counter()
    winner = None
    failure = None
    score = 0.0

    try:
        for move in opening:
            game.make_move(move)
            move_log.append(move)
        while game.status() is None:
            if game.player_to_move == agent_a_player:
                agent = agent_a
            else:
                agent = agent_b
            move = agent.choose_move(game)
            game.make_move(move)
            move_log.append(move)
        winner = game.status()
        if winner == agent_a_player:
            score = 1.0
    except Exception as error:
        failure = error.__class__.__name__ + ": " + str(error)

    return {
        "opening_index": opening_index,
        "agent_a_player": agent_a_player,
        "winner": winner,
        "agent_a_score": score,
        "moves": move_log,
        "elapsed_s": time.perf_counter() - started,
        "failure": failure,
    }


def wilson_interval(successes, games, z=1.96):
    """Return a binomial score interval that remains useful for small samples."""

    if games < 1:
        return (0.0, 1.0)
    proportion = successes / games
    denominator = 1 + z * z / games
    center = (proportion + z * z / (2 * games)) / denominator
    inside = (
        proportion * (1 - proportion) / games
        + z * z / (4 * games * games)
    )
    margin = z * math.sqrt(inside) / denominator
    return (max(0.0, center - margin), min(1.0, center + margin))


def evaluate_pair(
    agent_a,
    agent_b,
    agent_a_name,
    agent_b_name,
    opening_count=50,
    prefix_plies=4,
    board_size=5,
    starting_rows=1,
):
    """Play every opening twice, reversing the agents' colors."""

    openings = randomized_openings(
        opening_count,
        board_size,
        starting_rows,
        prefix_plies,
    )
    games = []
    for opening_index in range(len(openings)):
        # Using the identical opening with both color assignments separates agent
        # strength from first-player advantage and the luck of one opening draw.
        opening = openings[opening_index]
        game = play_arena_game(
            agent_a,
            agent_b,
            1,
            opening,
            opening_index,
            board_size,
            starting_rows,
        )
        games.append(game)
        game = play_arena_game(
            agent_a,
            agent_b,
            -1,
            opening,
            opening_index,
            board_size,
            starting_rows,
        )
        games.append(game)

    successful = []
    for game in games:
        if game["failure"] is None:
            successful.append(game)
    score = 0.0
    total_seconds = 0.0
    for game in successful:
        score += game["agent_a_score"]
        total_seconds += game["elapsed_s"]

    if successful:
        rate = score / len(successful)
        mean_seconds = total_seconds / len(successful)
    else:
        rate = 0.0
        mean_seconds = 0.0
    interval = wilson_interval(score, len(successful))
    return {
        "agent_a": agent_a_name,
        "agent_b": agent_b_name,
        "board_size": board_size,
        "opening_count": opening_count,
        "prefix_plies": prefix_plies,
        "games_requested": len(games),
        "games_completed": len(successful),
        "failures": len(games) - len(successful),
        "agent_a_score": score,
        "agent_a_score_rate": rate,
        "score_95_interval": list(interval),
        "mean_game_seconds": mean_seconds,
        "games": games,
    }
