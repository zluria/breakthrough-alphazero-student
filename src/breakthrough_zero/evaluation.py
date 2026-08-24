"""Paired-opening games, score intervals, and Elo estimates."""

import math
import random
import time

import numpy as np

from .game import Breakthrough


def randomized_openings(
    count,
    board_size=5,
    starting_rows=1,
    prefix_plies=4,
):
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


def score_to_elo(score):
    score = max(0.000001, min(0.999999, score))
    return 400 * math.log10(score / (1 - score))


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
    openings = randomized_openings(
        opening_count,
        board_size,
        starting_rows,
        prefix_plies,
    )
    games = []
    for opening_index in range(len(openings)):
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
    sequences = []
    for game in successful:
        score += game["agent_a_score"]
        total_seconds += game["elapsed_s"]
        sequences.append(tuple(game["moves"]))

    if successful:
        rate = score / len(successful)
        mean_seconds = total_seconds / len(successful)
    else:
        rate = 0.0
        mean_seconds = 0.0
    interval = wilson_interval(score, len(successful))
    duplicate_fraction = 1 - len(set(sequences)) / max(1, len(sequences))

    alarms = []
    if duplicate_fraction > 0.25:
        alarms.append("duplicate game fraction is %.3f" % duplicate_fraction)
    if len(successful) >= 40 and rate == 0.5:
        alarms.append("suspiciously exact 50/50 split; inspect paired games")

    elo_difference = None
    elo_interval = None
    if successful:
        elo_difference = score_to_elo(rate)
        elo_interval = [score_to_elo(interval[0]), score_to_elo(interval[1])]
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
        "score_95_interval": list(interval),
        "elo_difference": elo_difference,
        "elo_95_interval": elo_interval,
        "mean_game_seconds": mean_seconds,
        "duplicate_game_fraction": duplicate_fraction,
        "alarms": alarms,
        "games": games,
    }


def fit_elo_table(reports, anchor):
    """Fit one connected Bradley-Terry table, with the anchor fixed at zero."""

    name_set = set()
    for report in reports:
        name_set.add(report["agent_a"])
        name_set.add(report["agent_b"])
    names = sorted(name_set)
    if anchor not in names:
        raise ValueError("anchor is absent from the reports")

    free_names = []
    for name in names:
        if name != anchor:
            free_names.append(name)
    name_index = {}
    for index in range(len(free_names)):
        name_index[free_names[index]] = index
    strengths = np.zeros(len(free_names), dtype=np.float64)

    for unused_round in range(100):
        gradient = np.zeros(len(free_names), dtype=np.float64)
        information = np.eye(len(free_names), dtype=np.float64) * 0.000001
        for report in reports:
            games = int(report["games_completed"])
            if games <= 0:
                continue
            name_a = report["agent_a"]
            name_b = report["agent_b"]
            strength_a = 0.0 if name_a == anchor else strengths[name_index[name_a]]
            strength_b = 0.0 if name_b == anchor else strengths[name_index[name_b]]
            difference = max(-30.0, min(30.0, strength_a - strength_b))
            probability = 1.0 / (1.0 + math.exp(-difference))
            effective_games = games + 1.0
            successes = float(report["agent_a_score"]) + 0.5
            residual = effective_games * probability - successes
            curve = effective_games * probability * (1.0 - probability)

            if name_a != anchor:
                a = name_index[name_a]
                gradient[a] += residual
                information[a, a] += curve
            if name_b != anchor:
                b = name_index[name_b]
                gradient[b] -= residual
                information[b, b] += curve
            if name_a != anchor and name_b != anchor:
                information[a, b] -= curve
                information[b, a] -= curve
        step = np.linalg.solve(information, gradient)
        strengths = strengths - step
        if len(step) == 0 or np.max(np.abs(step)) < 0.000000001:
            break

    covariance = np.linalg.inv(information)
    scale = 400.0 / math.log(10.0)
    ratings = []
    for name in names:
        if name == anchor:
            rating = 0.0
            error = 0.0
        else:
            index = name_index[name]
            rating = float(strengths[index] * scale)
            error = float(math.sqrt(max(0.0, covariance[index, index])) * scale)
        ratings.append(
            {
                "agent": name,
                "elo": rating,
                "standard_error": error,
                "elo_95_interval": [rating - 1.96 * error, rating + 1.96 * error],
            }
        )
    def rating_value(row):
        return row["elo"]

    ratings.sort(key=rating_value, reverse=True)
    return {"anchor": anchor, "reports": len(reports), "ratings": ratings}
