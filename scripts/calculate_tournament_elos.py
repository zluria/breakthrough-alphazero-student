"""Calculate standings and repeated-update Elo ratings from arena reports."""

import glob
import json
import os
import sys


def read_games(directories):
    if isinstance(directories, str):
        directories = [directories]

    games = []
    names = set()
    reports = 0
    openings = set()
    move_logs = set()

    for directory in directories:
        paths = sorted(glob.glob(os.path.join(directory, "*.json")))
        for path in paths:
            with open(path, "r", encoding="utf-8") as stream:
                report = json.load(stream)
            reports += 1
            agent_a = report["agent_a"]
            agent_b = report["agent_b"]
            names.add(agent_a)
            names.add(agent_b)

            if report["failures"]:
                raise ValueError(path + " contains failed games")
            opening_players = {}
            for game in report["games"]:
                if game["failure"] is not None:
                    raise ValueError(path + " contains a failed game")
                games.append((agent_a, agent_b, game["agent_a_score"]))
                opening_length = report.get("prefix_plies", 6)
                opening = game["moves"][:opening_length]
                openings.add(tuple(tuple(move) for move in opening))
                move_logs.add(tuple(tuple(move) for move in game["moves"]))
                opening_index = game["opening_index"]
                if opening_index not in opening_players:
                    opening_players[opening_index] = set()
                opening_players[opening_index].add(game["agent_a_player"])
            for players in opening_players.values():
                if players != {1, -1}:
                    raise ValueError(path + " does not reverse every opening")

    if len(openings) * 2 != len(games):
        raise ValueError("the tournament reused an opening prefix")
    if len(move_logs) != len(games):
        raise ValueError("the tournament produced duplicate complete games")

    audit = {
        "directories": directories,
        "reports": reports,
        "games": len(games),
        "unique_openings": len(openings),
        "unique_complete_games": len(move_logs),
    }
    return sorted(names), games, audit


def calculate_elos(names, games, passes=100, k_factor=32):
    ratings = {}
    for name in names:
        ratings[name] = 1500.0

    last_pass_change = 0.0
    for unused_pass in range(passes):
        before = dict(ratings)
        for agent_a, agent_b, score_a in games:
            rating_a = ratings[agent_a]
            rating_b = ratings[agent_b]
            expected_a = 1 / (1 + 10 ** ((rating_b - rating_a) / 400))
            change = k_factor * (score_a - expected_a)
            ratings[agent_a] += change
            ratings[agent_b] -= change

        last_pass_change = 0.0
        for name in names:
            change = abs(ratings[name] - before[name])
            if change > last_pass_change:
                last_pass_change = change

    ordered = sorted(ratings.items(), key=lambda item: item[1], reverse=True)
    result = {
        "passes": passes,
        "k_factor": k_factor,
        "games_per_pass": len(games),
        "largest_change_in_last_pass": last_pass_change,
        "converged": last_pass_change < 0.001,
        "ratings": [],
    }
    for name, rating in ordered:
        result["ratings"].append({"name": name, "elo": rating})
    return result


def calculate_standings(names, games, ratings):
    scores = {}
    games_played = {}
    for name in names:
        scores[name] = 0.0
        games_played[name] = 0

    for agent_a, agent_b, score_a in games:
        scores[agent_a] += score_a
        scores[agent_b] += 1.0 - score_a
        games_played[agent_a] += 1
        games_played[agent_b] += 1

    elo_by_name = {}
    for item in ratings:
        elo_by_name[item["name"]] = item["elo"]
    ordered = sorted(
        names,
        key=lambda name: (scores[name], elo_by_name[name]),
        reverse=True,
    )

    standings = []
    for name in ordered:
        standings.append(
            {
                "name": name,
                "score": scores[name],
                "games": games_played[name],
                "elo": elo_by_name[name],
            }
        )
    return standings


def parse_arguments(arguments):
    directories = []
    passes = 100
    ranking_path = None
    index = 0
    while index < len(arguments):
        argument = arguments[index]
        if argument == "--passes":
            index += 1
            passes = int(arguments[index])
        elif argument == "--ranking":
            index += 1
            ranking_path = arguments[index]
        else:
            directories.append(argument)
        index += 1
    if not directories:
        raise ValueError("at least one arena-report directory is required")
    if passes < 1:
        raise ValueError("passes must be positive")
    return directories, passes, ranking_path


def main(arguments=None):
    if arguments is None:
        arguments = sys.argv[1:]
    try:
        directories, passes, ranking_path = parse_arguments(arguments)
        names, games, audit = read_games(directories)
    except (IndexError, ValueError) as error:
        raise SystemExit(str(error))
    if not games:
        raise SystemExit("no arena games found")

    result = calculate_elos(names, games, passes)
    result["standings"] = calculate_standings(
        names, games, result["ratings"]
    )
    result["audit"] = audit

    if ranking_path is not None:
        with open(ranking_path, "w", encoding="utf-8") as stream:
            for item in result["standings"]:
                stream.write(item["name"] + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
