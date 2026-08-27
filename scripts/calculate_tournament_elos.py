"""Calculate repeated-update Elo ratings from arena JSON files."""

import glob
import json
import os
import sys


def read_games(directory):
    games = []
    names = set()
    paths = sorted(glob.glob(os.path.join(directory, "*.json")))
    for path in paths:
        with open(path, "r", encoding="utf-8") as stream:
            report = json.load(stream)
        agent_a = report["agent_a"]
        agent_b = report["agent_b"]
        names.add(agent_a)
        names.add(agent_b)
        for game in report["games"]:
            games.append((agent_a, agent_b, game["agent_a_score"]))
    return sorted(names), games


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
        "ratings": [],
    }
    for name, rating in ordered:
        result["ratings"].append({"name": name, "elo": rating})
    return result


def main():
    if len(sys.argv) < 2:
        raise SystemExit("usage: calculate_tournament_elos.py DIRECTORY")
    names, games = read_games(sys.argv[1])
    if not games:
        raise SystemExit("no arena games found")
    result = calculate_elos(names, games)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
