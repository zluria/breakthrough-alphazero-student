import json
import os
import tempfile
import unittest

from scripts.calculate_tournament_elos import (
    calculate_elos,
    calculate_standings,
    read_games,
)


class TournamentEloTests(unittest.TestCase):
    def test_repeated_elo_starts_every_player_at_1500(self):
        result = calculate_elos(["a", "b"], [("a", "b", 1.0)], passes=1)
        ratings = {}
        for item in result["ratings"]:
            ratings[item["name"]] = item["elo"]
        self.assertEqual(ratings, {"a": 1516.0, "b": 1484.0})

    def test_standings_use_score_then_elo_to_break_ties(self):
        names = ["a", "b", "c"]
        games = [("a", "b", 1.0), ("c", "a", 1.0), ("b", "c", 1.0)]
        ratings = [
            {"name": "a", "elo": 1500.0},
            {"name": "b", "elo": 1490.0},
            {"name": "c", "elo": 1510.0},
        ]
        standings = calculate_standings(names, games, ratings)
        self.assertEqual([item["name"] for item in standings], ["c", "a", "b"])

    def test_reader_combines_directories_and_audits_games(self):
        with tempfile.TemporaryDirectory() as directory:
            first = os.path.join(directory, "first")
            second = os.path.join(directory, "second")
            os.makedirs(first)
            os.makedirs(second)
            self.write_report(os.path.join(first, "a-v-b.json"), "a", "b", 0)
            self.write_report(os.path.join(second, "b-v-c.json"), "b", "c", 10)

            names, games, audit = read_games([first, second])

            self.assertEqual(names, ["a", "b", "c"])
            self.assertEqual(len(games), 4)
            self.assertEqual(audit["reports"], 2)
            self.assertEqual(audit["unique_openings"], 2)
            self.assertEqual(audit["unique_complete_games"], 4)

    def write_report(self, path, agent_a, agent_b, offset):
        opening = [[offset, offset + 1]] * 6
        games = []
        for player in (1, -1):
            moves = list(opening)
            moves.append([offset + player + 20, offset + player + 21])
            games.append(
                {
                    "opening_index": 0,
                    "agent_a_player": player,
                    "agent_a_score": 1.0 if player == 1 else 0.0,
                    "moves": moves,
                    "failure": None,
                }
            )
        report = {
            "agent_a": agent_a,
            "agent_b": agent_b,
            "failures": 0,
            "games": games,
        }
        with open(path, "w", encoding="utf-8") as stream:
            json.dump(report, stream)


if __name__ == "__main__":
    unittest.main()
