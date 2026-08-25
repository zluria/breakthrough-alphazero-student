import os
import tempfile
import unittest

import numpy as np

from breakthrough_zero.agents import RandomAgent, solve_exact
from breakthrough_zero.data import (
    play_parallel_self_play_games,
    play_self_play_game,
    read_records,
    state_from_record,
    write_records,
)
from breakthrough_zero.diagnostics import tactical_suite
from breakthrough_zero.evaluation import (
    evaluate_pair,
    randomized_openings,
    wilson_interval,
)
from breakthrough_zero.game import game_from_rows
from breakthrough_zero.puct import PUCTPlayer, RolloutEvaluator
from breakthrough_zero.replay import ReplayBuffer, records_to_training_arrays
from breakthrough_zero.symmetry import transform_state


def sample_record(game_index=3):
    game = game_from_rows(["1....", ".1.2.", "..1..", ".2...", "....2"])
    actions = game.legal_actions()
    counts = list(range(1, len(actions) + 1))
    return {
        "game_index": game_index,
        "board_size": 5,
        "starting_rows": 1,
        "board": list(game.board),
        "player_to_move": game.player_to_move,
        "legal_actions": actions,
        "visit_counts": counts,
        "final_outcome": 1,
    }


class DataTests(unittest.TestCase):
    def test_raw_record_round_trip_reconstructs_position(self):
        record = sample_record()
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "records.jsonl.gz")
            self.assertEqual(write_records(path, [record]), 1)
            loaded = read_records(path)
        self.assertEqual(loaded, [record])
        loaded_game = state_from_record(loaded[0])
        original_game = state_from_record(record)
        self.assertEqual(loaded_game.to_rows(), original_game.to_rows())

    def test_augmentation_uses_original_and_reflection(self):
        record = sample_record()
        record["old_root_value"] = 0.25
        inputs, policies, values = records_to_training_arrays([record])
        self.assertEqual(len(inputs), 2)
        np.testing.assert_allclose(policies.sum(axis=1), 1.0)
        np.testing.assert_array_equal(
            values, np.ones((2, 1), dtype=np.float32)
        )

    def test_replay_keeps_capacity_and_samples_without_replacement(self):
        replay = ReplayBuffer(3)
        replay.add([sample_record(0), sample_record(1)])
        replay.add([sample_record(2), sample_record(3)])
        self.assertEqual([record["game_index"] for record in replay.data], [1, 2, 3])
        sample = replay.sample(3)
        self.assertEqual(len({record["game_index"] for record in sample}), 3)

    def test_dummy_evaluator_has_uniform_legal_policy_and_absolute_value(self):
        game = state_from_record(sample_record())
        priors, value = RolloutEvaluator().evaluate(game)
        self.assertEqual(set(priors), set(game.legal_actions()))
        self.assertEqual(len(set(priors.values())), 1)
        self.assertIn(value, (-1.0, 1.0))

    def test_self_play_records_keep_training_evidence(self):
        player = PUCTPlayer(RolloutEvaluator(), 2, 1.5)
        records = play_self_play_game(player, 5, 1, 7, 1.0, 2)
        self.assertGreater(len(records), 0)
        for record in records:
            self.assertEqual(
                set(record),
                {
                    "game_index",
                    "board_size",
                    "starting_rows",
                    "board",
                    "player_to_move",
                    "legal_actions",
                    "visit_counts",
                    "final_outcome",
                },
            )
            self.assertIn(record["final_outcome"], (-1, 1))
            game = state_from_record(record)
            self.assertEqual(
                set(record["legal_actions"]), set(game.legal_actions())
            )
            self.assertEqual(sum(record["visit_counts"]), 2)

    def test_parallel_self_play_saves_only_full_search_positions(self):
        player = PUCTPlayer(RolloutEvaluator(), 3, 1.5)
        report = play_parallel_self_play_games(
            player,
            3,
            10,
            5,
            1,
            3,
            1,
            1.0,
            1.0,
            2,
        )
        self.assertEqual(report["games"], 3)
        self.assertEqual(report["fast_searches"], 0)
        self.assertGreater(report["positions"], 0)
        for record in report["records"]:
            self.assertEqual(sum(record["visit_counts"]), 3)
            self.assertIn(record["game_index"], (10, 11, 12))

    def test_tactical_suite_has_twenty_balanced_exact_base_positions(self):
        positions = tactical_suite()
        self.assertEqual(len(positions), 20)

        positive = 0
        negative = 0
        categories = set()
        for position in positions:
            categories.add(position["category"])
            if position["outcome"] == 1:
                positive += 1
            else:
                negative += 1
            exact = solve_exact(position["game"].clone())
            self.assertEqual(exact, position["outcome"], position["name"])

            swapped = transform_state(position["game"], (True, False))
            self.assertEqual(solve_exact(swapped), -position["outcome"])

        self.assertEqual((positive, negative), (10, 10))
        self.assertEqual(
            categories,
            {
                "immediate wins",
                "immediate threats requiring defense",
                "forced wins and losses",
                "material advantages",
                "advanced passed pawns",
            },
        )


class EvaluationTests(unittest.TestCase):
    def test_random_openings_are_distinct(self):
        openings = randomized_openings(5, 5, 1, 3)
        unique = set()
        for opening in openings:
            unique.add(tuple(opening))
        self.assertEqual(len(unique), 5)

    def test_arena_pairs_every_opening_with_both_colors(self):
        report = evaluate_pair(
            RandomAgent(),
            RandomAgent(),
            "random-a",
            "random-b",
            3,
            2,
            5,
            1,
        )
        self.assertEqual(report["games_requested"], 6)
        self.assertEqual(report["games_completed"], 6)
        colors = []
        for game in report["games"]:
            colors.append((game["opening_index"], game["agent_a_player"]))
        expected = [(0, 1), (0, -1), (1, 1), (1, -1), (2, 1), (2, -1)]
        self.assertEqual(colors, expected)

    def test_wilson_interval_contains_observed_score(self):
        low, high = wilson_interval(60, 100)
        self.assertLess(low, 0.6)
        self.assertGreater(high, 0.6)

if __name__ == "__main__":
    unittest.main()
