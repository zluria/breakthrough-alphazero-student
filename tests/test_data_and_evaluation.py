import os
import tempfile
import unittest

import numpy as np

from breakthrough_zero.agents import RandomAgent, solve_exact
from breakthrough_zero.data import (
    play_self_play_game,
    read_records,
    state_from_record,
    write_records,
)
from breakthrough_zero.diagnostics import tactical_suite
from breakthrough_zero.evaluation import (
    evaluate_pair,
    fit_elo_table,
    randomized_openings,
    wilson_interval,
)
from breakthrough_zero.game import game_from_rows
from breakthrough_zero.puct import PUCTPlayer, RolloutEvaluator
from breakthrough_zero.replay import ReplayBuffer, records_to_training_arrays
from breakthrough_zero.training import tactical_decline_alarms


def sample_record():
    game = game_from_rows(["1....", ".1.2.", "..1..", ".2...", "....2"])
    actions = game.legal_actions()
    counts = list(range(1, len(actions) + 1))
    priors = [1 / len(actions)] * len(actions)
    return {
        "game_index": 3,
        "ply": 4,
        "board_size": 5,
        "starting_rows": 1,
        "board": list(game.board),
        "player_to_move": game.player_to_move,
        "legal_actions": actions,
        "visit_counts": counts,
        "priors": priors,
        "root_value": 0.25,
        "root_visits": sum(counts),
        "simulations": 32,
        "search_elapsed_s": 0.01,
        "played_action": actions[0],
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

    def test_augmentation_removes_canonical_swap_duplicates(self):
        inputs, policies, values, metrics = records_to_training_arrays(
            [sample_record()], True
        )
        self.assertEqual(len(inputs), 2)
        self.assertEqual(metrics["symmetry_duplicates_removed"], 2)
        np.testing.assert_allclose(policies.sum(axis=1), 1.0)
        np.testing.assert_array_equal(
            values, np.ones((2, 1), dtype=np.float32)
        )

    def test_replay_reports_actual_consumption(self):
        replay = ReplayBuffer(3)
        replay.add([sample_record(), sample_record()], 0)
        replay.sample(4)
        metrics = replay.metrics(2)
        self.assertEqual(metrics["size"], 2)
        self.assertEqual(metrics["oldest_age_iterations"], 2)
        self.assertEqual(metrics["replay_consumption_ratio"], 2.0)

    def test_dummy_evaluator_has_uniform_legal_policy_and_absolute_value(self):
        game = state_from_record(sample_record())
        priors, value = RolloutEvaluator().evaluate(game)
        self.assertEqual(set(priors), set(game.legal_actions()))
        self.assertEqual(len(set(priors.values())), 1)
        self.assertIn(value, (-1.0, 1.0))

    def test_self_play_records_keep_reconstructable_search_evidence(self):
        player = PUCTPlayer(RolloutEvaluator(), 2, 1.5)
        records = play_self_play_game(player, 5, 1, 7, 1.0, 2)
        self.assertGreater(len(records), 0)
        for record in records:
            self.assertIn(record["final_outcome"], (-1, 1))
            game = state_from_record(record)
            self.assertIn(record["played_action"], record["legal_actions"])
            self.assertEqual(
                set(record["legal_actions"]), set(game.legal_actions())
            )
            self.assertEqual(sum(record["visit_counts"]), 2)
            self.assertEqual(record["root_visits"], 2)

    def test_first_iteration_tactical_decline_is_compared_with_actor(self):
        actor = {
            "mean_signed_value": 0.30,
            "policy_accuracy": 1.0,
            "mean_color_swap_absolute_error": 0.0,
        }
        current = {
            "mean_signed_value": 0.20,
            "policy_accuracy": 0.8,
            "mean_color_swap_absolute_error": 0.0,
        }
        alarms = tactical_decline_alarms(current, actor)
        self.assertEqual(len(alarms), 2)
        self.assertIn("0.300 to 0.200", alarms[0])

    def test_tactical_suite_has_twenty_balanced_exact_base_positions(self):
        positions = tactical_suite()
        self.assertEqual(len(positions), 40)

        base_positions = []
        by_name = {}
        for position in positions:
            by_name[position["name"]] = position
            if not position["name"].endswith("_swapped"):
                base_positions.append(position)
        self.assertEqual(len(base_positions), 20)

        positive = 0
        negative = 0
        categories = set()
        for position in base_positions:
            categories.add(position["category"])
            if position["outcome"] == 1:
                positive += 1
            else:
                negative += 1
            exact = solve_exact(position["game"].clone())
            self.assertEqual(exact, position["outcome"], position["name"])

            swapped = by_name[position["name"] + "_swapped"]
            self.assertEqual(swapped["outcome"], -position["outcome"])

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

    def test_elo_table_recovers_a_simple_rating_difference(self):
        reports = [
            {
                "agent_a": "strong",
                "agent_b": "anchor",
                "games_completed": 100,
                "agent_a_score": 75.0,
            }
        ]
        table = fit_elo_table(reports, "anchor")
        ratings = {}
        for row in table["ratings"]:
            ratings[row["agent"]] = row["elo"]
        self.assertEqual(ratings["anchor"], 0.0)
        self.assertAlmostEqual(ratings["strong"], 189.5, delta=3.0)


if __name__ == "__main__":
    unittest.main()
