from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import numpy as np

from breakthrough_zero.agents import RandomAgent
from breakthrough_zero.data import (
    PositionRecord,
    play_self_play_game,
    read_records,
    write_records,
)
from breakthrough_zero.evaluation import (
    evaluate_pair,
    fit_elo_table,
    randomized_openings,
    wilson_interval,
)
from breakthrough_zero.game import Breakthrough
from breakthrough_zero.puct import PUCTPlayer, RolloutEvaluator
from breakthrough_zero.replay import ReplayBuffer, records_to_training_arrays
from breakthrough_zero.training import _tactical_decline_alarms


def sample_record() -> PositionRecord:
    game = Breakthrough.from_rows(
        ["1....", ".1.2.", "..1..", ".2...", "....2"]
    )
    actions = game.legal_actions()
    return PositionRecord(
        game_index=3,
        ply=4,
        board_size=5,
        starting_rows=1,
        board=game.board.copy(),
        player_to_move=game.player_to_move,
        legal_actions=actions,
        visit_counts=list(range(1, len(actions) + 1)),
        priors=[1 / len(actions)] * len(actions),
        root_value=0.25,
        root_visits=sum(range(1, len(actions) + 1)),
        simulations=32,
        search_elapsed_s=0.01,
        played_action=actions[0],
        final_outcome=1,
        seed=9,
    )


class DataTests(unittest.TestCase):
    def test_raw_record_round_trip_reconstructs_position(self) -> None:
        record = sample_record()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "records.jsonl.gz"
            self.assertEqual(write_records(path, [record]), 1)
            loaded = list(read_records(path))
        self.assertEqual(loaded, [record])
        self.assertEqual(loaded[0].state().to_rows(), record.state().to_rows())

    def test_augmentation_removes_canonical_swap_duplicates(self) -> None:
        x, p, z, metrics = records_to_training_arrays([sample_record()], augment=True)
        self.assertEqual(len(x), 2)
        self.assertEqual(metrics["symmetry_duplicates_removed"], 2)
        np.testing.assert_allclose(p.sum(axis=1), 1.0)
        np.testing.assert_array_equal(z, np.ones((2, 1), dtype=np.float32))

    def test_replay_reports_actual_consumption(self) -> None:
        replay = ReplayBuffer(3, seed=1)
        replay.add([sample_record(), sample_record()], iteration=0)
        replay.sample(4)
        metrics = replay.metrics(current_iteration=2)
        self.assertEqual(metrics["size"], 2)
        self.assertEqual(metrics["oldest_age_iterations"], 2)
        self.assertEqual(metrics["replay_consumption_ratio"], 2.0)

    def test_dummy_evaluator_has_uniform_legal_policy_and_absolute_value(self) -> None:
        game = Breakthrough(5, 1)
        priors, value = RolloutEvaluator(seed=12)(game)
        self.assertEqual(set(priors), set(game.legal_actions()))
        self.assertEqual(len(set(priors.values())), 1)
        self.assertIn(value, (-1.0, 1.0))

    def test_self_play_records_keep_reconstructable_search_evidence(self) -> None:
        records = play_self_play_game(
            PUCTPlayer(RolloutEvaluator(seed=5), simulations=2, seed=5),
            board_size=5,
            starting_rows=1,
            game_index=7,
            seed=5,
            temperature_plies=2,
        )
        self.assertGreater(len(records), 0)
        self.assertTrue(all(record.final_outcome in (-1, 1) for record in records))
        for record in records:
            state = record.state()
            self.assertIn(record.played_action, record.legal_actions)
            self.assertEqual(set(record.legal_actions), set(state.legal_actions()))
            self.assertEqual(sum(record.visit_counts), 2)
            self.assertEqual(record.root_visits, 2)

    def test_first_iteration_tactical_decline_is_compared_with_actor(self) -> None:
        actor = {
            "value_accuracy": 5 / 6,
            "policy_accuracy": 1.0,
            "mean_color_swap_absolute_error": 0.0,
        }
        current = {
            "value_accuracy": 4 / 6,
            "policy_accuracy": 5 / 6,
            "mean_color_swap_absolute_error": 0.0,
        }
        alarms = _tactical_decline_alarms(current, actor)
        self.assertEqual(len(alarms), 2)
        self.assertIn("0.833 to 0.667", alarms[0])


class EvaluationTests(unittest.TestCase):
    def test_random_openings_are_distinct_and_reproducible(self) -> None:
        first = randomized_openings(
            5, board_size=5, starting_rows=1, prefix_plies=3, seed=8
        )
        second = randomized_openings(
            5, board_size=5, starting_rows=1, prefix_plies=3, seed=8
        )
        self.assertEqual(first, second)
        self.assertEqual(len({tuple(opening) for opening in first}), 5)

    def test_arena_pairs_every_opening_with_both_colors(self) -> None:
        report = evaluate_pair(
            lambda: RandomAgent(1),
            lambda: RandomAgent(2),
            agent_a_name="random-a",
            agent_b_name="random-b",
            opening_count=3,
            prefix_plies=2,
            seed=99,
        )
        self.assertEqual(report["games_requested"], 6)
        self.assertEqual(report["games_completed"], 6)
        colors = [(game["opening_index"], game["agent_a_player"]) for game in report["games"]]
        self.assertEqual(colors, [(0, 1), (0, -1), (1, 1), (1, -1), (2, 1), (2, -1)])

    def test_wilson_interval_contains_observed_score(self) -> None:
        low, high = wilson_interval(60, 100)
        self.assertLess(low, 0.6)
        self.assertGreater(high, 0.6)

    def test_elo_table_recovers_a_simple_rating_difference(self) -> None:
        table = fit_elo_table(
            [
                {
                    "agent_a": "strong",
                    "agent_b": "anchor",
                    "games_completed": 100,
                    "agent_a_score": 75.0,
                }
            ],
            anchor="anchor",
        )
        ratings = {row["agent"]: row["elo"] for row in table["ratings"]}
        self.assertEqual(ratings["anchor"], 0.0)
        self.assertAlmostEqual(ratings["strong"], 189.5, delta=3.0)


if __name__ == "__main__":
    unittest.main()
