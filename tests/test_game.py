import math
import random
import unittest

import breakthrough_zero.agents as agents_module
from breakthrough_zero.agents import AlphaBetaAgent, evaluate_position, solve_exact
from breakthrough_zero.evaluation import evaluate_pair
from breakthrough_zero.game import (
    Breakthrough,
    PLAYER_1,
    PLAYER_2,
    game_from_rows,
)
from breakthrough_zero.symmetry import (
    SYMMETRIES,
    transform_move,
    transform_state,
)


def reference_moves(game):
    """A second move generator used only by the tests."""

    if game.status() is not None:
        return set()
    size = game.board_size
    if game.player_to_move == PLAYER_1:
        step = 1
    else:
        step = -1
    found = set()

    for row in range(size):
        for col in range(size):
            from_square = row * size + col
            if game.board[from_square] != game.player_to_move:
                continue
            next_row = row + step
            if next_row < 0 or next_row >= size:
                continue
            first_col = max(0, col - 1)
            last_col = min(size, col + 2)
            for next_col in range(first_col, last_col):
                to_square = next_row * size + next_col
                target = game.board[to_square]
                if next_col == col and target == 0:
                    found.add((from_square, to_square))
                if next_col != col and target != game.player_to_move:
                    found.add((from_square, to_square))
    return found


class GameRulesTests(unittest.TestCase):
    def test_initial_5x5_and_native_8x8_setup(self):
        small = Breakthrough(5, 1)
        standard = Breakthrough(8, 2)
        self.assertEqual(small.board.count(PLAYER_1), 5)
        self.assertEqual(small.board.count(PLAYER_2), 5)
        self.assertEqual(standard.board.count(PLAYER_1), 16)
        self.assertEqual(standard.board.count(PLAYER_2), 16)
        self.assertEqual(small.action_size, 75)
        self.assertEqual(standard.action_size, 192)

    def test_generator_matches_independent_reference_during_games(self):
        random_generator = random.Random()
        for size, rows in ((5, 1), (8, 2)):
            for unused_game in range(20):
                game = Breakthrough(size, rows)
                while game.status() is None:
                    self.assertEqual(set(game.legal_moves()), reference_moves(game))
                    game.make_move(random_generator.choice(game.legal_moves()))

    def test_make_unmake_restores_every_field_exactly(self):
        random_generator = random.Random()
        game = Breakthrough(5, 1)
        for unused_move in range(40):
            if game.status() is not None:
                break
            before = (list(game.board), game.player_to_move, game.winner)
            move = random_generator.choice(game.legal_moves())
            game.make_move(move)
            clone = game.clone()
            self.assertEqual(clone.history, [])
            self.assertEqual(clone.board, game.board)
            game.unmake_move()
            after = (game.board, game.player_to_move, game.winner)
            self.assertEqual(after, before)
            game.make_move(move)

    def test_capture_and_straight_move_restrictions(self):
        game = game_from_rows([".....", "..1..", ".212.", ".....", "....."])
        origin = game.square(1, 2)
        self.assertIn((origin, game.square(2, 1)), game.legal_moves())
        self.assertIn((origin, game.square(2, 3)), game.legal_moves())
        self.assertNotIn((origin, game.square(2, 2)), game.legal_moves())

        game = game_from_rows([".....", "..1..", ".....", ".....", "....2"])
        moves = set(game.legal_moves())
        self.assertIn((origin, game.square(2, 1)), moves)
        self.assertIn((origin, game.square(2, 2)), moves)
        self.assertIn((origin, game.square(2, 3)), moves)

    def test_goal_move_is_terminal_and_does_not_switch_player(self):
        game = game_from_rows([".....", "....2", ".....", "..1..", "....."])
        move = (game.square(3, 2), game.square(4, 2))
        game.make_move(move)
        self.assertEqual(game.status(), PLAYER_1)
        self.assertEqual(game.player_to_move, PLAYER_1)
        self.assertEqual(game.legal_moves(), [])
        game.unmake_move()
        self.assertIsNone(game.status())
        self.assertEqual(game.player_to_move, PLAYER_1)

    def test_no_legal_reply_is_a_win_without_turn_switch(self):
        game = game_from_rows(["22222", "22222", ".....", "..1..", "....."])
        move = (game.square(3, 2), game.square(4, 1))
        game.make_move(move)
        self.assertEqual(game.status(), PLAYER_1)
        self.assertEqual(game.player_to_move, PLAYER_1)

    def test_policy_round_trip_for_every_legal_move_in_random_games(self):
        random_generator = random.Random()
        for size, rows in ((5, 1), (8, 2)):
            game = Breakthrough(size, rows)
            while game.status() is None:
                for move in game.legal_moves():
                    action = game.encode_move(move)
                    self.assertEqual(game.decode(action), move)
                count = int(game.legal_action_mask().sum())
                self.assertEqual(count, len(game.legal_moves()))
                game.make_move(random_generator.choice(game.legal_moves()))

    def test_symmetries_are_involutions_and_preserve_legal_moves(self):
        game = game_from_rows(
            ["1....", ".1.2.", "..1..", ".2...", "....2"],
            PLAYER_2,
        )
        for symmetry in SYMMETRIES:
            twice = transform_state(transform_state(game, symmetry), symmetry)
            self.assertEqual(twice.to_rows(), game.to_rows())
            self.assertEqual(twice.player_to_move, game.player_to_move)

            transformed = transform_state(game, symmetry)
            expected = set()
            for move in game.legal_moves():
                expected.add(transform_move(move, game.board_size, symmetry))
            self.assertEqual(set(transformed.legal_moves()), expected)

    def test_swap_and_reflection_commute(self):
        game = game_from_rows(["1....", ".1.2.", "..1..", ".2...", "....2"])
        left = transform_state(transform_state(game, (False, True)), (True, False))
        right = transform_state(transform_state(game, (True, False)), (False, True))
        self.assertEqual(left.to_rows(), right.to_rows())
        self.assertEqual(left.player_to_move, right.player_to_move)

    def test_alpha_beta_matches_brute_force_on_tractable_positions(self):
        positions = [
            game_from_rows(["...", ".1.", "..2"], PLAYER_1),
            game_from_rows(["1..", ".2.", "..."], PLAYER_2),
            game_from_rows([".1.", "..2", "..."], PLAYER_1),
        ]
        agent = AlphaBetaAgent(12)
        for game in positions:
            expected = solve_exact(game.clone())
            value, move = agent.search(game, 12)
            game.make_move(move)
            actual = solve_exact(game)
            self.assertEqual(actual, expected)
            value_sign = 1 if value > 0 else -1
            self.assertEqual(value_sign, expected)

    def test_alpha_beta_terminal_scores_and_arena_scores_use_both_colors(self):
        player_1_win = game_from_rows(
            [".....", "....2", ".....", "..1..", "....."], PLAYER_1
        )
        player_2_win = game_from_rows(
            [".....", "..2..", ".....", "1....", "....."], PLAYER_2
        )
        agent = AlphaBetaAgent(2)
        for game, expected in (
            (player_1_win, PLAYER_1),
            (player_2_win, PLAYER_2),
        ):
            value, move = agent.search(game, 2)
            game.make_move(move)
            self.assertEqual(game.status(), expected)
            expected_value = math.inf if expected == PLAYER_1 else -math.inf
            self.assertEqual(value, expected_value)

        report = evaluate_pair(
            AlphaBetaAgent(8),
            AlphaBetaAgent(8),
            "alpha-a",
            "alpha-b",
            1,
            0,
            3,
            1,
        )
        for game in report["games"]:
            expected_score = float(game["winner"] == game["agent_a_player"])
            self.assertEqual(game["agent_a_score"], expected_score)

    def test_alpha_beta_heuristic_preserves_strong_position_order(self):
        large_player_1_advantage = game_from_rows(
            ["11111", "11111", "11111", "11111", "....2"]
        )
        smaller_player_1_advantage = game_from_rows(
            ["11111", "11111", "11111", ".....", "....2"]
        )
        large_player_2_advantage = game_from_rows(
            ["1....", "22222", "22222", "22222", "22222"]
        )
        smaller_player_2_advantage = game_from_rows(
            ["1....", ".....", "22222", "22222", "22222"]
        )

        large_positive = evaluate_position(large_player_1_advantage)
        smaller_positive = evaluate_position(smaller_player_1_advantage)
        large_negative = evaluate_position(large_player_2_advantage)
        smaller_negative = evaluate_position(smaller_player_2_advantage)

        self.assertTrue(math.isfinite(large_positive))
        self.assertGreater(large_positive, smaller_positive)
        self.assertTrue(math.isfinite(large_negative))
        self.assertLess(large_negative, smaller_negative)

    def test_alpha_beta_timeout_restores_position(self):
        game = Breakthrough(5, 1)
        before = (list(game.board), game.player_to_move, game.winner)
        agent = AlphaBetaAgent(3)
        agent.deadline = 1.0
        times = [0.0, 2.0]

        def fake_time():
            return times.pop(0)

        real_time = agents_module.time.perf_counter
        agents_module.time.perf_counter = fake_time
        try:
            with self.assertRaises(TimeoutError):
                agent.alpha_beta(game, 3, float("-inf"), float("inf"))
        finally:
            agents_module.time.perf_counter = real_time
        after = (game.board, game.player_to_move, game.winner)
        self.assertEqual(after, before)


if __name__ == "__main__":
    unittest.main()
