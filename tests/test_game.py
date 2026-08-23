from __future__ import annotations

import random
import unittest

from breakthrough_zero.agents import AlphaBetaAgent, solve_exact
from breakthrough_zero.game import Breakthrough, Move, PLAYER_1, PLAYER_2
from breakthrough_zero.symmetry import SYMMETRIES, Symmetry


def reference_moves(game: Breakthrough) -> set[Move]:
    """Independent, coordinate-based generator used only as a test oracle."""

    if game.status() is not None:
        return set()
    n = game.board_size
    step = 1 if game.player_to_move == PLAYER_1 else -1
    found: set[Move] = set()
    for row in range(n):
        for col in range(n):
            origin = row * n + col
            if game.board[origin] != game.player_to_move:
                continue
            next_row = row + step
            if not 0 <= next_row < n:
                continue
            for next_col in range(max(0, col - 1), min(n, col + 2)):
                destination = next_row * n + next_col
                occupant = game.board[destination]
                if next_col == col and occupant == 0:
                    found.add(Move(origin, destination))
                elif next_col != col and occupant != game.player_to_move:
                    found.add(Move(origin, destination))
    return found


class GameRulesTests(unittest.TestCase):
    def test_initial_5x5_and_native_8x8_setup(self) -> None:
        small = Breakthrough(5, 1)
        standard = Breakthrough(8, 2)
        self.assertEqual(small.board.count(PLAYER_1), 5)
        self.assertEqual(small.board.count(PLAYER_2), 5)
        self.assertEqual(standard.board.count(PLAYER_1), 16)
        self.assertEqual(standard.board.count(PLAYER_2), 16)

    def test_generator_matches_independent_reference_during_games(self) -> None:
        rng = random.Random(20260811)
        for size, rows in ((5, 1), (8, 2)):
            for _ in range(20):
                game = Breakthrough(size, rows)
                while game.status() is None:
                    self.assertEqual(set(game.legal_moves()), reference_moves(game))
                    game.make_move(rng.choice(game.legal_moves()))

    def test_make_unmake_restores_every_field_exactly(self) -> None:
        rng = random.Random(17)
        game = Breakthrough(5, 1)
        for _ in range(40):
            if game.status() is not None:
                break
            before = (game.board.copy(), game.player_to_move, game.winner)
            move = rng.choice(game.legal_moves())
            game.make_move(move)
            game.unmake_move(move)
            self.assertEqual(
                (game.board, game.player_to_move, game.winner), before
            )
            game.make_move(move)

    def test_capture_and_straight_move_restrictions(self) -> None:
        game = Breakthrough.from_rows(
            [".....", "..1..", ".212.", ".....", "....."]
        )
        origin = game.square(1, 2)
        self.assertIn(Move(origin, game.square(2, 1)), game.legal_moves())
        self.assertIn(Move(origin, game.square(2, 3)), game.legal_moves())
        self.assertNotIn(Move(origin, game.square(2, 2)), game.legal_moves())

        empty_diagonals = Breakthrough.from_rows(
            [".....", "..1..", ".....", ".....", "....2"]
        )
        moves = set(empty_diagonals.legal_moves())
        self.assertIn(Move(origin, empty_diagonals.square(2, 1)), moves)
        self.assertIn(Move(origin, empty_diagonals.square(2, 2)), moves)
        self.assertIn(Move(origin, empty_diagonals.square(2, 3)), moves)

    def test_goal_move_is_terminal_and_does_not_switch_player(self) -> None:
        game = Breakthrough.from_rows(
            [".....", "....2", ".....", "..1..", "....."]
        )
        move = Move(game.square(3, 2), game.square(4, 2))
        game.make_move(move)
        self.assertEqual(game.status(), PLAYER_1)
        self.assertEqual(game.player_to_move, PLAYER_1)
        self.assertEqual(game.legal_moves(), [])
        game.unmake_move(move)
        self.assertIsNone(game.status())
        self.assertEqual(game.player_to_move, PLAYER_1)

    def test_no_legal_reply_is_a_win_without_turn_switch(self) -> None:
        game = Breakthrough.from_rows(
            ["22222", "22222", ".....", "..1..", "....."]
        )
        move = Move(game.square(3, 2), game.square(4, 1))
        game.make_move(move)
        self.assertEqual(game.status(), PLAYER_1)
        self.assertEqual(game.player_to_move, PLAYER_1)

    def test_policy_round_trip_for_every_legal_move_in_random_games(self) -> None:
        rng = random.Random(4)
        for size, rows in ((5, 1), (8, 2)):
            game = Breakthrough(size, rows)
            while game.status() is None:
                for move in game.legal_moves():
                    self.assertEqual(game.decode(game.encode_move(move)), move)
                self.assertEqual(int(game.legal_action_mask().sum()), len(game.legal_moves()))
                game.make_move(rng.choice(game.legal_moves()))

    def test_symmetries_are_involutions_and_preserve_legal_moves(self) -> None:
        game = Breakthrough.from_rows(
            ["1....", ".1.2.", "..1..", ".2...", "....2"],
            player_to_move=PLAYER_2,
        )
        for symmetry in SYMMETRIES:
            twice = symmetry.state(symmetry.state(game))
            self.assertEqual(twice.to_rows(), game.to_rows())
            self.assertEqual(twice.player_to_move, game.player_to_move)
            transformed = symmetry.state(game)
            expected = {symmetry.move(move, game.board_size) for move in game.legal_moves()}
            self.assertEqual(set(transformed.legal_moves()), expected)

    def test_swap_and_reflection_commute(self) -> None:
        game = Breakthrough.from_rows(
            ["1....", ".1.2.", "..1..", ".2...", "....2"]
        )
        swap = Symmetry(swap_players=True)
        reflect = Symmetry(reflect_left_right=True)
        left = swap.state(reflect.state(game))
        right = reflect.state(swap.state(game))
        self.assertEqual(left.to_rows(), right.to_rows())
        self.assertEqual(left.player_to_move, right.player_to_move)

    def test_alpha_beta_matches_brute_force_on_tractable_positions(self) -> None:
        positions = [
            Breakthrough.from_rows(["...", ".1.", "..2"], player_to_move=PLAYER_1),
            Breakthrough.from_rows(["1..", ".2.", "..."], player_to_move=PLAYER_2),
            Breakthrough.from_rows([".1.", "..2", "..."], player_to_move=PLAYER_1),
        ]
        agent = AlphaBetaAgent(depth=12)
        for game in positions:
            expected = solve_exact(game.clone())
            value, move = agent.search(game, depth=12)
            game.make_move(move)
            actual = solve_exact(game)
            self.assertEqual(actual, expected)
            self.assertEqual(int(value), expected)


if __name__ == "__main__":
    unittest.main()

