from __future__ import annotations

import unittest

from breakthrough_zero.agents import AlphaBetaAgent, solve_exact
from breakthrough_zero.game import Breakthrough, PLAYER_1, PLAYER_2
from breakthrough_zero.puct import PUCTNode, PUCTPlayer
from breakthrough_zero.symmetry import Symmetry


class ExactEvaluator:
    def __call__(self, game: Breakthrough) -> tuple[dict[int, float], float]:
        actions = game.legal_actions()
        priors = {action: 1 / len(actions) for action in actions}
        return priors, float(solve_exact(game.clone()))


class HeuristicEvaluator:
    def __call__(self, game: Breakthrough) -> tuple[dict[int, float], float]:
        actions = game.legal_actions()
        priors = {action: 1 / len(actions) for action in actions}
        return priors, AlphaBetaAgent.evaluate(game)


class PUCTTests(unittest.TestCase):
    def test_backup_preserves_absolute_sign(self) -> None:
        path = [PUCTNode(1.0), PUCTNode(0.5), PUCTNode(0.2)]
        PUCTPlayer.backup(path, -0.75)
        self.assertEqual([node.value_sum for node in path], [-0.75] * 3)
        self.assertEqual([node.visit_count for node in path], [1] * 3)

    def test_parent_player_controls_absolute_q_exploitation(self) -> None:
        parent = PUCTNode(1.0, visit_count=20)
        high = PUCTNode(0.5, visit_count=5, value_sum=4.0)
        low = PUCTNode(0.5, visit_count=5, value_sum=-4.0)
        parent.children = {1: high, 2: low}
        player = PUCTPlayer(ExactEvaluator(), simulations=1, cpuct=0.0)
        self.assertEqual(player.select_child(parent, PLAYER_1)[0], 1)
        self.assertEqual(player.select_child(parent, PLAYER_2)[0], 2)

    def test_unvisited_child_uses_parent_q_as_first_play_urgency(self) -> None:
        parent = PUCTNode(1.0, visit_count=4, value_sum=2.0)
        child = PUCTNode(0.5)
        self.assertEqual(PUCTPlayer.child_q(parent, child), parent.Q)
        child.visit_count = 2
        child.value_sum = -1.0
        self.assertEqual(PUCTPlayer.child_q(parent, child), child.Q)

    def test_immediate_winning_moves_are_found_for_both_players(self) -> None:
        positions = [
            Breakthrough.from_rows(
                [".....", "....2", ".....", "..1..", "....."],
                player_to_move=PLAYER_1,
            ),
            Breakthrough.from_rows(
                [".....", "..2..", ".....", "1....", "....."],
                player_to_move=PLAYER_2,
            ),
        ]
        for game in positions:
            search = PUCTPlayer(ExactEvaluator(), simulations=40, cpuct=1.0)
            move = game.decode(search.search(game).best_action())
            game.make_move(move)
            self.assertIsNotNone(game.status())

    def test_swapped_positions_reverse_absolute_root_values(self) -> None:
        game = Breakthrough.from_rows(
            [".....", ".1...", "..2..", "...1.", "....2"],
            player_to_move=PLAYER_1,
        )
        swapped = Symmetry(swap_players=True).state(game)
        first = PUCTPlayer(HeuristicEvaluator(), simulations=80, cpuct=1.2).search(game)
        second = PUCTPlayer(HeuristicEvaluator(), simulations=80, cpuct=1.2).search(swapped)
        self.assertAlmostEqual(first.root_value, -second.root_value)

    def test_search_never_mutates_input(self) -> None:
        game = Breakthrough(5, 1)
        before = (game.board.copy(), game.player_to_move, game.winner)
        PUCTPlayer(HeuristicEvaluator(), simulations=8).search(game)
        self.assertEqual((game.board, game.player_to_move, game.winner), before)


if __name__ == "__main__":
    unittest.main()
