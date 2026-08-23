import unittest

from breakthrough_zero.agents import evaluate_position, solve_exact
from breakthrough_zero.game import PLAYER_1, PLAYER_2, Breakthrough, game_from_rows
from breakthrough_zero.puct import PUCTNode, PUCTPlayer, best_action
from breakthrough_zero.symmetry import transform_state


class ExactEvaluator:
    def evaluate(self, game):
        actions = game.legal_actions()
        priors = {}
        for action in actions:
            priors[action] = 1 / len(actions)
        return priors, float(solve_exact(game.clone()))


class HeuristicEvaluator:
    def evaluate(self, game):
        actions = game.legal_actions()
        priors = {}
        for action in actions:
            priors[action] = 1 / len(actions)
        return priors, evaluate_position(game)


class PUCTTests(unittest.TestCase):
    def test_backup_preserves_absolute_sign(self):
        path = [PUCTNode(1.0), PUCTNode(0.5), PUCTNode(0.2)]
        player = PUCTPlayer(ExactEvaluator())
        player.backup(path, -0.75)
        for node in path:
            self.assertEqual(node.value_sum, -0.75)
            self.assertEqual(node.visit_count, 1)

    def test_parent_player_controls_absolute_q_exploitation(self):
        parent = PUCTNode(1.0)
        parent.visit_count = 20
        high = PUCTNode(0.5)
        high.visit_count = 5
        high.value_sum = 4.0
        low = PUCTNode(0.5)
        low.visit_count = 5
        low.value_sum = -4.0
        parent.children = {1: high, 2: low}

        player = PUCTPlayer(ExactEvaluator(), 1, 0.0)
        self.assertEqual(player.select_child(parent, PLAYER_1)[0], 1)
        self.assertEqual(player.select_child(parent, PLAYER_2)[0], 2)

    def test_unvisited_child_uses_parent_q_as_first_play_urgency(self):
        parent = PUCTNode(1.0)
        parent.visit_count = 4
        parent.value_sum = 2.0
        child = PUCTNode(0.5)
        player = PUCTPlayer(ExactEvaluator())
        self.assertEqual(player.child_q(parent, child), parent.q())
        child.visit_count = 2
        child.value_sum = -1.0
        self.assertEqual(player.child_q(parent, child), child.q())

    def test_immediate_winning_moves_are_found_for_both_players(self):
        positions = [
            game_from_rows(
                [".....", "....2", ".....", "..1..", "....."],
                PLAYER_1,
            ),
            game_from_rows(
                [".....", "..2..", ".....", "1....", "....."],
                PLAYER_2,
            ),
        ]
        for game in positions:
            player = PUCTPlayer(ExactEvaluator(), 40, 1.0)
            result = player.search(game)
            game.make_move(game.decode(best_action(result)))
            self.assertIsNotNone(game.status())

    def test_swapped_positions_reverse_absolute_root_values(self):
        game = game_from_rows(
            [".....", ".1...", "..2..", "...1.", "....2"],
            PLAYER_1,
        )
        swapped = transform_state(game, (True, False))
        first = PUCTPlayer(HeuristicEvaluator(), 80, 1.2).search(game)
        second = PUCTPlayer(HeuristicEvaluator(), 80, 1.2).search(swapped)
        self.assertAlmostEqual(first["root_value"], -second["root_value"])

    def test_search_never_mutates_input(self):
        game = Breakthrough(5, 1)
        before = (list(game.board), game.player_to_move, game.winner)
        PUCTPlayer(HeuristicEvaluator(), 8).search(game)
        after = (game.board, game.player_to_move, game.winner)
        self.assertEqual(after, before)


if __name__ == "__main__":
    unittest.main()
