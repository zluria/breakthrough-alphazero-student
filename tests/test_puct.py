import unittest

from breakthrough_zero.agents import evaluate_position, solve_exact
from breakthrough_zero.game import PLAYER_1, PLAYER_2, Breakthrough, game_from_rows
from breakthrough_zero.puct import PUCTNode, PUCTPlayer, best_action


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
        root = PUCTNode(1.0)
        middle = PUCTNode(0.5, root)
        leaf = PUCTNode(0.2, middle)
        player = PUCTPlayer(ExactEvaluator())
        player.backup(leaf, -0.75)
        for node in (root, middle, leaf):
            self.assertEqual(node.q, -0.75)
            self.assertEqual(node.visits, 1)

    def test_parent_player_controls_absolute_q_exploitation(self):
        parent = PUCTNode(1.0)
        parent.visits = 20
        high = PUCTNode(0.5, parent)
        high.visits = 5
        high.q = 0.8
        low = PUCTNode(0.5, parent)
        low.visits = 5
        low.q = -0.8
        parent.children = {1: high, 2: low}

        player = PUCTPlayer(ExactEvaluator(), 1, 0.0)
        self.assertEqual(player.select_child(parent, PLAYER_1)[0], 1)
        self.assertEqual(player.select_child(parent, PLAYER_2)[0], 2)

    def test_child_has_parent_pointer_and_inherits_parent_q(self):
        parent = PUCTNode(1.0)
        parent.q = 0.5
        child = PUCTNode(0.25, parent)
        self.assertIs(child.parent, parent)
        self.assertEqual(child.q, 0.5)
        parent.q = -0.25
        self.assertEqual(child.q, 0.5)

        PUCTPlayer(ExactEvaluator()).backup(child, -1.0)
        self.assertEqual(child.q, -1.0)

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

    def test_search_never_mutates_input(self):
        game = Breakthrough(5, 1)
        before = (list(game.board), game.player_to_move, game.winner)
        PUCTPlayer(HeuristicEvaluator(), 8).search(game)
        after = (game.board, game.player_to_move, game.winner)
        self.assertEqual(after, before)


if __name__ == "__main__":
    unittest.main()
