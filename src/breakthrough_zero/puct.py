"""PUCT search with values measured from Player 1's point of view.

The evaluator and every node store the same absolute value: positive favors
Player 1 and negative favors Player 2. Selection accounts for whose turn it is;
backup therefore does not alternate the sign.
"""

import math
import time

import numpy as np

from .agents import rollout_outcome


class RolloutEvaluator:
    """The non-neural evaluator: uniform priors and one random rollout."""

    def evaluate(self, game):
        actions = game.legal_actions()
        probability = 1.0 / len(actions)
        priors = {}
        for action in actions:
            priors[action] = probability
        value = rollout_outcome(game)
        return priors, float(value)


class PUCTNode:
    """Search statistics for one state reached through its parent's action.

    ``q`` is always an absolute Player-1 value. A new child inherits its
    parent's current estimate as its first-play value.
    """

    def __init__(self, prior, parent=None):
        self.prior = float(prior)
        self.parent = parent
        self.visits = 0
        if parent is None:
            self.q = 0.0
        else:
            self.q = parent.q
        self.children = {}

    def expand(self, priors):
        if self.children:
            return
        for action in priors:
            self.children[action] = PUCTNode(priors[action], self)


class PUCTPlayer:
    def __init__(
        self,
        evaluator,
        simulations=100,
        cpuct=1.5,
        move_time_s=None,
        dirichlet_alpha=None,
        dirichlet_fraction=0.25,
    ):
        if simulations < 1:
            raise ValueError("simulations must be positive")
        self.evaluator = evaluator
        self.simulations = simulations
        self.cpuct = cpuct
        self.move_time_s = move_time_s
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_fraction = dirichlet_fraction

    def select_child(self, parent, parent_player):
        if not parent.children:
            raise ValueError("cannot select from an unexpanded node")

        best_action = None
        best_child = None
        best_score = -math.inf
        parent_scale = math.sqrt(parent.visits)

        for action in parent.children:
            child = parent.children[action]
            exploration = (
                self.cpuct
                * child.prior
                * parent_scale
                / (1 + child.visits)
            )
            # Player 1 maximizes Q and Player 2 minimizes it. Multiplying by the
            # parent player lets both cases use the same maximizing selection.
            score = parent_player * child.q + exploration
            if score > best_score:
                best_score = score
                best_action = action
                best_child = child
        return best_action, best_child

    def backup(self, node, absolute_value):
        # The value has already been converted to Player 1's viewpoint, so its
        # sign stays unchanged at every level of the path.
        while node is not None:
            node.visits += 1
            node.q += (absolute_value - node.q) / node.visits
            node = node.parent

    def add_root_noise(self, root):
        # Dirichlet noise is applied only at the root during self-play. It makes
        # different games investigate different plausible opening moves.
        if self.dirichlet_alpha is None:
            return
        actions = list(root.children)
        settings = [self.dirichlet_alpha] * len(actions)
        noise = np.random.dirichlet(settings)
        for index in range(len(actions)):
            child = root.children[actions[index]]
            child.prior = (
                (1 - self.dirichlet_fraction) * child.prior
                + self.dirichlet_fraction * float(noise[index])
            )

    def search(self, game, add_root_noise=False):
        if game.status() is not None:
            raise ValueError("cannot search a terminal position")

        if self.move_time_s is None:
            deadline = math.inf
        else:
            deadline = time.perf_counter() + self.move_time_s

        root = PUCTNode(1.0)
        priors, root_value = self.evaluator.evaluate(game)
        root.q = root_value
        root.expand(priors)
        if add_root_noise:
            self.add_root_noise(root)

        completed = 0
        while completed < self.simulations and time.perf_counter() < deadline:
            # All simulated moves are made on a clone, leaving the caller's
            # position unchanged.
            state = game.clone()
            node = root

            # Selection follows PUCT until it reaches a terminal state or a
            # previously unvisited child.
            while node.children and state.status() is None:
                action, node = self.select_child(node, state.player_to_move)
                state.make_move(state.decode(action))
                if node.visits == 0:
                    break

            if state.status() is not None:
                value = float(state.status())
                leaf_priors = None
            else:
                # The leaf evaluation supplies both the value to back up and the
                # policy priors used when this leaf is visited again.
                leaf_priors, value = self.evaluator.evaluate(state)
            # Backing up first makes the leaf's first Q exactly its evaluation;
            # newly expanded children then inherit that value.
            self.backup(node, value)
            if leaf_priors is not None:
                node.expand(leaf_priors)
            completed += 1

        visit_counts = {}
        for action in root.children:
            visit_counts[action] = root.children[action].visits
        return {"visit_counts": visit_counts}

    def choose_move(self, game, add_root_noise=False):
        result = self.search(game, add_root_noise)
        action = best_action(result)
        return game.decode(action)


def best_action(search_result):
    counts = search_result["visit_counts"]
    if not counts:
        raise ValueError("search result has no actions")
    best = None
    best_count = -1
    for action in counts:
        if counts[action] > best_count:
            best = action
            best_count = counts[action]
    return best
