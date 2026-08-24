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

    def __init__(self, tactical=False):
        self.tactical = tactical

    def evaluate(self, game):
        actions = game.legal_actions()
        probability = 1.0 / len(actions)
        priors = {}
        for action in actions:
            priors[action] = probability
        value = rollout_outcome(game, self.tactical)
        return priors, float(value)


class NeuralEvaluator:
    """Adapt the neural boundary to the evaluator interface used by PUCT."""

    def __init__(self, boundary):
        self.boundary = boundary

    def evaluate(self, game):
        prediction = self.boundary.predict(game)
        return prediction["priors"], prediction["value"]


class PUCTNode:
    """Search statistics for one state reached through its parent's action.

    ``value_sum`` and ``q()`` are always absolute Player-1 values. Children are
    indexed by mover-relative policy actions.
    """

    def __init__(self, prior):
        self.prior = float(prior)
        self.visit_count = 0
        self.value_sum = 0.0
        self.children = {}

    def q(self):
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count

    def expand(self, priors):
        if self.children:
            return
        for action in sorted(priors):
            self.children[action] = PUCTNode(priors[action])


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

    def child_q(self, parent, child):
        # First-play urgency gives an unvisited move the parent's current value.
        # This avoids treating every new move as if its value were exactly zero.
        if child.visit_count == 0:
            return parent.q()
        return child.q()

    def select_child(self, parent, parent_player):
        if not parent.children:
            raise ValueError("cannot select from an unexpanded node")

        best_action = None
        best_child = None
        best_score = -math.inf
        parent_scale = math.sqrt(parent.visit_count)

        for action in sorted(parent.children):
            child = parent.children[action]
            q = self.child_q(parent, child)
            exploration = (
                self.cpuct
                * child.prior
                * parent_scale
                / (1 + child.visit_count)
            )
            # Player 1 maximizes Q and Player 2 minimizes it. Multiplying by the
            # parent player lets both cases use the same maximizing selection.
            score = parent_player * q + exploration
            if score > best_score:
                best_score = score
                best_action = action
                best_child = child
        return best_action, best_child

    def backup(self, path, absolute_value):
        # The value has already been converted to Player 1's viewpoint, so its
        # sign stays unchanged at every level of the path.
        for node in path:
            node.visit_count += 1
            node.value_sum += absolute_value

    def add_root_noise(self, root):
        # Dirichlet noise is applied only at the root during self-play. It makes
        # different games investigate different plausible opening moves.
        if self.dirichlet_alpha is None:
            return
        actions = sorted(root.children)
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

        before = (tuple(game.board), game.player_to_move, game.winner)
        started = time.perf_counter()
        if self.move_time_s is None:
            deadline = math.inf
        else:
            deadline = started + self.move_time_s

        root = PUCTNode(1.0)
        priors, unused_value = self.evaluator.evaluate(game.clone())
        root.expand(priors)
        # Keep the evaluator's priors as search evidence. Root noise changes the
        # search trajectory and visit target, but not this recorded prediction.
        original_priors = dict(priors)
        if add_root_noise:
            self.add_root_noise(root)

        completed = 0
        while completed < self.simulations and time.perf_counter() < deadline:
            # Each simulation uses a clone, so search cannot change the position
            # supplied by its caller.
            state = game.clone()
            node = root
            path = [root]

            # Selection follows PUCT until it reaches a terminal state or a
            # previously unvisited child.
            while node.children and state.status() is None:
                action, node = self.select_child(node, state.player_to_move)
                state.make_move(state.decode(action))
                path.append(node)
                if node.visit_count == 0:
                    break

            if state.status() is not None:
                value = float(state.status())
            else:
                # The leaf evaluation supplies both the value to back up and the
                # policy priors used when this leaf is visited again.
                leaf_priors, value = self.evaluator.evaluate(state)
                node.expand(leaf_priors)
            self.backup(path, value)
            completed += 1

        after = (tuple(game.board), game.player_to_move, game.winner)
        if after != before:
            raise AssertionError("search mutated its input state")

        visit_counts = {}
        for action in root.children:
            visit_counts[action] = root.children[action].visit_count
        return {
            "visit_counts": visit_counts,
            "priors": original_priors,
            "root_value": root.q(),
            "simulations": completed,
            "elapsed_s": time.perf_counter() - started,
        }

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
    for action in sorted(counts):
        if counts[action] > best_count:
            best = action
            best_count = counts[action]
    return best
