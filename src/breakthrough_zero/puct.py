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

    def evaluate_batch(self, games):
        evaluations = []
        for game in games:
            evaluations.append(self.evaluate(game))
        return evaluations


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
        results = self.search_batch(
            [game],
            [self.simulations],
            [add_root_noise],
        )
        return results[0]

    def evaluate_batch(self, games):
        if hasattr(self.evaluator, "evaluate_batch"):
            return self.evaluator.evaluate_batch(games)
        evaluations = []
        for game in games:
            evaluations.append(self.evaluator.evaluate(game))
        return evaluations

    def search_batch(self, games, simulation_counts, add_root_noise):
        """Search independent roots together and batch their leaf evaluations.

        Tree selection and backup still happen separately for every game. Only
        the neural evaluations are combined, which leaves the PUCT mathematics
        unchanged while giving the GPU a useful batch of positions.
        """

        if len(games) != len(simulation_counts):
            raise ValueError("each game needs a simulation count")
        if len(games) != len(add_root_noise):
            raise ValueError("each game needs a root-noise setting")
        if not games:
            return []
        for index in range(len(games)):
            if games[index].status() is not None:
                raise ValueError("cannot search a terminal position")
            if simulation_counts[index] < 1:
                raise ValueError("simulations must be positive")

        started = time.perf_counter()
        roots = []
        root_evaluations = self.evaluate_batch(games)
        for index in range(len(games)):
            root = PUCTNode(1.0)
            priors, root_value = root_evaluations[index]
            root.q = root_value
            root.expand(priors)
            if add_root_noise[index]:
                self.add_root_noise(root)
            roots.append(root)

        completed = [0] * len(games)
        while True:
            states = []
            nodes = []
            any_search_active = False

            for index in range(len(games)):
                if completed[index] >= simulation_counts[index]:
                    continue
                if self.move_time_s is not None:
                    if time.perf_counter() - started >= self.move_time_s:
                        continue
                any_search_active = True

                # All simulated moves are made on a clone, leaving the real
                # self-play position unchanged.
                state = games[index].clone()
                node = roots[index]
                while node.children and state.status() is None:
                    action, node = self.select_child(node, state.player_to_move)
                    state.make_move(state.decode(action))
                    if node.visits == 0:
                        break

                if state.status() is not None:
                    self.backup(node, float(state.status()))
                else:
                    states.append(state)
                    nodes.append(node)
                completed[index] += 1

            if not any_search_active:
                break

            if states:
                leaf_evaluations = self.evaluate_batch(states)
                for index in range(len(states)):
                    leaf_priors, value = leaf_evaluations[index]
                    node = nodes[index]
                    # Backup first gives this leaf its evaluated Q. Children
                    # expanded afterwards inherit that first-play value.
                    self.backup(node, value)
                    node.expand(leaf_priors)

        results = []
        for root in roots:
            visit_counts = {}
            priors = {}
            q_values = {}
            for action in root.children:
                child = root.children[action]
                visit_counts[action] = child.visits
                priors[action] = child.prior
                q_values[action] = child.q
            results.append(
                {
                    "visit_counts": visit_counts,
                    "priors": priors,
                    "q_values": q_values,
                    "root_visits": root.visits,
                    "root_q": root.q,
                }
            )
        return results

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
