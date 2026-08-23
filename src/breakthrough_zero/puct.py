"""Readable PUCT search with absolute Player-1 values throughout."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
import random
import time
from typing import Protocol

import numpy as np

from .agents import rollout_outcome
from .game import Breakthrough, Move
from .neural import NeuralBoundary


class Evaluator(Protocol):
    def __call__(self, game: Breakthrough) -> tuple[dict[int, float], float]: ...


class RolloutEvaluator:
    """The assignment's dummy network: uniform priors plus a seeded rollout."""

    def __init__(self, seed: int = 0, tactical: bool = False) -> None:
        self.rng = random.Random(seed)
        self.tactical = tactical

    def __call__(self, game: Breakthrough) -> tuple[dict[int, float], float]:
        actions = game.legal_actions()
        probability = 1.0 / len(actions)
        priors = {action: probability for action in actions}
        value = rollout_outcome(game, self.rng, tactical=self.tactical)
        return priors, float(value)


class NeuralEvaluator:
    def __init__(self, boundary: NeuralBoundary) -> None:
        self.boundary = boundary

    def __call__(self, game: Breakthrough) -> tuple[dict[int, float], float]:
        prediction = self.boundary.predict(game)
        return prediction.priors, prediction.value


@dataclass
class PUCTNode:
    """One edge/node in the search tree.

    ``value_sum`` and ``Q`` are always absolute Player-1 values. They never flip
    sign during backup.
    """

    prior: float
    visit_count: int = 0
    value_sum: float = 0.0
    children: dict[int, "PUCTNode"] = field(default_factory=dict)

    @property
    def Q(self) -> float:
        return self.value_sum / self.visit_count if self.visit_count else 0.0

    @property
    def is_expanded(self) -> bool:
        return bool(self.children)

    def expand(self, priors: dict[int, float]) -> None:
        if self.children:
            return
        self.children = {
            action: PUCTNode(float(prior)) for action, prior in sorted(priors.items())
        }


@dataclass(frozen=True)
class SearchResult:
    visit_counts: dict[int, int]
    priors: dict[int, float]
    root_value: float
    simulations: int
    elapsed_s: float

    def best_action(self) -> int:
        if not self.visit_counts:
            raise ValueError("search result has no actions")
        return max(self.visit_counts, key=lambda action: (self.visit_counts[action], -action))


class PUCTPlayer:
    """PUCT generalized to absolute values by making selection player-aware."""

    def __init__(
        self,
        evaluator: Evaluator,
        *,
        simulations: int = 100,
        cpuct: float = 1.5,
        seed: int = 0,
        move_time_s: float | None = None,
        dirichlet_alpha: float | None = None,
        dirichlet_fraction: float = 0.25,
    ) -> None:
        if simulations < 1:
            raise ValueError("simulations must be positive")
        self.evaluator = evaluator
        self.simulations = simulations
        self.cpuct = cpuct
        self.move_time_s = move_time_s
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_fraction = dirichlet_fraction
        self.rng = np.random.default_rng(seed)

    @staticmethod
    def child_q(parent: PUCTNode, child: PUCTNode) -> float:
        """First-play urgency: an unvisited child starts at the parent's Q."""

        return child.Q if child.visit_count else parent.Q

    def select_child(self, parent: PUCTNode, player_to_move: int) -> tuple[int, PUCTNode]:
        if not parent.children:
            raise ValueError("cannot select from an unexpanded node")
        parent_scale = math.sqrt(max(1, parent.visit_count))

        def score(item: tuple[int, PUCTNode]) -> tuple[float, int]:
            action, child = item
            q = self.child_q(parent, child)
            exploration = (
                self.cpuct * child.prior * parent_scale / (1 + child.visit_count)
            )
            # Player 1 prefers high Q; Player 2 prefers low Q. Exploration is a
            # positive incentive for both, so maximize player*Q + U.
            return player_to_move * q + exploration, -action

        action, child = max(parent.children.items(), key=score)
        return action, child

    @staticmethod
    def backup(path: list[PUCTNode], absolute_value: float) -> None:
        for node in path:
            node.visit_count += 1
            node.value_sum += absolute_value

    def search(self, game: Breakthrough, *, add_root_noise: bool = False) -> SearchResult:
        if game.status() is not None:
            raise ValueError("cannot search a terminal position")
        before = (tuple(game.board), game.player_to_move, game.winner)
        root = PUCTNode(1.0)
        priors, _ = self.evaluator(game.clone())
        root.expand(priors)
        original_priors = dict(priors)
        if add_root_noise:
            self._add_root_noise(root)

        started = time.perf_counter()
        deadline = (
            started + self.move_time_s if self.move_time_s is not None else math.inf
        )
        completed = 0
        while completed < self.simulations and time.perf_counter() < deadline:
            state = game.clone()
            node = root
            path = [root]
            while node.is_expanded and state.status() is None:
                action, node = self.select_child(node, state.player_to_move)
                state.make_move(state.decode(action))
                path.append(node)
                if node.visit_count == 0:
                    break

            if state.status() is not None:
                value = float(state.status())
            else:
                leaf_priors, value = self.evaluator(state)
                node.expand(leaf_priors)
            self.backup(path, value)
            completed += 1

        after = (tuple(game.board), game.player_to_move, game.winner)
        if after != before:
            raise AssertionError("search mutated its input state")
        return SearchResult(
            {action: child.visit_count for action, child in root.children.items()},
            original_priors,
            root.Q,
            completed,
            time.perf_counter() - started,
        )

    def _add_root_noise(self, root: PUCTNode) -> None:
        if self.dirichlet_alpha is None:
            return
        children = list(root.children.values())
        noise = self.rng.dirichlet([self.dirichlet_alpha] * len(children))
        for child, sample in zip(children, noise):
            child.prior = (
                (1 - self.dirichlet_fraction) * child.prior
                + self.dirichlet_fraction * float(sample)
            )

    def choose_move(self, game: Breakthrough, *, add_root_noise: bool = False) -> Move:
        result = self.search(game, add_root_noise=add_root_noise)
        return game.decode(result.best_action())
