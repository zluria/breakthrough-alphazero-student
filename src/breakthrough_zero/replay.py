"""Replay storage and symmetry augmentation for self-play positions."""

import random

import numpy as np

from .data import state_from_record
from .neural import NeuralBoundary, canonical_planes
from .symmetry import transform_action, transform_state


class ReplayBuffer:
    def __init__(self, capacity):
        if capacity < 1:
            raise ValueError("capacity must be positive")
        self.capacity = capacity
        self.data = []

    def add(self, records):
        self.data.extend(records)
        if len(self.data) > self.capacity:
            self.data = self.data[-self.capacity :]

    def sample(self, count):
        return random.sample(self.data, count)


def records_to_training_arrays(records):
    """Build neural inputs and targets from recorded search positions.

    Each record contributes its original position and its left-right reflection.
    Player swapping would duplicate the same mover-relative input and targets.
    Policy targets are normalized MCTS visit counts, and absolute outcomes are
    converted to the value head's mover-relative convention.
    """

    inputs = []
    policies = []
    values = []
    boundary = NeuralBoundary()

    for record in records:
        if record["final_outcome"] not in (-1, 1):
            raise ValueError("training records require a final outcome")
        game = state_from_record(record)
        source_counts = {}
        for index in range(len(record["legal_actions"])):
            action = record["legal_actions"][index]
            source_counts[action] = record["visit_counts"][index]

        for symmetry in [(False, False), (False, True)]:
            new_game = transform_state(game, symmetry)
            policy = np.zeros(new_game.action_size, dtype=np.float32)
            for action in source_counts:
                new_action = transform_action(game, action, symmetry)
                policy[new_action] += source_counts[action]
            if policy.sum() <= 0:
                raise ValueError("visit counts contain no target mass")
            policy = policy / policy.sum()

            planes = canonical_planes(new_game)
            absolute_outcome = record["final_outcome"]
            relative_value = boundary.relative_target(
                absolute_outcome, new_game.player_to_move
            )
            inputs.append(planes)
            policies.append(policy)
            values.append(relative_value)

    if not inputs:
        raise ValueError("no training examples were produced")
    value_array = np.array(values, dtype=np.float32).reshape(-1, 1)
    return np.stack(inputs), np.stack(policies), value_array
