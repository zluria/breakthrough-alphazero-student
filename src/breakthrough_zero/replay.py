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


def records_to_training_arrays(records, random_reflection=False):
    """Build neural inputs and targets from recorded search positions.

    Pretraining can materialize both left-right versions of every record. During
    the AlphaZero loop, ``random_reflection`` chooses just one version whenever
    a record is sampled. That gives continuing augmentation without presenting
    the same position twice in one batch. Player swapping would duplicate the
    same mover-relative input and targets.
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

        if random_reflection:
            reflect = bool(np.random.randint(2))
            symmetries = [(False, reflect)]
        else:
            symmetries = [(False, False), (False, True)]

        for symmetry in symmetries:
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
