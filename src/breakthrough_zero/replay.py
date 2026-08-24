"""Replay storage and symmetry augmentation for self-play positions."""

import random

import numpy as np

from .data import state_from_record
from .neural import NeuralBoundary, canonical_planes
from .symmetry import SYMMETRIES, transform_action, transform_state


class ReplayBuffer:
    def __init__(self, capacity):
        if capacity < 1:
            raise ValueError("capacity must be positive")
        self.capacity = capacity
        self.entries = []
        self.total_added = 0
        self.total_sampled = 0

    def add(self, records, iteration):
        for record in records:
            self.entries.append((record, iteration))
            self.total_added += 1
        if len(self.entries) > self.capacity:
            # Keep the most recent experience when the bounded window is full.
            extra = len(self.entries) - self.capacity
            self.entries = self.entries[extra:]
        return len(records)

    def sample(self, count):
        if not self.entries:
            raise ValueError("cannot sample an empty replay buffer")
        records = []
        # Sampling with replacement permits a fixed number of training examples
        # even during the first iterations, while the replay window is small.
        for unused_number in range(count):
            record, unused_iteration = random.choice(self.entries)
            records.append(record)
        self.total_sampled += count
        return records

    def metrics(self, current_iteration):
        ages = []
        for record, added_iteration in self.entries:
            ages.append(current_iteration - added_iteration)
        if ages:
            mean_age = sum(ages) / len(ages)
            oldest_age = max(ages)
        else:
            mean_age = 0.0
            oldest_age = 0
        return {
            "size": len(self.entries),
            "capacity": self.capacity,
            "fill_fraction": len(self.entries) / self.capacity,
            "mean_age_iterations": mean_age,
            "oldest_age_iterations": oldest_age,
            "total_added": self.total_added,
            "total_sampled": self.total_sampled,
            "replay_consumption_ratio": self.total_sampled
            / max(1, self.total_added),
        }


def records_to_training_arrays(records, augment=True):
    """Build neural inputs and targets from recorded search positions.

    Policy targets are normalized MCTS visit counts. Game outcomes are stored in
    the absolute convention and converted here to the mover-relative convention
    used by the value head.
    """

    inputs = []
    policies = []
    values = []
    duplicates = 0
    boundary = NeuralBoundary()

    for record in records:
        if record["final_outcome"] not in (-1, 1):
            raise ValueError("training records require a final outcome")
        game = state_from_record(record)
        source_counts = {}
        for index in range(len(record["legal_actions"])):
            action = record["legal_actions"][index]
            source_counts[action] = record["visit_counts"][index]

        if augment:
            symmetries = SYMMETRIES
        else:
            symmetries = [SYMMETRIES[0]]
        seen = set()

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
            # Swapping player identities reverses an absolute Player-1 outcome.
            # Reflection alone leaves it unchanged.
            if symmetry[0]:
                absolute_outcome = -absolute_outcome
            relative_value = boundary.relative_target(
                absolute_outcome, new_game.player_to_move
            )

            # Symmetric-looking positions can collapse to identical canonical
            # examples. Count each exact input/target triple only once.
            key = (planes.tobytes(), policy.tobytes(), float(relative_value))
            if key in seen:
                duplicates += 1
                continue
            seen.add(key)
            inputs.append(planes)
            policies.append(policy)
            values.append(relative_value)

    if not inputs:
        raise ValueError("no training examples were produced")
    value_array = np.array(values, dtype=np.float32).reshape(-1, 1)
    metrics = {
        "examples": len(inputs),
        "symmetry_duplicates_removed": duplicates,
    }
    return np.stack(inputs), np.stack(policies), value_array, metrics
