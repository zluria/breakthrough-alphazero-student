"""A bounded list of old positions and simple symmetry augmentation."""

import random

import numpy as np

from .data import state_from_record
from .neural import NeuralBoundary, canonical_planes
from .symmetry import SYMMETRIES, transform_action, transform_state


class ReplayBuffer:
    def __init__(self, capacity, seed=0):
        if capacity < 1:
            raise ValueError("capacity must be positive")
        self.capacity = capacity
        self.entries = []
        self.random = random.Random(seed)
        self.total_added = 0
        self.total_sampled = 0

    def add(self, records, iteration):
        for record in records:
            self.entries.append((record, iteration))
            self.total_added += 1
        if len(self.entries) > self.capacity:
            extra = len(self.entries) - self.capacity
            self.entries = self.entries[extra:]
        return len(records)

    def sample(self, count):
        if not self.entries:
            raise ValueError("cannot sample an empty replay buffer")
        records = []
        for unused_number in range(count):
            record, unused_iteration = self.random.choice(self.entries)
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
            if symmetry[0]:
                absolute_outcome = -absolute_outcome
            relative_value = boundary.relative_target(
                absolute_outcome, new_game.player_to_move
            )

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
