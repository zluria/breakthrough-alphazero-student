"""Bounded replay storage and duplicate-safe symmetry augmentation."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import hashlib
import random
from typing import Iterable

import numpy as np

from .data import PositionRecord
from .neural import NeuralBoundary, canonical_planes
from .symmetry import SYMMETRIES


@dataclass(frozen=True)
class ReplayEntry:
    record: PositionRecord
    added_iteration: int


class ReplayBuffer:
    def __init__(self, capacity: int, seed: int = 0) -> None:
        if capacity < 1:
            raise ValueError("capacity must be positive")
        self.capacity = capacity
        self.entries: deque[ReplayEntry] = deque(maxlen=capacity)
        self.rng = random.Random(seed)
        self.total_added = 0
        self.total_sampled = 0

    def add(self, records: Iterable[PositionRecord], iteration: int) -> int:
        count = 0
        for record in records:
            self.entries.append(ReplayEntry(record, iteration))
            self.total_added += 1
            count += 1
        return count

    def sample(self, count: int) -> list[PositionRecord]:
        if not self.entries:
            raise ValueError("cannot sample an empty replay buffer")
        selected = self.rng.choices(list(self.entries), k=count)
        self.total_sampled += count
        return [entry.record for entry in selected]

    def metrics(self, current_iteration: int) -> dict[str, float | int]:
        ages = [current_iteration - entry.added_iteration for entry in self.entries]
        return {
            "size": len(self.entries),
            "capacity": self.capacity,
            "fill_fraction": len(self.entries) / self.capacity,
            "mean_age_iterations": sum(ages) / len(ages) if ages else 0.0,
            "oldest_age_iterations": max(ages, default=0),
            "total_added": self.total_added,
            "total_sampled": self.total_sampled,
            "replay_consumption_ratio": self.total_sampled / max(1, self.total_added),
        }


def records_to_training_arrays(
    records: Iterable[PositionRecord],
    *,
    augment: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, int]]:
    """Reconstruct mover-relative tensors while deduplicating exact symmetries."""

    inputs: list[np.ndarray] = []
    policies: list[np.ndarray] = []
    values: list[float] = []
    duplicates = 0

    for record in records:
        if record.final_outcome not in (-1, 1):
            raise ValueError("training records require a final absolute outcome")
        game = record.state()
        source_counts = dict(zip(record.legal_actions, record.visit_counts))
        symmetries = SYMMETRIES if augment else SYMMETRIES[:1]
        seen: set[bytes] = set()
        for symmetry in symmetries:
            transformed = symmetry.state(game)
            policy = np.zeros(transformed.action_size, dtype=np.float32)
            for action, count in source_counts.items():
                transformed_action = symmetry.action(game, action)
                policy[transformed_action] += count
            if policy.sum() <= 0:
                raise ValueError("visit counts contain no target mass")
            policy /= policy.sum()
            planes = canonical_planes(transformed)
            absolute_outcome = (
                -record.final_outcome if symmetry.swap_players else record.final_outcome
            )
            relative_value = NeuralBoundary.relative_target(
                absolute_outcome, transformed.player_to_move
            )
            digest = hashlib.sha256(
                planes.tobytes() + policy.tobytes() + np.float32(relative_value).tobytes()
            ).digest()
            if digest in seen:
                duplicates += 1
                continue
            seen.add(digest)
            inputs.append(planes)
            policies.append(policy)
            values.append(relative_value)

    if not inputs:
        raise ValueError("no training examples were produced")
    return (
        np.stack(inputs),
        np.stack(policies),
        np.asarray(values, dtype=np.float32)[:, None],
        {"examples": len(inputs), "symmetry_duplicates_removed": duplicates},
    )
