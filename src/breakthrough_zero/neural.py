"""The only boundary between mover-relative neural values and absolute values.

Everything outside :class:`NeuralBoundary` uses Player-1-relative values. The
network sees the mover as the bottom player moving toward larger row numbers.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

from .game import Breakthrough, PLAYER_1


def canonical_planes(game: Breakthrough) -> np.ndarray:
    """Return ``(size, size, 2)`` binary planes: my pawns, opponent pawns."""

    board = np.asarray(game.board, dtype=np.int8).reshape(
        game.board_size, game.board_size
    )
    mover = game.player_to_move
    if mover != PLAYER_1:
        board = np.rot90(board, 2)
    mine = board == mover
    theirs = board == -mover
    return np.stack((mine, theirs), axis=-1).astype(np.float32)


def masked_softmax(logits: np.ndarray, legal_mask: np.ndarray) -> np.ndarray:
    """Stable softmax with exactly zero probability on illegal actions."""

    logits = np.asarray(logits, dtype=np.float64)
    legal_mask = np.asarray(legal_mask, dtype=np.bool_)
    if logits.shape != legal_mask.shape:
        raise ValueError("logits and legal_mask must have the same shape")
    if not legal_mask.any():
        return np.zeros_like(logits)
    shifted = logits[legal_mask] - np.max(logits[legal_mask])
    weights = np.exp(shifted)
    probabilities = np.zeros_like(logits)
    probabilities[legal_mask] = weights / weights.sum()
    return probabilities


class RawNetwork(Protocol):
    def predict_raw(self, planes: np.ndarray) -> tuple[np.ndarray, float]: ...


@dataclass(frozen=True)
class AbsolutePrediction:
    priors: dict[int, float]
    value: float


class NeuralBoundary:
    """Small, tested owner of every neural perspective conversion."""

    def __init__(self, network: RawNetwork) -> None:
        self.network = network

    @staticmethod
    def absolute_value(raw_relative_value: float, player_to_move: int) -> float:
        return float(raw_relative_value) * player_to_move

    @staticmethod
    def relative_target(absolute_player1_target: float, player_to_move: int) -> float:
        return float(absolute_player1_target) * player_to_move

    def predict(self, game: Breakthrough) -> AbsolutePrediction:
        logits, raw_value = self.network.predict_raw(canonical_planes(game))
        logits = np.asarray(logits).reshape(-1)
        if logits.shape != (game.action_size,):
            raise ValueError(
                f"network returned {logits.shape}, expected {(game.action_size,)}"
            )
        probabilities = masked_softmax(logits, game.legal_action_mask())
        priors = {action: float(probabilities[action]) for action in game.legal_actions()}
        value = self.absolute_value(raw_value, game.player_to_move)
        return AbsolutePrediction(priors, value)


class GameNetwork:
    """Small native-size Keras CNN with policy and value heads.

    TensorFlow is imported only when this class is constructed, so the rules,
    baseline agents, and MCTS tests stay lightweight.
    """

    def __init__(
        self,
        board_size: int,
        *,
        filters: int = 48,
        residual_blocks: int = 3,
        learning_rate: float = 1e-3,
        l2_strength: float = 1e-4,
        model=None,
    ) -> None:
        self.board_size = board_size
        self.action_size = board_size * board_size * 3
        self.filters = filters
        self.residual_blocks = residual_blocks
        self.learning_rate = learning_rate
        self.l2_strength = l2_strength
        if model is None:
            self.model = self._build_model()
        else:
            self.model = model

    @staticmethod
    def _tf():
        try:
            import tensorflow as tf
        except ImportError as error:
            raise RuntimeError(
                "TensorFlow is required for GameNetwork; install the 'train' extra"
            ) from error
        return tf

    def _build_model(self):
        tf = self._tf()
        keras = tf.keras
        regularizer = keras.regularizers.l2(self.l2_strength)
        inputs = keras.Input(
            (self.board_size, self.board_size, 2), name="relative_board"
        )
        x = keras.layers.Conv2D(
            self.filters,
            3,
            padding="same",
            use_bias=False,
            kernel_regularizer=regularizer,
        )(inputs)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Activation("relu")(x)
        for _ in range(self.residual_blocks):
            residual = x
            x = keras.layers.Conv2D(
                self.filters,
                3,
                padding="same",
                use_bias=False,
                kernel_regularizer=regularizer,
            )(x)
            x = keras.layers.BatchNormalization()(x)
            x = keras.layers.Activation("relu")(x)
            x = keras.layers.Conv2D(
                self.filters,
                3,
                padding="same",
                use_bias=False,
                kernel_regularizer=regularizer,
            )(x)
            x = keras.layers.BatchNormalization()(x)
            x = keras.layers.Add()([x, residual])
            x = keras.layers.Activation("relu")(x)

        policy = keras.layers.Conv2D(3, 1, activation="relu")(x)
        policy = keras.layers.Flatten()(policy)
        policy_logits = keras.layers.Dense(
            self.action_size, name="policy", kernel_regularizer=regularizer
        )(policy)

        value = keras.layers.Conv2D(1, 1, activation="relu")(x)
        value = keras.layers.Flatten()(value)
        value = keras.layers.Dense(
            64, activation="relu", kernel_regularizer=regularizer
        )(value)
        value = keras.layers.Dense(1, activation="tanh", name="value")(value)

        model = keras.Model(inputs, {"policy": policy_logits, "value": value})
        model.compile(
            optimizer=keras.optimizers.Adam(self.learning_rate),
            loss={
                "policy": keras.losses.CategoricalCrossentropy(from_logits=True),
                "value": keras.losses.MeanSquaredError(),
            },
        )
        return model

    def predict_raw(self, planes: np.ndarray) -> tuple[np.ndarray, float]:
        batch = np.asarray(planes, dtype=np.float32)[None, ...]
        prediction = self.model(batch, training=False)
        logits = np.asarray(prediction["policy"])[0]
        value = float(np.asarray(prediction["value"])[0, 0])
        return logits, value

    def fit(self, inputs, policy_targets, value_targets, **kwargs):
        return self.model.fit(
            inputs,
            {"policy": policy_targets, "value": value_targets},
            **kwargs,
        )

    def save(self, path: str | Path) -> None:
        """Save architecture, weights, and optimizer state in one Keras file."""

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.model.save(path)

    def save_weights(self, path: str | Path) -> None:
        self.model.save_weights(path)

    def load_weights(self, path: str | Path) -> None:
        self.model.load_weights(path)

    @classmethod
    def load(cls, path: str | Path) -> "GameNetwork":
        tf = cls._tf()
        model = tf.keras.models.load_model(path)
        board_size = int(model.input_shape[1])
        return cls(board_size, model=model)

