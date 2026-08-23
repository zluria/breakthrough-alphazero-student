"""Mover-relative neural input and the one value-conversion boundary."""

import os

import numpy as np

from .game import PLAYER_1


def canonical_planes(game):
    """Return two planes: the mover's pawns and the opponent's pawns."""

    board = np.array(game.board, dtype=np.int8)
    board = board.reshape(game.board_size, game.board_size)
    mover = game.player_to_move
    if mover != PLAYER_1:
        board = np.rot90(board, 2)
    mine = board == mover
    theirs = board == -mover
    return np.stack([mine, theirs], axis=-1).astype(np.float32)


def masked_softmax(logits, legal_mask):
    """Softmax over legal actions, with zero on every illegal action."""

    logits = np.array(logits, dtype=np.float64)
    legal_mask = np.array(legal_mask, dtype=np.bool_)
    if logits.shape != legal_mask.shape:
        raise ValueError("logits and legal_mask must have the same shape")
    probabilities = np.zeros_like(logits)
    if not legal_mask.any():
        return probabilities

    legal_logits = logits[legal_mask]
    legal_logits = legal_logits - np.max(legal_logits)
    weights = np.exp(legal_logits)
    probabilities[legal_mask] = weights / weights.sum()
    return probabilities


class NeuralBoundary:
    """The only place that converts relative neural values."""

    def __init__(self, network=None):
        self.network = network

    def absolute_value(self, relative_value, player_to_move):
        return float(relative_value) * player_to_move

    def relative_target(self, absolute_value, player_to_move):
        return float(absolute_value) * player_to_move

    def predict(self, game):
        logits, relative_value = self.network.predict_raw(canonical_planes(game))
        logits = np.array(logits).reshape(-1)
        if logits.shape != (game.action_size,):
            raise ValueError("network returned the wrong policy shape")

        probabilities = masked_softmax(logits, game.legal_action_mask())
        priors = {}
        for action in game.legal_actions():
            priors[action] = float(probabilities[action])
        value = self.absolute_value(relative_value, game.player_to_move)
        return {"priors": priors, "value": value}


def get_tensorflow():
    try:
        import tensorflow as tf
    except ImportError:
        raise RuntimeError("TensorFlow is required for neural training")
    return tf


class GameNetwork:
    """A small native-size Keras policy and value network."""

    def __init__(
        self,
        board_size,
        filters=48,
        residual_blocks=3,
        learning_rate=0.001,
        l2_strength=0.0001,
        model=None,
    ):
        self.board_size = board_size
        self.action_size = board_size * board_size * 3
        self.filters = filters
        self.residual_blocks = residual_blocks
        self.learning_rate = learning_rate
        self.l2_strength = l2_strength
        if model is None:
            self.model = self.build_model()
        else:
            self.model = model

    def build_model(self):
        tf = get_tensorflow()
        keras = tf.keras
        regularizer = keras.regularizers.l2(self.l2_strength)

        inputs = keras.Input(
            shape=(self.board_size, self.board_size, 2),
            name="relative_board",
        )
        x = keras.layers.Conv2D(
            self.filters,
            3,
            padding="same",
            use_bias=False,
            kernel_regularizer=regularizer,
        )(inputs)
        x = keras.layers.Activation("relu")(x)

        for unused_block in range(self.residual_blocks):
            old_x = x
            x = keras.layers.Conv2D(
                self.filters,
                3,
                padding="same",
                use_bias=False,
                kernel_regularizer=regularizer,
            )(x)
            x = keras.layers.Activation("relu")(x)
            x = keras.layers.Conv2D(
                self.filters,
                3,
                padding="same",
                use_bias=False,
                kernel_regularizer=regularizer,
            )(x)
            x = keras.layers.Add()([x, old_x])
            x = keras.layers.Activation("relu")(x)

        policy = keras.layers.Conv2D(3, 1, activation="relu")(x)
        policy = keras.layers.Flatten()(policy)
        policy = keras.layers.Dense(
            self.action_size,
            name="policy",
            kernel_regularizer=regularizer,
        )(policy)

        value = keras.layers.Conv2D(1, 1, activation="relu")(x)
        value = keras.layers.Flatten()(value)
        value = keras.layers.Dense(
            64,
            activation="relu",
            kernel_regularizer=regularizer,
        )(value)
        value = keras.layers.Dense(1, activation="tanh", name="value")(value)

        model = keras.Model(inputs, {"policy": policy, "value": value})
        model.compile(
            optimizer=keras.optimizers.Adam(self.learning_rate),
            loss={
                "policy": keras.losses.CategoricalCrossentropy(from_logits=True),
                "value": keras.losses.MeanSquaredError(),
            },
        )
        return model

    def predict_raw(self, planes):
        batch = np.array(planes, dtype=np.float32)[None, ...]
        prediction = self.model(batch, training=False)
        logits = np.array(prediction["policy"])[0]
        value = float(np.array(prediction["value"])[0, 0])
        return logits, value

    def fit(
        self,
        inputs,
        policy_targets,
        value_targets,
        validation_data=None,
        epochs=1,
        batch_size=32,
        callbacks=None,
        verbose=0,
    ):
        targets = {"policy": policy_targets, "value": value_targets}
        return self.model.fit(
            inputs,
            targets,
            validation_data=validation_data,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=verbose,
        )

    def save(self, path):
        directory = os.path.dirname(str(path))
        if directory:
            os.makedirs(directory, exist_ok=True)
        self.model.save(path)

    def save_weights(self, path):
        self.model.save_weights(path)

    def load_weights(self, path):
        self.model.load_weights(path)


def load_network(path):
    tf = get_tensorflow()
    model = tf.keras.models.load_model(path)
    board_size = int(model.input_shape[1])
    return GameNetwork(board_size, model=model)
