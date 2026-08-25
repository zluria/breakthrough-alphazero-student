"""Neural input, policy masking, and value-perspective conversion.

The network always sees the side to move as "me" advancing toward larger row
numbers. Its value is therefore mover-relative. PUCT uses absolute Player-1
values, and ``NeuralBoundary`` is the one place that converts between them.
"""

import os

import numpy as np

from .game import PLAYER_1


def canonical_planes(game):
    """Return two planes: the mover's pawns and the opponent's pawns.

    Player-2 positions are rotated by 180 degrees, so the same pattern has the
    same representation whichever color is to move.
    """

    board = np.array(game.board, dtype=np.int8)
    board = board.reshape(game.board_size, game.board_size)
    mover = game.player_to_move
    if mover != PLAYER_1:
        board = np.rot90(board, 2)
    mine = board == mover
    theirs = board == -mover
    return np.stack([mine, theirs], axis=-1).astype(np.float32)


def masked_softmax(logits, legal_mask):
    """Softmax over legal actions, with exactly zero on illegal actions."""

    logits = np.array(logits, dtype=np.float64)
    legal_mask = np.array(legal_mask, dtype=np.bool_)
    if logits.shape != legal_mask.shape:
        raise ValueError("logits and legal_mask must have the same shape")
    probabilities = np.zeros_like(logits)
    if not legal_mask.any():
        return probabilities

    legal_logits = logits[legal_mask]
    # Subtracting the largest legal logit leaves softmax unchanged and prevents
    # overflow when exponentiating.
    legal_logits = legal_logits - np.max(legal_logits)
    weights = np.exp(legal_logits)
    probabilities[legal_mask] = weights / weights.sum()
    return probabilities


class NeuralBoundary:
    """Convert between mover-relative network values and absolute game values."""

    def __init__(self, network=None):
        self.network = network

    def absolute_value(self, relative_value, player_to_move):
        # relative = absolute * player, and player is either 1 or -1.
        return float(relative_value) * player_to_move

    def relative_target(self, absolute_value, player_to_move):
        return float(absolute_value) * player_to_move

    def evaluate(self, game):
        return self.evaluate_batch([game])[0]

    def evaluate_batch(self, games):
        """Evaluate several independent leaves in one neural-network call."""

        planes = []
        for game in games:
            planes.append(canonical_planes(game))
        predictions = self.network.predict_batch(planes)

        evaluations = []
        for index in range(len(games)):
            game = games[index]
            logits, relative_value = predictions[index]
            logits = np.array(logits).reshape(-1)
            if logits.shape != (game.action_size,):
                raise ValueError("network returned the wrong policy shape")

            actions = game.legal_actions()
            legal_mask = np.zeros(game.action_size, dtype=np.bool_)
            for action in actions:
                legal_mask[action] = True
            probabilities = masked_softmax(logits, legal_mask)
            priors = {}
            for action in actions:
                priors[action] = float(probabilities[action])
            value = self.absolute_value(relative_value, game.player_to_move)
            evaluations.append((priors, value))
        return evaluations


class GameNetwork:
    """A residual Keras network with separate policy and value heads.

    Each board size has its own native input and policy size. The policy emits
    logits for three forward directions from every square; legality is enforced
    later by ``masked_softmax``.
    """

    def __init__(
        self,
        board_size,
        filters=48,
        residual_blocks=3,
        learning_rate=0.001,
        model=None,
    ):
        self.board_size = board_size
        self.action_size = board_size * board_size * 3
        self.filters = filters
        self.residual_blocks = residual_blocks
        self.learning_rate = learning_rate
        if model is None:
            self.model = self.build_model()
        else:
            self.model = model

    def build_model(self):
        import keras

        inputs = keras.Input(
            shape=(self.board_size, self.board_size, 2),
            name="relative_board",
        )
        x = keras.layers.Conv2D(
            self.filters,
            3,
            padding="same",
        )(inputs)
        x = keras.layers.Activation("relu")(x)

        for unused_block in range(self.residual_blocks):
            # The skip connection lets a block learn a correction to its input,
            # rather than having to relearn the entire representation.
            old_x = x
            x = keras.layers.Conv2D(
                self.filters,
                3,
                padding="same",
            )(x)
            x = keras.layers.Activation("relu")(x)
            x = keras.layers.Conv2D(
                self.filters,
                3,
                padding="same",
            )(x)
            x = keras.layers.Add()([x, old_x])
            x = keras.layers.Activation("relu")(x)

        # There are exactly three actions per square, so these three channels are
        # already the complete vector of unnormalized policy logits.
        policy = keras.layers.Conv2D(3, 1)(x)
        policy = keras.layers.Flatten(name="policy")(policy)

        # The value head predicts the final result from the mover's viewpoint;
        # tanh keeps the prediction in the same [-1, 1] range as game outcomes.
        value = keras.layers.Conv2D(1, 1, activation="relu")(x)
        value = keras.layers.Flatten()(value)
        value = keras.layers.Dense(
            64,
            activation="relu",
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
        return self.predict_batch([planes])[0]

    def predict_batch(self, planes):
        """Predict several board positions together so the GPU stays busy."""

        batch = np.array(planes, dtype=np.float32)
        prediction = self.model(batch, training=False)
        policy_batch = np.array(prediction["policy"])
        value_batch = np.array(prediction["value"])
        results = []
        for index in range(len(batch)):
            logits = policy_batch[index]
            value = float(value_batch[index, 0])
            results.append((logits, value))
        return results

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
        """Save the network together with the Adam optimizer state."""

        directory = os.path.dirname(str(path))
        if directory:
            os.makedirs(directory, exist_ok=True)
        self.model.save(path)

    def save_weights(self, path):
        self.model.save_weights(path)

    def load_weights(self, path):
        self.model.load_weights(path)


def load_network(path):
    import keras

    model = keras.models.load_model(path)
    board_size = int(model.input_shape[1])
    return GameNetwork(board_size, model=model)
