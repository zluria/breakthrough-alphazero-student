import os
import tempfile
import unittest

import numpy as np

from breakthrough_zero.game import PLAYER_1, PLAYER_2, game_from_rows
from breakthrough_zero.neural import (
    GameNetwork,
    NeuralBoundary,
    canonical_planes,
    load_network,
    masked_softmax,
)
from breakthrough_zero.symmetry import (
    transform_action,
    transform_move,
    transform_state,
)


try:
    import keras

    KERAS_INSTALLED = True
except ImportError:
    KERAS_INSTALLED = False


class RecordingNetwork:
    def __init__(self, action_size, value=0.4):
        self.action_size = action_size
        self.value = value
        self.inputs = []

    def predict_raw(self, planes):
        self.inputs.append(planes.copy())
        return np.linspace(-1, 1, self.action_size), self.value

    def predict_batch(self, planes):
        predictions = []
        for position in planes:
            predictions.append(self.predict_raw(position))
        return predictions


class NeuralBoundaryTests(unittest.TestCase):
    def setUp(self):
        self.game = game_from_rows(
            ["1....", ".1.2.", "..1..", ".2...", "....2"],
            PLAYER_1,
        )

    def test_swapped_and_rotated_positions_have_identical_inputs(self):
        swapped = transform_state(self.game, (True, False))
        first = canonical_planes(self.game)
        second = canonical_planes(swapped)
        np.testing.assert_array_equal(first, second)

    def test_identical_canonical_inputs_have_identical_raw_predictions(self):
        swapped = transform_state(self.game, (True, False))
        network = RecordingNetwork(self.game.action_size)
        first = network.predict_raw(canonical_planes(self.game))
        second = network.predict_raw(canonical_planes(swapped))
        np.testing.assert_array_equal(first[0], second[0])
        self.assertEqual(first[1], second[1])
        np.testing.assert_array_equal(network.inputs[0], network.inputs[1])

    def test_wrapper_opposes_absolute_values_for_swapped_players(self):
        network = RecordingNetwork(self.game.action_size, 0.35)
        boundary = NeuralBoundary(network)
        swapped = transform_state(self.game, (True, False))
        unused_priors, first_value = boundary.evaluate(self.game)
        unused_priors, second_value = boundary.evaluate(swapped)
        self.assertAlmostEqual(first_value, 0.35)
        self.assertAlmostEqual(second_value, -0.35)

    def test_boundary_batches_positions_in_one_network_call(self):
        swapped = transform_state(self.game, (True, False))
        network = RecordingNetwork(self.game.action_size, 0.35)
        evaluations = NeuralBoundary(network).evaluate_batch(
            [self.game, swapped]
        )
        self.assertEqual(len(network.inputs), 2)
        self.assertAlmostEqual(evaluations[0][1], 0.35)
        self.assertAlmostEqual(evaluations[1][1], -0.35)

    def test_all_absolute_targets_convert_relative(self):
        boundary = NeuralBoundary()
        for target in (-1.0, -0.5, 0.0, 0.25, 1.0):
            self.assertEqual(boundary.relative_target(target, PLAYER_1), target)
            self.assertEqual(boundary.relative_target(target, PLAYER_2), -target)

    def test_illegal_actions_are_masked_exactly(self):
        logits = np.arange(self.game.action_size, dtype=float)
        mask = self.game.legal_action_mask()
        probabilities = masked_softmax(logits, mask)
        self.assertAlmostEqual(float(probabilities.sum()), 1.0)
        self.assertTrue(np.all(probabilities[~mask] == 0))
        self.assertTrue(np.all(probabilities[mask] > 0))

    def test_left_right_policy_transform_is_exact(self):
        reflection = (False, True)
        transformed = transform_state(self.game, reflection)
        for move in self.game.legal_moves():
            action = self.game.encode_move(move)
            new_action = transform_action(self.game, action, reflection)
            new_move = transform_move(move, self.game.board_size, reflection)
            self.assertEqual(transformed.decode(new_action), new_move)
            back = transform_action(transformed, new_action, reflection)
            self.assertEqual(back, action)

    def test_keras_save_load_preserves_predictions(self):
        if not KERAS_INSTALLED:
            self.skipTest("Keras not installed")
        network = GameNetwork(5, 8, 1)
        planes = canonical_planes(self.game)
        policy = np.zeros((1, self.game.action_size), dtype=np.float32)
        policy[0, self.game.legal_actions()[0]] = 1.0
        targets = {
            "policy": policy,
            "value": np.array([[1.0]], dtype=np.float32),
        }
        network.model.train_on_batch(planes[None, ...], targets)
        before = network.predict_raw(planes)
        optimizer_step = int(network.model.optimizer.iterations.numpy())

        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "model.keras")
            network.save(path)
            loaded = load_network(path)
            after = loaded.predict_raw(planes)
        np.testing.assert_allclose(before[0], after[0], atol=0.000001)
        self.assertAlmostEqual(before[1], after[1], places=6)
        loaded_step = int(loaded.model.optimizer.iterations.numpy())
        self.assertEqual(loaded_step, optimizer_step)

    def test_native_network_shapes_are_never_padded(self):
        if not KERAS_INSTALLED:
            self.skipTest("Keras not installed")
        small = GameNetwork(5, 8, 1)
        standard = GameNetwork(8, 8, 1)
        self.assertEqual(small.model.input_shape, (None, 5, 5, 2))
        self.assertEqual(standard.model.input_shape, (None, 8, 8, 2))
        self.assertEqual(small.model.output["policy"].shape[-1], 75)
        self.assertEqual(standard.model.output["policy"].shape[-1], 192)
        self.assertEqual(small.model.get_layer("policy").__class__.__name__, "Flatten")
        for layer in small.model.layers:
            if layer.__class__.__name__ == "Conv2D":
                self.assertTrue(layer.use_bias)


if __name__ == "__main__":
    unittest.main()
