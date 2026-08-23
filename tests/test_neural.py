from __future__ import annotations

import importlib.util
from pathlib import Path
import tempfile
import unittest

import numpy as np

from breakthrough_zero.game import Breakthrough, PLAYER_1, PLAYER_2
from breakthrough_zero.neural import (
    GameNetwork,
    NeuralBoundary,
    canonical_planes,
    masked_softmax,
)
from breakthrough_zero.symmetry import Symmetry


class RecordingNetwork:
    def __init__(self, action_size: int, value: float = 0.4) -> None:
        self.action_size = action_size
        self.value = value
        self.inputs: list[np.ndarray] = []

    def predict_raw(self, planes: np.ndarray) -> tuple[np.ndarray, float]:
        self.inputs.append(planes.copy())
        return np.linspace(-1, 1, self.action_size), self.value


class NeuralBoundaryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.game = Breakthrough.from_rows(
            ["1....", ".1.2.", "..1..", ".2...", "....2"],
            player_to_move=PLAYER_1,
        )

    def test_swapped_and_rotated_positions_have_identical_inputs(self) -> None:
        swapped = Symmetry(swap_players=True).state(self.game)
        np.testing.assert_array_equal(canonical_planes(self.game), canonical_planes(swapped))

    def test_identical_canonical_inputs_have_identical_raw_predictions(self) -> None:
        swapped = Symmetry(swap_players=True).state(self.game)
        network = RecordingNetwork(self.game.action_size)
        first = network.predict_raw(canonical_planes(self.game))
        second = network.predict_raw(canonical_planes(swapped))
        np.testing.assert_array_equal(first[0], second[0])
        self.assertEqual(first[1], second[1])
        np.testing.assert_array_equal(network.inputs[0], network.inputs[1])

    def test_wrapper_opposes_absolute_values_for_swapped_players(self) -> None:
        network = RecordingNetwork(self.game.action_size, value=0.35)
        boundary = NeuralBoundary(network)
        swapped = Symmetry(swap_players=True).state(self.game)
        self.assertAlmostEqual(boundary.predict(self.game).value, 0.35)
        self.assertAlmostEqual(boundary.predict(swapped).value, -0.35)

    def test_all_absolute_targets_convert_relative(self) -> None:
        for target in (-1.0, -0.5, 0.0, 0.25, 1.0):
            self.assertEqual(
                NeuralBoundary.relative_target(target, PLAYER_1), target
            )
            self.assertEqual(
                NeuralBoundary.relative_target(target, PLAYER_2), -target
            )

    def test_illegal_actions_are_masked_exactly(self) -> None:
        logits = np.arange(self.game.action_size, dtype=float)
        mask = self.game.legal_action_mask()
        probabilities = masked_softmax(logits, mask)
        self.assertAlmostEqual(float(probabilities.sum()), 1.0)
        self.assertTrue(np.all(probabilities[~mask] == 0))
        self.assertTrue(np.all(probabilities[mask] > 0))

    def test_left_right_policy_transform_is_exact(self) -> None:
        reflect = Symmetry(reflect_left_right=True)
        transformed = reflect.state(self.game)
        for move in self.game.legal_moves():
            action = self.game.encode_move(move)
            reflected_action = reflect.action(self.game, action)
            expected_move = reflect.move(move, self.game.board_size)
            self.assertEqual(transformed.decode(reflected_action), expected_move)
            self.assertEqual(reflect.action(transformed, reflected_action), action)

    @unittest.skipUnless(importlib.util.find_spec("tensorflow"), "TensorFlow not installed")
    def test_keras_save_load_preserves_predictions(self) -> None:
        network = GameNetwork(5, filters=8, residual_blocks=1)
        planes = canonical_planes(self.game)
        policy = np.zeros((1, self.game.action_size), dtype=np.float32)
        policy[0, self.game.legal_actions()[0]] = 1.0
        network.model.train_on_batch(
            planes[None, ...],
            {"policy": policy, "value": np.asarray([[1.0]], dtype=np.float32)},
        )
        before = network.predict_raw(planes)
        optimizer_step = int(network.model.optimizer.iterations.numpy())
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.keras"
            network.save(path)
            loaded = GameNetwork.load(path)
            after = loaded.predict_raw(planes)
        np.testing.assert_allclose(before[0], after[0], atol=1e-6)
        self.assertAlmostEqual(before[1], after[1], places=6)
        self.assertEqual(int(loaded.model.optimizer.iterations.numpy()), optimizer_step)

    @unittest.skipUnless(importlib.util.find_spec("tensorflow"), "TensorFlow not installed")
    def test_native_network_shapes_are_never_padded(self) -> None:
        small = GameNetwork(5, filters=8, residual_blocks=1)
        standard = GameNetwork(8, filters=8, residual_blocks=1)
        self.assertEqual(small.model.input_shape, (None, 5, 5, 2))
        self.assertEqual(standard.model.input_shape, (None, 8, 8, 2))
        self.assertEqual(small.model.output["policy"].shape[-1], 75)
        self.assertEqual(standard.model.output["policy"].shape[-1], 192)


if __name__ == "__main__":
    unittest.main()
