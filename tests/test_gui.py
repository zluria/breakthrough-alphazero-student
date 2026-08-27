import os
import tempfile
import unittest

from breakthrough_zero.gui import checkpoint_choices, find_checkpoints


class GUIConfigurationTests(unittest.TestCase):
    def test_checkpoint_discovery_finds_keras_models_recursively(self):
        with tempfile.TemporaryDirectory() as directory:
            nested = os.path.join(directory, "accepted")
            os.makedirs(nested)
            paths = [
                os.path.join(directory, "start.keras"),
                os.path.join(nested, "iteration-0015.h5"),
                os.path.join(nested, "notes.txt"),
            ]
            for path in paths:
                with open(path, "w", encoding="utf-8") as stream:
                    stream.write("test")

            found = find_checkpoints(directory)

            self.assertEqual(
                found,
                sorted([os.path.abspath(paths[0]), os.path.abspath(paths[1])]),
            )

    def test_checkpoint_choices_show_relative_names(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "accepted", "iteration-0096.keras")
            choices = checkpoint_choices([path], directory)
            self.assertEqual(
                choices,
                {"Neural: accepted/iteration-0096": os.path.abspath(path)},
            )


if __name__ == "__main__":
    unittest.main()
