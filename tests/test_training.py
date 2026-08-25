import unittest

from breakthrough_zero.training import (
    search_limits,
    split_training_and_validation,
)


class TrainingLoopTests(unittest.TestCase):
    def test_progressive_search_schedule(self):
        config = {
            "search_schedule": [
                {
                    "until_iteration": 2,
                    "full_simulations": 8,
                    "fast_simulations": 2,
                },
                {
                    "until_iteration": 4,
                    "full_simulations": 16,
                    "fast_simulations": 4,
                },
            ]
        }
        self.assertEqual(search_limits(config, 0), (8, 2))
        self.assertEqual(search_limits(config, 1), (8, 2))
        self.assertEqual(search_limits(config, 2), (16, 4))

    def test_validation_holds_out_whole_games(self):
        records = []
        for game_index in range(10):
            records.append({"game_index": game_index, "position": 0})
            records.append({"game_index": game_index, "position": 1})
        training, validation = split_training_and_validation(records, 0.2)

        training_games = set()
        validation_games = set()
        for record in training:
            training_games.add(record["game_index"])
            self.assertFalse(record["validation"])
        for record in validation:
            validation_games.add(record["game_index"])
            self.assertTrue(record["validation"])

        self.assertEqual(len(validation_games), 2)
        self.assertFalse(training_games & validation_games)
        self.assertEqual(len(training) + len(validation), 20)


if __name__ == "__main__":
    unittest.main()
