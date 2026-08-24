import os
import unittest


class SimplePythonTests(unittest.TestCase):
    def test_source_avoids_advanced_language_features(self):
        banned_text = [
            "from __future__",
            "dataclass",
            "from typing",
            "Protocol",
            "@property",
            "@staticmethod",
            "@classmethod",
            "__call__",
            " -> ",
            "lambda ",
            "from pathlib",
            "deque",
        ]
        test_directory = os.path.dirname(__file__)
        source_directory = os.path.join(
            os.path.dirname(test_directory), "src", "breakthrough_zero"
        )
        for filename in os.listdir(source_directory):
            if not filename.endswith(".py"):
                continue
            path = os.path.join(source_directory, filename)
            with open(path, "r", encoding="utf-8") as stream:
                source = stream.read()
            for text in banned_text:
                self.assertNotIn(text, source, filename + " contains " + text)


if __name__ == "__main__":
    unittest.main()
