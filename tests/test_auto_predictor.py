import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import unittest
from core.auto_predictor import run_prediction

class TestAutoPredictor(unittest.TestCase):

    def test_prediction_output(self):
        result = run_prediction("AAPL", model_choice="hybrid")
        self.assertIsInstance(result, list)
        self.assertGreater(len(result), 0)
        self.assertIn("mae", result[0])
        self.assertIn("direction_accuracy", result[0])
        self.assertGreaterEqual(result[0]["direction_accuracy"], 0)

    def test_invalid_symbol(self):
        result = run_prediction("INVALID123")
        self.assertIsInstance(result, list)
        self.assertIn("error", result[0])

if __name__ == '__main__':
    unittest.main()