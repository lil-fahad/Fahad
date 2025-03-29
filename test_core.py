
import unittest
from core.auto_predictor import run_prediction

class TestPrediction(unittest.TestCase):
    def test_run_prediction_output(self):
        results = run_prediction(symbol="AAPL", model_choice="hybrid")
        self.assertIsInstance(results, list)
        self.assertIn("model", results[0])
        self.assertIn("mae", results[0])
        self.assertIn("direction_accuracy", results[0])

if __name__ == "__main__":
    unittest.main()
