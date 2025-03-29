import logging
logging.basicConfig(level=logging.INFO)
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

def evaluate_predictions(y_true, y_pred):
    """
    وظيفة: evaluate_predictions
    """
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": mean_squared_error(y_true, y_pred, squared=False),
        "R2": r2_score(y_true, y_pred)
    }

def direction_accuracy(y_true, y_pred):
    """
    وظيفة: direction_accuracy
    """
    return np.mean(np.sign(np.diff(y_true)) == np.sign(np.diff(y_pred))) * 100