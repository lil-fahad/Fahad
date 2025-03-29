import logging
logging.basicConfig(level=logging.INFO)
from sklearn.metrics import mean_absolute_error
import numpy as np

def select_best_model(model_dict, X_seq, X_tabular, y_true):
    """
    Select the best model based on mean absolute error (MAE).
    """
    try:
        results = []
        if not model_dict:
            raise ValueError('Model dictionary is empty.')
        if len(X_seq) == 0 or len(X_tabular) == 0 or len(y_true) == 0:
            raise ValueError('Input data for model selection is empty.')

        for name, model_func in model_dict.items():
            try:
                preds = model_func(X_seq, X_tabular)
                if len(preds) != len(y_true):
                    raise ValueError(f'Predictions length mismatch for model {name}.')
                mae = mean_absolute_error(y_true, preds)
                direction_acc = np.mean(np.sign(np.diff(y_true)) == np.sign(np.diff(preds))) * 100
                results.append({
                    'model': name,
                    'mae': mae,
                    'direction_accuracy': direction_acc
                })
            except Exception as e:
                logging.error(f'Error evaluating model {name}: {e}')

        if not results:
            raise ValueError('No valid model results found.')
        logging.info('Model selection completed successfully.')
        return sorted(results, key=lambda x: x['mae'])
    except Exception as e:
        logging.exception('Error during model selection')
        return []
