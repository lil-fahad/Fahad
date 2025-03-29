import logging
logging.basicConfig(level=logging.INFO)
from models.lstm_advanced import build_lstm_advanced
import xgboost as xgb
import numpy as np

def hybrid_model_predict(seq_data, tabular_data):
    """
    Predict using a hybrid LSTM and XGBoost model.
    """
    try:
        # Validate input data
        if seq_data is None or len(seq_data) == 0:
            raise ValueError('Sequential data is empty or None.')
        if tabular_data is None or len(tabular_data) == 0:
            raise ValueError('Tabular data is empty or None.')

        lstm_model = build_lstm_advanced(seq_data.shape[1:])
        if lstm_model is None:
            raise ValueError('Failed to build LSTM model.')
        lstm_model.fit(seq_data, np.mean(seq_data, axis=2), epochs=3, verbose=0)
        lstm_preds = lstm_model.predict(seq_data)

        xgb_model = xgb.XGBRegressor()
        xgb_model.fit(tabular_data, lstm_preds.flatten())
        final_preds = xgb_model.predict(tabular_data)

        logging.info('Hybrid model prediction completed successfully.')
        return final_preds
    except Exception as e:
        logging.exception('Error during hybrid model prediction')
        return np.array([])
