import logging
logging.basicConfig(level=logging.INFO)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

def build_bayesian_lstm(input_shape):
    """
    Build and compile a Bayesian LSTM model.
    """
    try:
        model = Sequential()
        model.add(LSTM(64, return_sequences=True, input_shape=input_shape, dropout=0.3, recurrent_dropout=0.3))
        model.add(LSTM(64, dropout=0.3, recurrent_dropout=0.3))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mse')
        logging.info('Bayesian LSTM model compiled successfully.')
        return model
    except Exception as e:
        logging.exception('Error building Bayesian LSTM model')
        return None
