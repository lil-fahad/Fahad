import logging
logging.basicConfig(level=logging.INFO)
from data.polygon_api import fetch_polygon_data

def load_data(symbol='AAPL', from_date='2022-01-01', to_date='2024-12-31'):
    try:
        data = fetch_polygon_data(symbol, from_date, to_date)
        if data is None or data.empty:
            logging.warning(f'No data fetched for symbol {symbol}.')
            return None
        return data
    except Exception as e:
        logging.exception(f'Error loading data for symbol {symbol}')
        return None
