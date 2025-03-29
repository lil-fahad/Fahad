
import pandas as pd
from tsfresh import extract_features
from tsfresh.utilities.dataframe_functions import impute

class TimeSeriesFeatureEngineer:
    def __init__(self):
        pass

    def transform(self, df, column_id='id', column_sort='time'):
        features = extract_features(df, column_id=column_id, column_sort=column_sort)
        impute(features)
        return features
