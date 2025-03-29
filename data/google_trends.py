
from pytrends.request import TrendReq
import pandas as pd

class GoogleTrendsFetcher:
    def __init__(self, hl='en-US', tz=360):
        self.pytrends = TrendReq(hl=hl, tz=tz)

    def fetch_trend_data(self, keyword, timeframe='now 7-d', geo='US'):
        self.pytrends.build_payload([keyword], cat=0, timeframe=timeframe, geo=geo, gprop='')
        data = self.pytrends.interest_over_time()
        if 'isPartial' in data.columns:
            data = data.drop(columns=['isPartial'])
        return data
