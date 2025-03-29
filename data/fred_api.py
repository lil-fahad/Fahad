
import requests

class FREDClient:
    def __init__(self, api_key):
        self.api_key = api_key
        self.endpoint = "https://api.stlouisfed.org/fred/series/observations"

    def fetch_series(self, series_id, start_date="2020-01-01", end_date=None):
        if end_date is None:
            from datetime import datetime
            end_date = datetime.today().strftime('%Y-%m-%d')

        params = {
            "series_id": series_id,
            "api_key": self.api_key,
            "file_type": "json",
            "observation_start": start_date,
            "observation_end": end_date
        }

        response = requests.get(self.endpoint, params=params)
        if response.status_code == 200:
            data = response.json().get("observations", [])
            return [(obs["date"], float(obs["value"])) for obs in data if obs["value"] != "."]
        else:
            raise Exception(f"FRED API Error: {response.status_code} - {response.text}")
