
import requests
import datetime

class NewsFetcher:
    def __init__(self, api_key):
        self.api_key = api_key
        self.endpoint = "https://newsapi.org/v2/everything"

    def fetch_news(self, query, from_days=3, language="en", page_size=50):
        from_date = (datetime.datetime.now() - datetime.timedelta(days=from_days)).strftime("%Y-%m-%d")
        params = {
            "q": query,
            "from": from_date,
            "language": language,
            "sortBy": "publishedAt",
            "pageSize": page_size,
            "apiKey": self.api_key
        }
        response = requests.get(self.endpoint, params=params)
        if response.status_code == 200:
            articles = response.json().get("articles", [])
            return [article["title"] + " " + article.get("description", "") for article in articles]
        else:
            raise Exception(f"NewsAPI Error: {response.status_code} - {response.text}")
