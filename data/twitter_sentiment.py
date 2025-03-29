
import snscrape.modules.twitter as sntwitter
from transformers import pipeline
import datetime

class TwitterSentimentAnalyzer:
    def __init__(self, model_name="distilbert-base-uncased-finetuned-sst-2-english"):
        self.sentiment_pipeline = pipeline("sentiment-analysis", model=model_name)

    def fetch_tweets(self, query, max_tweets=50, since_days=2):
        since_date = (datetime.datetime.now() - datetime.timedelta(days=since_days)).strftime("%Y-%m-%d")
        tweets = []
        for i, tweet in enumerate(sntwitter.TwitterSearchScraper(f"{query} since:{since_date}").get_items()):
            if i >= max_tweets:
                break
            tweets.append(tweet.content)
        return tweets

    def analyze_sentiment(self, tweets):
        return self.sentiment_pipeline(tweets)

    def fetch_and_analyze(self, query, max_tweets=50):
        tweets = self.fetch_tweets(query, max_tweets=max_tweets)
        sentiments = self.analyze_sentiment(tweets)
        return list(zip(tweets, sentiments))
