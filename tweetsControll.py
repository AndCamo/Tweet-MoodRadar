import csv

import pandas as pd
from ntscraper import Nitter
import langid


def get_tweet_language(text):
    # Use landid library to get the language of the text
    lang, _ = langid.classify(text)

    return lang

def get_tweet_by_hashtag(hashtag, amount, mode):
    scraper = Nitter()
    scrapedTweets = scraper.get_tweets(hashtag, number=amount, mode="hashtag", instance="https://nitter.mint.lgbt")
    final_tweets = []
    for tweet in scrapedTweets["tweets"]:
        lang = get_tweet_language(tweet['text'])
        tmpTweet = [tweet['user']['name'], tweet['text'], lang, tweet['stats']['likes'], tweet['date'], tweet['link']]
        final_tweets.append(tmpTweet)


    dataset = pd.DataFrame(final_tweets, columns=['Name', 'Text', 'Language', 'Likes', 'Date', 'Link'])
    
    datasetName = f"{amount}_tweet_{hashtag}_[{mode}]"
    
    dataset.to_csv(f'/Users/andrea/Desktop/UNIVERSITÀ/CORSI/Machine Learning /Tweet_MoodRadar/tweets/{datasetName}.csv', encoding='utf-8', index=False)


def get_tweet_by_apify(dataset, name):
    tmpDataset = []
    for index, row in dataset.iterrows():
        if row["lang"] == "en":
            tmp_row = [row["user/name"], row["full_text"], row["lang"], row["views_count"], row["url"]]
            tmpDataset.append(tmp_row)

    newDataset = pd.DataFrame(tmpDataset, columns=['Name', 'Text', 'Language', 'Views', 'Link'])
    newDataset.to_csv(f'/Users/andrea/Desktop/UNIVERSITÀ/CORSI/Machine Learning /Tweet_MoodRadar/tweets/{name}_Tweets_Cleaned.csv', encoding='utf-8', index=False)


dataset = pd.read_csv("/Users/andrea/Desktop/UNIVERSITÀ/CORSI/Machine Learning /Tweet_MoodRadar/tweets/ChatGPT_Tweets_Cleaned.csv", on_bad_lines='skip', sep=";")

dataset.to_csv("/Users/andrea/Desktop/UNIVERSITÀ/CORSI/Machine Learning /Tweet_MoodRadar/tweets/ChatGPT_Tweets_Cleaned2.csv", sep=",", index=False)