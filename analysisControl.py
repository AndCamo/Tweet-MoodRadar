import tweetsControll
from model import classifier
import pandas as pd
from model import datasetControll
from model import dataPreparation

#tweetsControll.get_tweet_by_hashtag("ChampionsLeague", 100, "hashtag")



def classifyTweets(tweets):
    classifierNB, vectorizer = classifier.get_model()

    newDataset = []

    for index, row in tweets.iterrows():
        text = row[1]
        prediction, probability = classifier.get_prediction(text, classifierNB, vectorizer)

        newRow = [row["Name"], row[1], prediction, probability, row["Views"], row["Link"]]
        newDataset.append(newRow)

    finalDataset = pd.DataFrame(newDataset, columns=["Username", "Text", "Emotion", "Probability", "Views", "Link"])
    finalDataset.to_csv("prediction_result/predictedOutput.csv", encoding='utf-8', index=False)
    return finalDataset