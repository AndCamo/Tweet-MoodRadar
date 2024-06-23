import eel
import os
import time
import pandas as pd
import tweetsControll
import json
import analysisControl

dirname = os.path.dirname(__file__)
eel.init(os.path.join(dirname, "app/"))

@eel.expose
def downloadTweet(query, amount, mode):
    print(f"Stai cercando {query} in mdoalità {mode}")
    tweetsControll.get_tweet_by_hashtag(query, amount, mode)

    print("fatto anche in python")

@eel.expose
def startAnalysis(filename):
    try:
        dataset =  pd.read_csv(f"tweets/{filename}", on_bad_lines='skip')
        classification = analysisControl.classifyTweets(dataset)
        tweetAnalysis = []


        for index, row in classification.iterrows():
            tmpObj = {
                "Username" : row["Username"],
                "Text" : row["Text"],
                "Emotion" : row["Emotion"],
                "Views" : row["Views"],
                "Link" : row["Link"]
            }
            tweetAnalysis.append(tmpObj)

        response = {
            "status" : "success",
            "classification" : tweetAnalysis
        }
        return json.dumps(response)

    except FileNotFoundError:
        response = {
            "status" : "error",
            "classification" : []
        }
        return json.dumps(response)

eel.start("pages/homepage.html")
