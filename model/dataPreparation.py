import pandas as pd
import nltk
import re
from collections import defaultdict
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, SnowballStemmer
import string
import emoji



STOP_WORDS = set(stopwords.words('english'))
STOP_WORDS.remove("not")


def setupStopwords():
    STOP_WORDS.update(["like", "im", "feel", "know", "time", "go"])
    moreStopwords = []
    with open("/Users/andrea/Desktop/UNIVERSITÀ/CORSI/Machine Learning /Tweet_MoodRadar/src/model/assets/stopwords.txt", 'rb') as file:
        text = file.read().decode(errors='replace')
        moreStopwords = text.split("\n")

    STOP_WORDS.update(moreStopwords)

setupStopwords()
def tokenize(text):
    tokens = nltk.word_tokenize(text)

    tokens = [token for token in tokens if not token in STOP_WORDS]

    return tokens

def removeMentionAndHashtag(testo):
    parole = testo.split(" ")
    parole_filtrate = [parola for parola in parole if not (parola.startswith("@") or parola.startswith("#"))]
    testo_senza_mention = ' '.join(parole_filtrate)
    return testo_senza_mention

def cleanText(text):
    mapping_table = str.maketrans({'\n': '', '\t': ''})

    text = text.lower()
    # Remove Tag and Hashtag
    text = removeMentionAndHashtag(text)
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Remove numbers
    text = text.translate(str.maketrans('', '', string.digits))
    # Remove Links
    text = re.sub(r'http\S+', '', text, flags=re.MULTILINE)
    # Remove Emojis
    text = emoji.replace_emoji(text, replace="")


    text = text.translate(mapping_table)

    text = stemmer(text)

    return text


def stemmer(text):
    word_tokenize = tokenize(text)

    snowStemmer = SnowballStemmer("english")
    textStemmed = " ".join([snowStemmer.stem(words) for words in word_tokenize])

    return textStemmed


def setupDocs(dataset):
    docs = []  # (label, text)
    for index, row in dataset.iterrows():
        text = cleanText(str(row[1]))
        label = row[0]
        doc = (label, text.strip())

        docs.append(doc)

    return docs


def printFrequencyDist(docs):
    # create the data structure to contain the frequency
    tokens = defaultdict(list)

    # list of all the words for each category
    for doc in docs:
        docLabel = doc[0]
        docText = doc[1]

        docTokens = tokenize(docText)

        # update the dictionary with the frequency of the tokens in the class
        tokens[docLabel].extend(docTokens)

    for categoryLabel, categoryTokens in tokens.items():
        print(categoryLabel)
        fd = nltk.FreqDist(categoryTokens)
        print(fd.most_common(20))
        print(" ")


def setUpDocsNoClean(dataset):
    docs = []  # (label, text)
    for index, row in dataset.iterrows():
        text = row[1]
        label = row[0]
        doc = (label, text.strip())

        docs.append(doc)

    return docs
