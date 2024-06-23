
import pandas as pd
from matplotlib import pyplot as plt
from wordcloud import WordCloud, STOPWORDS

from model import datasetControll
from model import dataPreparation
from model import classifier
import pprint



DATASET_PATH = "/Users/andrea/Desktop/UNIVERSITÀ/CORSI/Machine Learning /Tweet_MoodRadar/dataset/sentiment140_dataset.csv"

# dataset = pd.read_csv(DATASET_PATH)
# datasetControll.splitDataset(dataset)


# ----TRAIN MODEL----
train_dataset = pd.read_csv("/Users/andrea/Desktop/UNIVERSITÀ/CORSI/Machine Learning /Tweet_MoodRadar/dataset/train_dataset.csv")
test_dataset = pd.read_csv("/Users/andrea/Desktop/UNIVERSITÀ/CORSI/Machine Learning /Tweet_MoodRadar/dataset/test_dataset.csv")

train_docs = dataPreparation.setupDocs(train_dataset)
test_docs = dataPreparation.setupDocs(test_dataset)

classifier.trainClassifier(train_docs, test_docs)

#---------------------------EVALUATE MODEL------------------------#

train_dataset = pd.read_csv(DATASET_PATH)
train_docs = dataPreparation.setupDocs(train_dataset)
x_test, y_test = classifier.getSplits(train_docs)
title = "Train"
classificatore, vectorizer = classifier.get_model();
classifier.evaluateClassifier(title,classificatore,vectorizer,x_test,y_test)