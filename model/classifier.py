import os

import numpy as np
import pandas as pd
import sys
import json
import random
import pprint

from matplotlib import pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay, roc_curve, auc

from . import dataPreparation

"""sys.path.insert(1, f'src{os.sep}ApplicationLayer')
import ApplicationLayer.ML_Model.dataset_manager as dataset_manager"""
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
import pickle

PATH_SEPARATOR = os.sep

LABEL_LIST = ['negative', 'positive']

def getSplits(docs):
    random.shuffle(docs)

    x_list = []  # list of text
    y_list = []  # list of corresponding labels

    for i in range(0, len(docs)):
        x_list.append(docs[i][1])
        y_list.append(docs[i][0])

    return x_list, y_list


def evaluateClassifierWeighted(title, classifier, vectorizer, x_list, y_list):
    x_test_tfidf = vectorizer.transform(x_list)
    y_pred = classifier.predict(x_test_tfidf)

    plotConfusionMatrix(y_list,y_pred)

    precision = metrics.precision_score(y_list, y_pred, average="macro")
    recall = metrics.recall_score(y_list, y_pred, average="macro")
    f1 = metrics.f1_score(y_list, y_pred, average="macro")

    print(title)
    print("--------------PRECISION--------------")
    print(precision)
    print("\n--------------RECALL--------------")
    print(recall)
    print("\n--------------F1_SCORE--------------")
    print(f1)


def evaluateClassifier(title, classifier, vectorizer, x_list, y_list):
    x_test_tfidf = vectorizer.transform(x_list)
    y_pred = classifier.predict(x_test_tfidf)

    precision = metrics.precision_score(y_list, y_pred, labels=LABEL_LIST, average=None)
    recall = metrics.recall_score(y_list, y_pred, labels=LABEL_LIST, average=None)
    f1 = metrics.f1_score(y_list, y_pred, labels=LABEL_LIST, average=None)

    precision_dic = createDic(precision.tolist())
    recall_dic = createDic(recall.tolist())
    f1_dic = createDic(f1.tolist())

    print(title)
    print("--------------PRECISION--------------\n")
    pprint.pprint(precision_dic, indent=2)
    print("\n--------------RECALL--------------\n")
    pprint.pprint(recall_dic, indent=2)
    print("\n--------------F1_SCORE--------------\n")
    pprint.pprint(f1_dic, indent=2)

def plotConfusionMatrix(y_list, pred):
   confusion_mat = metrics.confusion_matrix(y_list, pred, labels=LABEL_LIST)
   disp = ConfusionMatrixDisplay(confusion_mat, display_labels=LABEL_LIST)

   
   disp.plot(values_format='d', xticks_rotation='vertical') 
   plt.show()

def plot_roc_curve(test_docs):
    # Load the trained classifier and vectorizer
    clf_filename = "/Users/andrea/Desktop/UNIVERSITÀ/CORSI/Machine Learning /Tweet_MoodRadar/src/naive_bayes_classiefier.pkl"
    vec_filename = "/Users/andrea/Desktop/UNIVERSITÀ/CORSI/Machine Learning /Tweet_MoodRadar/src/count_vectorizer.pkl"
    naiveBayesClassifier = pickle.load(open(clf_filename, 'rb'))
    vectorizer = pickle.load(open(vec_filename, 'rb'))

    # Transform test data into document-term matrix
    x_test, y_test = getSplits(test_docs)
    dtm_test = vectorizer.transform(x_test)

    # Predict probabilities of positive class
    y_prob = naiveBayesClassifier.predict_proba(dtm_test)[:, 1]

    # Compute ROC curve and ROC area for each class
    fpr, tpr, _ = roc_curve(y_test, y_prob, pos_label="positive")
    roc_auc = auc(fpr, tpr)

    # Plot ROC curve
    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic Curve')
    plt.legend(loc="lower right")
    plt.show()

def createDic(lista):
    dictionary = {}
    i = 0
    for label in LABEL_LIST:
        dictionary.update({label: lista[i]})
        i = i + 1

    return dictionary


def trainClassifier(train_docs, test_docs):
    x_train, y_train = getSplits(train_docs)
    x_test, y_test = getSplits(test_docs)

    vectorizer = CountVectorizer(stop_words='english', ngram_range=(1, 3), min_df=3, analyzer='word')

    # create doc-term matrix
    dtm = vectorizer.fit_transform(x_train)

    # Train the Naive Bayes Classifier
    naiveBayesClassifier = MultinomialNB().fit(dtm, y_train)
    evaluateClassifier("Naive Bayes\tTRAIN\t\n", naiveBayesClassifier, vectorizer, x_train, y_train)
    evaluateClassifier("Naive Bayes\tTEST\t\n", naiveBayesClassifier, vectorizer, x_test, y_test)

    evaluateClassifierWeighted("Naive Bayes\tTRAIN\t\n", naiveBayesClassifier, vectorizer, x_train, y_train)
    evaluateClassifierWeighted("Naive Bayes\tTEST\t\n", naiveBayesClassifier, vectorizer, x_test, y_test)

    # store the classifier
    clf_filename = 'naive_bayes_classiefier.pkl'
    pickle.dump(naiveBayesClassifier, open(clf_filename, 'wb'))

    # also sotre the vectorizer so we can transform new data
    vec_filename = 'count_vectorizer.pkl'
    pickle.dump(vectorizer, open(vec_filename, 'wb'))


def get_model():
    # load classifier
    clf_filename = "/Users/andrea/Desktop/UNIVERSITÀ/CORSI/Machine Learning /Tweet_MoodRadar/src/naive_bayes_classiefier.pkl"
    classifier = pickle.load(open(clf_filename, 'rb'))

    # vectorize the new text
    vec_filename = "/Users/andrea/Desktop/UNIVERSITÀ/CORSI/Machine Learning /Tweet_MoodRadar/src/count_vectorizer.pkl"
    vectorizer = pickle.load(open(vec_filename, 'rb'))

    return classifier, vectorizer


def get_prediction(text, classifier, vectorizer):
    cleanedText = dataPreparation.cleanText(text)
    prediction = classifier.predict(vectorizer.transform([cleanedText]))
    probabilityList = classifier.predict_proba(vectorizer.transform([text]))
    tmpList = probabilityList.astype(float).round(3)
    sorted_array = np.sort(tmpList[0])[::-1]

    label = prediction[0]
    probability = sorted_array[0]

    if probability <= 0.6:
        label = "neutral"

    return label, probability 