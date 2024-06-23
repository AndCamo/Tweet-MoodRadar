import pandas as pd
import  random
import pprint
import matplotlib.pyplot as plt

LABEL_DICTIONARY = {0 : "negative", 4 : "positive"}

def createRandomDataset(dataset, rowsNumber):
   rowsForLabel = rowsNumber // dataset.iloc[:, 0].nunique()
   uniqueLabel = dataset.iloc[:, 0].unique()
   newDataset = []

   for label in uniqueLabel:
      tmp_data = dataset[dataset["sentiment"] == label]
      if len(tmp_data) < rowsForLabel:
         rows = len(tmp_data)
      else:
         rows = rowsForLabel
      tmp_data = tmp_data.sample(n=rows, ignore_index=False)
      for index, row in tmp_data.iterrows():
         tmp_row = [row[0], row[1]]
         newDataset.append(tmp_row)

   random.shuffle(newDataset)
   dataframe = pd.DataFrame(newDataset, columns=['sentiment', 'text'])
   dataframe.to_csv(f"/Users/andrea/Desktop/UNIVERSITÀ/CORSI/Machine Learning /Tweet_MoodRadar/dataset/final_dataset_short.csv", encoding='utf-8', index=False)

   return dataframe

def countLabelValues(dataset):
    columnName = "0"
    sentiments = dataset.iloc[:,0].unique()


    sentimentCount = {}

    total_count = 0
    for sentiment in sentiments:
        count = len(dataset[(dataset.iloc[:,0] == sentiment)])
        sentimentCount.update({sentiment : count})
        total_count += count

    return  sentimentCount



def splitDataset(dataset):
    tmp_train = []
    tmp_test = []

    dataset1 = dataset[dataset["sentiment"] == "negative"]
    dataset2 = dataset[dataset["sentiment"] == "positive"]
    pivot = int(.90 * len(dataset1))

    for i in range(0, pivot):
        tmp_row = [dataset1.iloc[i, 0], dataset1.iloc[i, 1]]
        tmp_train.append(tmp_row)
    for i in range(0, pivot):
        tmp_row = [dataset2.iloc[i, 0], dataset2.iloc[i, 1]]
        tmp_train.append(tmp_row)

    for i in range(pivot, len(dataset1)):
        tmp_row = [dataset1.iloc[i, 0], dataset1.iloc[i, 1]]
        tmp_test.append(tmp_row)
    for i in range(pivot, len(dataset1)):
        tmp_row = [dataset2.iloc[i, 0], dataset2.iloc[i, 1]]
        tmp_test.append(tmp_row)

    train_dataframe = pd.DataFrame(tmp_train, columns=["sentiment", "text"])
    test_dataframe = pd.DataFrame(tmp_test, columns=["sentiment", "text"])

    train_dataframe.to_csv(f"/Users/andrea/Desktop/UNIVERSITÀ/CORSI/Machine Learning /Tweet_MoodRadar/dataset/train_dataset.csv", encoding='utf-8', index=False)
    test_dataframe.to_csv(f"/Users/andrea/Desktop/UNIVERSITÀ/CORSI/Machine Learning /Tweet_MoodRadar/dataset/test_dataset.csv", encoding='utf-8', index=False)

def categorizeLabel(dataset):
    # Seach the int class and replace it with the String
    for index, row in dataset.iterrows():
        tmp_target = LABEL_DICTIONARY[row[0]]
        dataset.iloc[index, 0] = tmp_target

    # Update e save the new dataset
    return dataset


def removeLabels(dataset, labels):

    for label in labels:
        dataset.drop(dataset[dataset["sentiment"] == label].index, inplace=True)

    print("NELLA FUNZIONE")
    print(dataset["sentiment"].nunique())

    return dataset


def oversampleData(dataset, labels, amount):
    for label in labels:
        tmp_data = dataset[dataset["sentiment"] == label]
        duplicateRows = tmp_data.sample(n=amount, ignore_index=False, replace=True)

        for index, row in duplicateRows.iterrows():
            newRow = [row[0], row[1]]
            dataset.loc[len(dataset.index)] =newRow

    return dataset