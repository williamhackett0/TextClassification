#Feature Selection and Transforms
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

#Classification Models
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

#Evaluation
from sklearn.metrics import recall_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score

#Dataset functions
from sklearn.model_selection import train_test_split
from nltk.stem import PorterStemmer
import nltk
import pandas as pd
import numpy as np

#
# Pre-Process Processing function
#
def PreProcessDataset(dataset):
    #Map labels to integer values for classifier
    dataset['Label'] = dataset.Label.map({'Software / IT': 0, 'Regulations': 1, 'Support & Sales': 2})

    #Change all characters to lower case
    dataset['Ticket'] = dataset.Ticket.map(lambda a: a.lower())

    #Remove punctionation
    dataset['Ticket'] = dataset.Ticket.str.replace('[^\w\s]', '')

    #Apply tokenisation
    dataset['Ticket'] = dataset['Ticket'].apply(nltk.word_tokenize)

    #Apply stemmer. Removes past, current, future tense
    stemmer = PorterStemmer()
    dataset['Ticket'] = dataset['Ticket'].apply(lambda x: [stemmer.stem(y) for y in x])

    #Transform data into occurrences
    dataset['Ticket'] = dataset['Ticket'].apply(lambda x: ' '.join(x))

    #Create Vector and Fit data
    vector = CountVectorizer()
    counts = vector.fit_transform(dataset['Ticket'])

    #Term Frequency Inverse Document Frequency
    transformer = TfidfTransformer().fit(counts)
    counts = transformer.transform(counts)

    return dataset, counts

#
# Evaluation function
#
def Evaluation(name, predicted, actual):
    print("-----------------------", name, "-----------------------")
    
    #Matrix
    ConfusionMatrix = confusion_matrix(actual['Label'], predicted)
    print(ConfusionMatrix)

    #Accuracy
    Accuracy = round(np.mean(predicted == actual['Label']), 4)
    print("Accuracy:", Accuracy)

    #Recall
    Recall = round(recall_score(actual['Label'], predicted, average='weighted'), 4)
    print("Recall:", Recall)

    #Precision
    Precision = round(precision_score(actual['Label'], predicted, average='weighted'), 4)
    print("Precision:", Precision)

    #F1
    F1Score = round(f1_score(actual['Label'], predicted, average='weighted'), 4)
    print("F1:", F1Score)
    
#Load, Split, and Pre-Process Dataset
dataset, datasetCounts = PreProcessDataset(pd.read_excel('Dataset.xlsx', index_col=None))
trainingData, testingData, trainingCounts, testCounts = train_test_split(dataset, datasetCounts,
                                                                         test_size = 0.33, random_state = 0)

#Train Models

nvb = MultinomialNB().fit(trainingCounts, trainingData['Label'])
dec = DecisionTreeClassifier().fit(trainingCounts, trainingData['Label'])
ran = RandomForestClassifier(n_estimators=100).fit(trainingCounts, trainingData['Label'])
svmClf = SVC().fit(trainingCounts, trainingData['Label'])

#Test Models

nvbPredicted = nvb.predict(testCounts)
decPredicted = dec.predict(testCounts)
ranPredicted = ran.predict(testCounts)
svmPredicted = svmClf.predict(testCounts)

#Perform evaluation
Evaluation("Naive Bayes", nvbPredicted, testingData)
Evaluation("Decision Teee", decPredicted, testingData)
Evaluation("Random Forest", ranPredicted, testingData)
Evaluation("Support Vector Machine", svmPredicted, testingData)

