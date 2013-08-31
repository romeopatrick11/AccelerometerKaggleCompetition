'''
Created on 31 Jul 2013

@author: apuigdom
'''
import pandas as pd
import numpy as np
from sklearn.ensemble.forest import RandomForestClassifier
from sklearn.svm import libsvm
from features import getStats
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression

def getSeparationIndexes(train):
    index = 0
    sequence = 0
    indexList = []
    for element in range(len(train)):
        if train['Device'][element] != sequence:
            index = element
            indexList.append(index)
            print index    
            sequence = train['Device'][element]
    print "hello"
    return np.array(indexList)
    
def getTrainData(splittedData, i):
    toReturn = np.delete(splittedData, i, 0)
    toReturn = tuple(tuple(x) for x in toReturn)
    toReturn = np.concatenate(toReturn)
    return toReturn

def getAttributes(train, indexes, goal):
    #Generate 2000 random 300 long samples
    size = 300
    samples = 3000
    trainVect = []
    targetVect = []
    np.random.seed(4)
    beginning = indexes[goal]
    if goal == len(indexes)-1:
        end = len(train)
    else:
        end = indexes[goal+1]
    print "2500 positive sequences"
    for _ in range(samples/2):
        number = np.random.choice(range(beginning,end-size))
        trainVect.append(getStats(train[number:number+size]))
        targetVect.append(1)
    print "2500 negative sequences"
    for _ in range(samples/2):
        indexNumber = np.random.choice(range(len(indexes)))
        while indexNumber == goal:
            indexNumber = np.random.choice(range(len(indexes)))
        beginning = indexes[indexNumber]
        if indexNumber == len(indexes)-1:
            end = len(train)
        else:
            end = indexes[indexNumber+1]
        number = np.random.choice(range(beginning,end-size))
        trainVect.append(getStats(train[number:number+size]))
        targetVect.append(0)
    return (np.array(trainVect), np.array(targetVect))
        
    

print "Reading the training data..."
train = pd.read_csv('train.csv')
print "Reading device separations..."
indexes = np.load("indexesTrain.npy")
print "Getting attributes..."
(trainVect, targetVect) = getAttributes(train, indexes, 150)

indexes = np.arange(len(trainVect))
np.random.shuffle(indexes)
newTrain = trainVect[indexes]
newTarget = targetVect[indexes]

newTrain = np.split(newTrain, 8)
newTarget = np.split(newTarget,8)

trainData = getTrainData(newTrain,0)
trainTarget = getTrainData(newTarget,0)

testData = newTrain[0]
testTarget = newTarget[0]

classifier = RandomForestClassifier(n_estimators=500, verbose=2, n_jobs=1, random_state=1)
#classifier = LogisticRegression()
classifier.fit(trainData, trainTarget)
predictions = classifier.predict(testData)
TP = 0
FP = 0
FN = 0
TN = 0
for i in range(len(predictions)):
    if predictions[i] == 1 and testTarget[i] == 1:
        TP += 1
    if predictions[i] == 0 and testTarget[i] == 0:
        TN += 1
    if predictions[i] == 0 and testTarget[i] == 1:
        FN += 1
    if predictions[i] == 1 and testTarget[i] == 0:
        FP += 1

print TP
print FP
print TN
print FN
