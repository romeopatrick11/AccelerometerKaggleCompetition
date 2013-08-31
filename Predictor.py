'''
Created on 3 Aug 2013

@author: apuigdom
'''
import numpy as np
import pickle
import pandas as pd
from features import getStats
from utils import loadModels, getLookupTable



def writePredictions(predictions):
    questions = pd.read_csv('questions.csv')
    idq = questions['QuestionId']
    results = open('submission.csv','w')
    results.write('QuestionId,IsTrue\n')
    for i in range(len(idq)):
        results.write(str(idq[i]) + "," + str(predictions[i]))
        results.write('\n')
#Read questions

print "Loading devices..."
devices = np.load("devices.npy")
print "Getting lookup device table..."
devicesIndexes = getLookupTable(devices)
print "Loading questions..."
questionFile = open("questions.csv","r")
questions = questionFile.readlines()
print "Loading test data..."
test = pd.read_csv('test.csv')
test = test.values
print "Loading models..."
models = loadModels()

index = 1
sequence = 0
indexSeq = 0
predictions = []
for element in range(len(test)):
    if test[element][4] != sequence:
        if sequence != 0:
            print "Sequence "+str(index)
            testSequence = np.array([getStats(test[indexSeq:element])])
            questionsLine = questions[index].split(',')
            if sequence != int(questionsLine[1]):
                raise "Something weird happens"
            modelNum = devicesIndexes[int(questionsLine[2])]
            classifier = models[int(modelNum)]
            prediction = classifier.predict(testSequence)
            print "Prediction "+str(prediction)
            predictions.append(prediction[0])
            indexSeq = element
            index += 1
        sequence = test[element][4]
        
print "Sequence "+str(index)
testSequence = np.array([getStats(test[indexSeq:len(test)])])
questionsLine = questions[index].split(',')
if sequence != int(questionsLine[1]):
    raise "Something weird happens"
modelNum = devicesIndexes[int(questionsLine[2])]
classifier = models[int(modelNum)]
prediction = classifier.predict(testSequence)
print "Prediction "+str(prediction)
predictions.append(prediction[0])
newPredictions = np.array(predictions)
np.save("predictions", newPredictions)
writePredictions(newPredictions)
