'''
Created on 3 Aug 2013

@author: apuigdom
'''
import numpy as np
import pandas as pd
from features import getStats
from utils import loadModels, loadReModels
import pickle

class Repredictor:
    def getLookupTable(self, devices):
        lookupTable = np.zeros(1038)
        for index in range(len(devices)):
            lookupTable[devices[index]] = index
        return lookupTable
    
    def getTestStats(self):
        stats = []
        sequence = 0
        indexSeq = 0
        index = 0
        print "Calculating stats of the test"
        test = pd.read_csv('test.csv')
        test = test.values
        for element in range(len(test)):
            if test[element][4] != sequence:
                if sequence != 0:
                    print "Sequence "+str(index)
                    testSequence = np.array(getStats(test[indexSeq:element]))
                    stats.append(testSequence)
                    indexSeq = element
                    index += 1
                sequence = test[element][4]
                
        print "Sequence "+str(index)
        testSequence = np.array(getStats(test[indexSeq:len(test)]))
        stats.append(testSequence)
        stats = np.array(stats)
        np.save("testStats", stats)
        return stats
    
    def writePredictions(self, predictions):
        questions = pd.read_csv('questions.csv')
        idq = questions['QuestionId']
        results = open('submission.csv','w')
        results.write('QuestionId,IsTrue\n')
        for i in range(len(idq)):
            results.write(str(idq[i]) + "," + str(predictions[i]))
            results.write('\n')
    #Read questions
    
    def getPredictionModels(self, sequences):
        print "Model "+str(0)
        model = pickle.load(open("models/models"+str(0)+".mod"))
        modelPredictions = model.predict(sequences)
        predictions = np.array([[pred] for pred in modelPredictions])
        for i in range(1,387):
            print "Model "+str(i)
            model = pickle.load(open("models/models"+str(i)+".mod"))
            modelPredictions = model.predict(sequences)
            modelPredictions = np.array([[pred] for pred in modelPredictions])
            predictions = np.concatenate((predictions,modelPredictions), axis=1)
        np.save("subpredictions", predictions)
        return predictions
        
    def run(self, calculateStats=True, calculatePredictions=True):
        print "Loading test data..."
        sequences = np.load("testStats.npy") if not calculateStats else self.getTestStats()
        print "Loading devices..."
        devices = np.load("devices.npy")
        print "Getting lookup device table..."
        devicesIndexes = self.getLookupTable(devices)
        print "Loading questions..."
        questionFile = open("questions.csv","r")
        questions = questionFile.readlines()
        print "Getting intermediate predictions..."
        predictions = np.load("subpredictions.npy") if not calculatePredictions else self.getPredictionModels(sequences)
        print "loading remodels..."
        reModels = loadReModels()
        print "Going for the final predictions..."
        finalPredictions = []
        for element in range(len(predictions)):
                index = element + 1
                print "Sequence "+str(index)
                questionsLine = questions[index].split(',')
                modelNum = devicesIndexes[int(questionsLine[2])]
                reClassifier = reModels[int(modelNum)]
                prediction = reClassifier.predict(predictions[element])
                print "Prediction "+str(prediction)
                finalPredictions.append(prediction[0])
        newPredictions = np.array(finalPredictions)
        np.save("predictions", newPredictions)
        self.writePredictions(finalPredictions)
        
if __name__ == "__main__":
    repredictor = Repredictor()
    repredictor.run()